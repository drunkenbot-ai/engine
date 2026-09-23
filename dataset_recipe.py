"""Dynamic dataset recipe data models, presets, and normalization engine.

Enables fine-grained, dynamic control over pre-training and fine-tuning dataset
category mixtures without hardcoding ratios or static assumptions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class RecipeCategory:
    """Configuration for a single dataset category within a recipe."""

    name: str
    slug: str
    target_percentage: float  # 0.0 to 100.0
    enabled: bool = True
    locked: bool = False
    source_paths: list[str] = field(default_factory=list)
    tokens_estimated: int = 0
    is_builtin: bool = True
    files_count: int = 0
    disk_bytes: int = 0
    disk_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecipeCategory:
        return cls(
            name=str(data.get("name", "Untitled")),
            slug=str(data.get("slug", "untitled")),
            target_percentage=float(data.get("target_percentage", 0.0)),
            enabled=bool(data.get("enabled", True)),
            locked=bool(data.get("locked", False)),
            source_paths=[str(p) for p in data.get("source_paths", [])],
            tokens_estimated=int(data.get("tokens_estimated", 0)),
            is_builtin=bool(data.get("is_builtin", False)),
            files_count=int(data.get("files_count", 0)),
            disk_bytes=int(data.get("disk_bytes", 0)),
            disk_tokens=int(data.get("disk_tokens", 0)),
        )


@dataclass
class DatasetRecipe:
    """A complete mixture recipe configuring proportions across all categories."""

    recipe_id: str
    name: str
    description: str = ""
    total_target_tokens: int = 250_000_000
    categories: list[RecipeCategory] = field(default_factory=list)

    def total_percentage(self) -> float:
        """Return the sum of target percentages of all enabled categories."""
        return sum(c.target_percentage for c in self.categories if c.enabled)

    def is_balanced(self, tolerance: float = 0.01) -> bool:
        """Check if the enabled categories sum to 100% within tolerance."""
        return abs(self.total_percentage() - 100.0) <= tolerance

    def get_category(self, slug: str) -> Optional[RecipeCategory]:
        """Find category by slug."""
        for c in self.categories:
            if c.slug == slug:
                return c
        return None

    def normalize_to_100(self) -> None:
        """Proportionally rebalance unlocked, enabled categories so total is exactly 100.0%.

        Locked categories retain their exact values. If all enabled categories are locked,
        normalization cannot adjust and leaves values unchanged.
        """
        enabled = [c for c in self.categories if c.enabled]
        if not enabled:
            return

        locked_sum = sum(c.target_percentage for c in enabled if c.locked)
        unlocked = [c for c in enabled if not c.locked]

        if not unlocked:
            LOGGER.warning("Cannot normalize recipe: all enabled categories are locked.")
            return

        budget_remaining = max(0.0, 100.0 - locked_sum)
        unlocked_current_sum = sum(c.target_percentage for c in unlocked)

        if unlocked_current_sum <= 0.0:
            # Even distribution across unlocked categories
            equal_share = budget_remaining / len(unlocked)
            for c in unlocked:
                c.target_percentage = round(equal_share, 2)
        else:
            scale = budget_remaining / unlocked_current_sum
            for c in unlocked:
                c.target_percentage = round(c.target_percentage * scale, 2)

        # Micro-correction for floating point rounding to hit exactly 100.0
        current_sum = self.total_percentage()
        diff = round(100.0 - current_sum, 2)
        if diff != 0.0 and unlocked:
            unlocked[0].target_percentage = round(max(0.0, unlocked[0].target_percentage + diff), 2)

        self.recalculate_token_projections()

    def rebalance_on_category_change(self, changed_slug: str, new_percentage: float) -> None:
        """Dynamically rebalance unlocked categories when one category slider changes.

        Locked categories and the changed category retain their exact values. Remaining unlocked
        categories are scaled proportionally so the total sum is kept at exactly 100.0%.
        """
        changed_cat = self.get_category(changed_slug)
        if not changed_cat or not changed_cat.enabled:
            return

        enabled = [c for c in self.categories if c.enabled]
        locked = [c for c in enabled if c.locked and c.slug != changed_slug]
        locked_sum = sum(c.target_percentage for c in locked)

        max_allowed = max(0.0, 100.0 - locked_sum)
        bounded_pct = max(0.0, min(max_allowed, round(new_percentage, 2)))
        changed_cat.target_percentage = bounded_pct

        other_unlocked = [c for c in enabled if not c.locked and c.slug != changed_slug]
        if not other_unlocked:
            self.recalculate_token_projections()
            return

        budget_remaining = max(0.0, 100.0 - locked_sum - bounded_pct)
        other_current_sum = sum(c.target_percentage for c in other_unlocked)

        if other_current_sum <= 0.0:
            equal_share = budget_remaining / len(other_unlocked)
            for c in other_unlocked:
                c.target_percentage = round(equal_share, 2)
        else:
            scale = budget_remaining / other_current_sum
            for c in other_unlocked:
                c.target_percentage = round(c.target_percentage * scale, 2)

        # Micro-rounding correction
        current_sum = self.total_percentage()
        diff = round(100.0 - current_sum, 2)
        if diff != 0.0 and other_unlocked:
            other_unlocked[0].target_percentage = round(
                max(0.0, other_unlocked[0].target_percentage + diff), 2
            )

        self.recalculate_token_projections()

    def recalculate_token_projections(self, total_tokens: Optional[int] = None) -> None:
        """Project token counts per category based on target mixture and total token budget.

        Calculates exact target token quota = (target_percentage / 100.0) * total_target_tokens.
        This guarantees locked categories retain constant token allocations when other categories change.
        """
        if total_tokens is not None:
            self.total_target_tokens = total_tokens

        for c in self.categories:
            if c.enabled:
                c.tokens_estimated = int(round((c.target_percentage / 100.0) * self.total_target_tokens))
            else:
                c.tokens_estimated = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "name": self.name,
            "description": self.description,
            "total_target_tokens": self.total_target_tokens,
            "categories": [c.to_dict() for c in self.categories],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetRecipe:
        return cls(
            recipe_id=str(data.get("recipe_id", "custom_recipe")),
            name=str(data.get("name", "Custom Recipe")),
            description=str(data.get("description", "")),
            total_target_tokens=int(data.get("total_target_tokens", 250_000_000)),
            categories=[
                RecipeCategory.from_dict(c) for c in data.get("categories", [])
            ],
        )

    def save_to_file(self, path: Path) -> None:
        """Persist recipe to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        LOGGER.info("Saved dataset recipe '%s' to %s", self.name, path)

    @classmethod
    def load_from_file(cls, path: Path) -> Optional[DatasetRecipe]:
        """Load recipe from a JSON file."""
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return cls.from_dict(payload)
        except Exception as exc:
            LOGGER.error("Could not load recipe from %s: %s", path, exc)
            return None


def resolve_category_directories(category: RecipeCategory, candidate_roots: list[Path]) -> list[Path]:
    """Find all existing directory paths on disk for a given category."""
    resolved: list[Path] = []
    seen: set[Path] = set()

    for path_str in category.source_paths or [category.slug]:
        p = Path(path_str)
        if p.is_absolute() and p.is_dir():
            res = p.resolve()
            if res not in seen:
                seen.add(res)
                resolved.append(p)
            continue

        for root in candidate_roots:
            if not root or not root.exists():
                continue
            candidate = (root / path_str).resolve()
            if candidate.is_dir() and candidate not in seen:
                seen.add(candidate)
                resolved.append(candidate)

    return resolved


def scan_category_disk_stats(
    category: RecipeCategory,
    candidate_roots: list[Path],
) -> dict[str, Any]:
    """Scan disk folders for a category to report files count, total bytes, and estimated tokens."""
    dirs = resolve_category_directories(category, candidate_roots)
    all_files: list[Path] = []
    supported_extensions = {
        ".json", ".jsonl", ".txt", ".md", ".text", ".py", ".rs", ".go", ".c", ".cpp",
        ".h", ".hpp", ".java", ".js", ".ts", ".tsx", ".jsx", ".cs", ".sh", ".ps1",
        ".v", ".sv", ".vhd", ".vhdl"
    }

    ignored_parts = {"_quarantine", ".git", ".idea", "dist", "__pycache__"}
    for d in dirs:
        try:
            for p in d.rglob("*"):
                if any(part in p.parts for part in ignored_parts):
                    continue
                if p.is_file() and p.suffix.lower() in supported_extensions and p.stat().st_size > 0:
                    all_files.append(p)
        except Exception as exc:
            LOGGER.warning("Could not scan directory %s: %s", d, exc)

    total_bytes = sum(f.stat().st_size for f in all_files)
    # Average tokens in code/json/prose is ~3.85 bytes per token
    estimated_tokens = int(round(total_bytes / 3.85))

    category.files_count = len(all_files)
    category.disk_bytes = total_bytes
    category.disk_tokens = estimated_tokens

    return {
        "files_count": len(all_files),
        "disk_bytes": total_bytes,
        "disk_tokens": estimated_tokens,
        "directories": dirs,
        "files": all_files,
    }


# ==============================================================================
# Built-In Recipe Presets
# ==============================================================================

def create_frontier_11_pillar_recipe() -> DatasetRecipe:
    """The gold-standard balanced 11-pillar mixture (250M tokens) for frontier models."""
    return DatasetRecipe(
        recipe_id="frontier_11_pillar",
        name="Default 11-Pillar Frontier Base",
        description="Comprehensive frontier pretraining foundation across code, math, hardware, cyber, science, and multilingual domains.",
        total_target_tokens=250_000_000,
        categories=[
            RecipeCategory("Encyclopedic & Science", "encyclopedic", 25.0, source_paths=["encyclopedic", "science_pretraining", "clean_subject_corpora"]),
            RecipeCategory("Systems Code & Architecture", "code_pretraining", 15.0, source_paths=["code_pretraining"]),
            RecipeCategory("Competitive Algorithms & Graphs", "algorithms_pretraining", 10.0, source_paths=["algorithms_pretraining"]),
            RecipeCategory("STEM & Formal Mathematics", "stem_pretraining", 10.0, source_paths=["stem_pretraining"]),
            RecipeCategory("Biomedicine & Clinical Sciences", "medicine_pretraining", 10.0, source_paths=["medicine_pretraining"]),
            RecipeCategory("Hardware & Semiconductor RTL", "hardware_pretraining", 8.0, source_paths=["hardware_pretraining"]),
            RecipeCategory("Cybersecurity & Exploits", "cybersecurity_pretraining", 7.0, source_paths=["cybersecurity_pretraining"]),
            RecipeCategory("Quantitative Finance & Economics", "finance", 5.0, source_paths=["finance"]),
            RecipeCategory("Jurisprudence & Legal Reasoning", "law_pretraining", 4.0, source_paths=["law_pretraining"]),
            RecipeCategory("Multilingual Cross-Alignment", "multilingual_pretraining", 3.0, source_paths=["multilingual_pretraining"]),
            RecipeCategory("Multi-Turn Reasoning & Dialogue", "fine_tune_conversation", 3.0, source_paths=["fine_tune_conversation"]),
        ],
    )


def create_code_heavy_recipe() -> DatasetRecipe:
    """Specialist mixture weighted heavily toward systems code, algorithms, and hardware."""
    return DatasetRecipe(
        recipe_id="code_heavy",
        name="Code & Systems Heavy",
        description="Emphasizes production systems engineering, competitive algorithms, and low-level kernel/hardware programming.",
        total_target_tokens=250_000_000,
        categories=[
            RecipeCategory("Systems Code & Architecture", "code_pretraining", 40.0, source_paths=["code_pretraining"]),
            RecipeCategory("Competitive Algorithms & Graphs", "algorithms_pretraining", 20.0, source_paths=["algorithms_pretraining"]),
            RecipeCategory("Hardware & Semiconductor RTL", "hardware_pretraining", 12.0, source_paths=["hardware_pretraining"]),
            RecipeCategory("Cybersecurity & Exploits", "cybersecurity_pretraining", 10.0, source_paths=["cybersecurity_pretraining"]),
            RecipeCategory("STEM & Formal Mathematics", "stem_pretraining", 10.0, source_paths=["stem_pretraining"]),
            RecipeCategory("Encyclopedic & World Knowledge", "encyclopedic", 8.0, source_paths=["encyclopedic"]),
        ],
    )


def create_stem_reasoning_recipe() -> DatasetRecipe:
    """Specialist mixture weighted toward formal mathematics, scientific derivation, and biomedicine."""
    return DatasetRecipe(
        recipe_id="stem_reasoning",
        name="STEM & Formal Reasoning",
        description="Prioritizes deep mathematical proofs, physical sciences, biochemical pathways, and medical reasoning.",
        total_target_tokens=250_000_000,
        categories=[
            RecipeCategory("STEM & Formal Mathematics", "stem_pretraining", 30.0, source_paths=["stem_pretraining"]),
            RecipeCategory("Biomedicine & Life Sciences", "medicine_pretraining", 25.0, source_paths=["medicine_pretraining", "science_pretraining"]),
            RecipeCategory("Competitive Algorithms & Graphs", "algorithms_pretraining", 15.0, source_paths=["algorithms_pretraining"]),
            RecipeCategory("Systems Code & Architecture", "code_pretraining", 15.0, source_paths=["code_pretraining"]),
            RecipeCategory("Quantitative Finance", "finance", 10.0, source_paths=["finance"]),
            RecipeCategory("Encyclopedic & World Knowledge", "encyclopedic", 5.0, source_paths=["encyclopedic"]),
        ],
    )


def create_balanced_tiny_recipe() -> DatasetRecipe:
    """Evenly distributed mixture across all primary pretraining categories."""
    cats = [
        ("Systems Code", "code_pretraining", ["code_pretraining"]),
        ("Formal Mathematics", "stem_pretraining", ["stem_pretraining"]),
        ("Algorithms", "algorithms_pretraining", ["algorithms_pretraining"]),
        ("Hardware RTL", "hardware_pretraining", ["hardware_pretraining"]),
        ("Cybersecurity", "cybersecurity_pretraining", ["cybersecurity_pretraining"]),
        ("Biomedicine", "medicine_pretraining", ["medicine_pretraining"]),
        ("Finance", "finance", ["finance"]),
        ("Law", "law_pretraining", ["law_pretraining"]),
        ("Multilingual", "multilingual_pretraining", ["multilingual_pretraining"]),
        ("Encyclopedic", "encyclopedic", ["encyclopedic"]),
    ]
    pct = round(100.0 / len(cats), 2)
    categories = [RecipeCategory(name, slug, pct, source_paths=paths) for name, slug, paths in cats]
    recipe = DatasetRecipe(
        recipe_id="balanced_tiny",
        name="Balanced Tiny LLM",
        description="Equal weights across all core disciplines for well-rounded small models.",
        total_target_tokens=100_000_000,
        categories=categories,
    )
    recipe.normalize_to_100()
    return recipe


DEFAULT_RECIPE_PRESETS: dict[str, DatasetRecipe] = {
    "frontier_11_pillar": create_frontier_11_pillar_recipe(),
    "code_heavy": create_code_heavy_recipe(),
    "stem_reasoning": create_stem_reasoning_recipe(),
    "balanced_tiny": create_balanced_tiny_recipe(),
}


def get_default_recipe() -> DatasetRecipe:
    """Return a fresh instance of the default 11-pillar recipe."""
    return create_frontier_11_pillar_recipe()
