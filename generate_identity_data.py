"""Generate a base-pretraining identity corpus file for a Micro LLM project.

Reads the model's name and creation date straight from the project's
``project.json`` (written by the app itself), combines them with fixed facts
about the creator, and writes a plain-text file of many distinctly-phrased
sentences about who the model is. Designed to be dropped straight into a
project's ``training_data`` folder as an ordinary source file.

Why sentences must be genuinely distinct, not just recombined:
    The dataset-build pipeline drops documents whose sentences are
    dominated by exact repetition (see ``dataset_mixture.MAX_REPETITIVE_UNIT_RATIO``,
    checked per document against the *sentence* level, not the paragraph
    level) and removes exact duplicate documents outright. Shuffling a small
    fixed sentence pool into different paragraph groupings does NOT create
    new sentences from the filter's point of view -- it still sees the same
    handful of sentences repeated over and over and will exclude the whole
    file. This script instead builds sentences combinatorially (varying verb
    choice, phrase order, and which facts are mentioned) so the pool of
    distinct sentences is large, and writes each one only once.

Why file size does not need to be huge:
    Training runs for multiple epochs (see ``TrainingConfig.epochs``), so
    every sentence in this file is seen again on every epoch automatically.
    A few hundred distinct sentences already gives real, repeated exposure
    across a training run without this file dominating a multi-hundred-MB
    corpus.

Usage:
    python generate_identity_data.py /path/to/project_dir \
        [--output training_data/identity/identity_facts.txt] \
        [--sentence-count 500] \
        [--creator "DrunkenBot"] \
        [--maker "Nilesh Jadhav"] \
        [--role "AI assistant"]

If ``--output`` is a relative path, it is resolved against the project
directory. The default output path lands under the project's
``training_data`` folder, in its own subfolder, so it is picked up by the
normal source-vault scan alongside every other category.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import random
from datetime import datetime
from pathlib import Path

# Mirrors dataset_mixture.py's thresholds exactly, so this script's
# self-check reflects what the real pipeline will do with this file.
MAX_REPETITIVE_UNIT_RATIO = 0.35
MIN_REPETITION_CHECK_UNITS = 20
MIN_REPETITION_CHECK_CHARS = 2_000


def load_project_facts(project_dir: Path) -> dict[str, str]:
    """Read the project name and creation date from project.json.

    Args:
        project_dir: Project folder containing ``project.json``.

    Returns:
        Dict with ``name`` and ``created`` (a human-readable date string).

    Raises:
        FileNotFoundError: If ``project.json`` does not exist.
        ValueError: If ``project.json`` does not contain a project name.
    """

    project_file = project_dir / "project.json"
    if not project_file.exists():
        raise FileNotFoundError(
            f"Could not find {project_file}. Pass the folder that contains your project's project.json."
        )
    data = json.loads(project_file.read_text(encoding="utf-8"))
    name = str(data.get("project_name", "")).strip()
    if not name:
        raise ValueError(f"{project_file} has no project_name set.")

    raw_created = str(data.get("created_at") or data.get("saved_at") or "").strip()
    created = _format_date(raw_created)
    return {"name": name, "created": created}


def _format_date(raw: str) -> str:
    """Format an ISO-ish timestamp into a human-readable month/year.

    Args:
        raw: Timestamp string, typically ``datetime.isoformat()`` output.

    Returns:
        A "Month YYYY" string, or "an unknown date" if parsing fails.
    """

    if not raw:
        return "an unknown date"
    try:
        return datetime.fromisoformat(raw).strftime("%B %Y")
    except ValueError:
        return "an unknown date"


# Independent slot dimensions. itertools.product over these (crossed with
# several sentence "shapes" below) is what makes the combinatorial pool
# large without hand-writing hundreds of sentences.
CREATE_VERBS = ["created", "built", "made", "developed", "brought to life"]
ROLE_PHRASES = ["an {role}", "a helpful {role}", "an {role} you can talk to"]
CONNECTORS = ["", " -- a project by {maker}", ", led by {maker}"]
DATE_PHRASES = ["", " in {created}", ", first developed in {created}"]


def _render_shapes(values: dict[str, str]) -> set[str]:
    """Render every template shape across every slot combination.

    Args:
        values: Base fact values (``name``, ``creator``, ``maker``, ``role``,
            ``created``).

    Returns:
        Set of distinct rendered sentences.
    """

    sentences: set[str] = set()
    for verb, role_phrase, connector, date_phrase in itertools.product(
        CREATE_VERBS, ROLE_PHRASES, CONNECTORS, DATE_PHRASES
    ):
        role_text = role_phrase.format(role=values["role"])
        connector_text = connector.format(maker=values["maker"])
        date_text = date_phrase.format(created=values["created"])

        sentences.add(
            f"{values['name']} is {role_text} {verb} by {values['creator']}{connector_text}{date_text}.".replace(
                "  ", " "
            )
        )
        sentences.add(
            f"{values['creator']}{connector_text} {verb} {values['name']}, {role_text}{date_text}.".replace(
                "  ", " "
            )
        )
        sentences.add(
            f"{values['name']}, {role_text}, was {verb} by {values['creator']}{connector_text}{date_text}.".replace(
                "  ", " "
            )
        )

    # A modest number of naturally-phrased question/answer lines, written as
    # prose rather than a structured instruction format (this file stays
    # inside the base-pretraining corpus, not a fine-tuning dataset).
    qa_lines = [
        "What is your name? {name}.",
        "Who made you? {creator}, a project by {maker}, created me.",
        "Who created you? I was created by {creator}.",
        "Are you an AI? Yes, {name} is an {role}.",
        "When were you created? Around {created}.",
        "Who is your creator? {creator}, founded by {maker}.",
        "Do you belong to another company? No, {name} was made by {creator}, not any other company.",
    ]
    sentences.update(line.format(**values) for line in qa_lines)
    return sentences


def build_sentence_pool(
    facts: dict[str, str],
    creator: str,
    maker: str,
    role: str,
    target_count: int,
    seed: int,
) -> list[str]:
    """Build a pool of distinct sentences, sized to ``target_count``.

    Args:
        facts: Project facts (``name``, ``created``).
        creator: Creator/organization name.
        maker: Person who made the creator.
        role: What the model is (e.g. "AI assistant").
        target_count: Desired number of sentences.
        seed: Random seed used when sampling down to ``target_count``.

    Returns:
        List of distinct sentences, shuffled, at most ``target_count`` long.
    """

    values = {
        "name": facts["name"],
        "created": facts["created"],
        "creator": creator,
        "maker": maker,
        "role": role,
    }
    all_sentences = sorted(_render_shapes(values))
    rng = random.Random(seed)
    rng.shuffle(all_sentences)
    if target_count >= len(all_sentences):
        print(
            f"Note: requested {target_count} sentences, but the combinatorial "
            f"template pool only has {len(all_sentences)} distinct options. "
            "Using all of them. Add more verbs/phrases/connectors to the "
            "CREATE_VERBS / ROLE_PHRASES / CONNECTORS / DATE_PHRASES lists "
            "above to raise this ceiling."
        )
        return all_sentences
    return all_sentences[:target_count]


def group_into_paragraphs(sentences: list[str], sentences_per_paragraph: tuple[int, int], seed: int) -> list[str]:
    """Group sentences into paragraphs, each sentence used exactly once.

    Args:
        sentences: Distinct sentences to group (already shuffled/ordered).
        sentences_per_paragraph: Inclusive (min, max) sentence count range
            per generated paragraph.
        seed: Random seed for paragraph-size choices.

    Returns:
        List of paragraph strings covering every input sentence exactly once.
    """

    rng = random.Random(seed)
    low, high = sentences_per_paragraph
    paragraphs: list[str] = []
    index = 0
    while index < len(sentences):
        size = rng.randint(low, high)
        chunk = sentences[index : index + size]
        paragraphs.append(" ".join(chunk))
        index += size
    return paragraphs


def _canonical_block(text: str) -> str:
    """Match dataset_mixture._canonical_corpus_block exactly."""

    return re.sub(r"\s+", " ", text).strip().lower()


def self_check_diversity(full_text: str) -> tuple[int, float, bool]:
    """Reproduce the pipeline's low-diversity check on the generated text.

    Args:
        full_text: The full generated file content.

    Returns:
        Tuple of (unit_count, duplicate_ratio, would_be_excluded).
    """

    raw_units = re.split(
        r"(?<=[.!?])\s+|\n+(?=(?:User|Assistant|System|Instruction|Response):)", full_text
    )
    units = [_canonical_block(unit) for unit in raw_units if len(_canonical_block(unit)) >= 24]
    if len(full_text) < MIN_REPETITION_CHECK_CHARS or len(units) < MIN_REPETITION_CHECK_UNITS:
        return len(units), 0.0, False
    duplicate_ratio = 1.0 - (len(set(units)) / len(units))
    return len(units), duplicate_ratio, duplicate_ratio > MAX_REPETITIVE_UNIT_RATIO


def main() -> None:
    """Parse arguments, generate the identity corpus, and write it to disk."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_dir", type=Path, help="Project folder containing project.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_data/identity/identity_facts.txt"),
        help="Output path, resolved against project_dir if relative (default: %(default)s)",
    )
    parser.add_argument("--creator", default="DrunkenBot", help="Creator/organization name")
    parser.add_argument("--maker", default="Nilesh Jadhav", help="Person who made the creator")
    parser.add_argument("--role", default="AI assistant", help="What the model is")
    parser.add_argument(
        "--sentence-count",
        type=int,
        default=500,
        help="Target number of distinct sentences (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    facts = load_project_facts(project_dir)
    sentence_pool = build_sentence_pool(
        facts, args.creator, args.maker, args.role, args.sentence_count, args.seed
    )
    paragraphs = group_into_paragraphs(sentence_pool, sentences_per_paragraph=(2, 4), seed=args.seed)
    full_text = "\n\n".join(paragraphs) + "\n"

    unit_count, duplicate_ratio, would_be_excluded = self_check_diversity(full_text)

    output_path = args.output if args.output.is_absolute() else project_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full_text, encoding="utf-8")

    print(f"Project name:       {facts['name']}")
    print(f"Created:            {facts['created']}")
    print(f"Creator / maker:    {args.creator} / {args.maker}")
    print(f"Distinct sentences: {len(sentence_pool)}")
    print(f"Paragraphs written: {len(paragraphs)}")
    print(f"Output size:        {len(full_text):,} characters")
    print(f"Written to:         {output_path}")
    print()
    print("Diversity self-check (same rule the pipeline applies):")
    print(f"  sentence units:    {unit_count}")
    print(f"  duplicate ratio:   {duplicate_ratio:.1%} (excluded if over 35%)")
    print(f"  would be excluded: {would_be_excluded}")
    if would_be_excluded:
        print(
            "  WARNING: this file would be dropped by the low-diversity filter. "
            "Increase --sentence-count so the distinct-sentence pool covers more of the file."
        )
    print()
    print("Next steps:")
    print("  1. Re-run dataset preparation (this is a new file, so it will")
    print("     be extracted automatically -- no need for force reprocess).")
    print("  2. Train for multiple epochs; this file is re-seen every epoch,")
    print("     which is what gives a small model repeated exposure to it.")
    print("  3. Recall from a small from-scratch model is probabilistic, not")
    print("     guaranteed -- if you need reliable identity answers, pair")
    print("     this with a system-prompt/prepended-context mechanism at")
    print("     inference time if your chat interface supports one.")


if __name__ == "__main__":
    main()
