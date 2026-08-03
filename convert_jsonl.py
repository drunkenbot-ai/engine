import json
import re
from pathlib import Path

# ==========================================================
# Configuration
# ==========================================================

INPUT_FOLDER = "F:\\Micro_LLM_Projects\\Nero\\training_data\\dnet_scape\\"
OUTPUT_FOLDER = "F:\\Micro_LLM_Projects\\Nero\\training_data\\dnet_scape\\Converted"
MIN_DOCUMENT_SIZE_KB = 10
MIN_DOCUMENT_SIZE_BYTES = MIN_DOCUMENT_SIZE_KB * 1024

SAVE_AS_MARKDOWN = True

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# ==========================================================
# Things to remove
# ==========================================================

REMOVE_EXACT = {
    "[TABLE]",
    "[/TABLE]",
    "Project Data",
    "Overview",
    "Description",
    "Background",
    "Assumptions",
    "Objectives",
    "RACI(J)",
    "Development Management",
    "Testing Plan",
    "Additional Development Documentation",
    "Additional Close Information",
    "Project Lessons Learned",
    "Scope Sign Offs",
    "Deliverables",
    "Feasibility",
    "Development",
    "Closure",
    "What",
    "Why",
    "Who",
    "How",
}

REMOVE_PREFIXES = (
    "Viewed By",
    "Additional ",
    "Project Ticket",
    "Idea Ticket",
    "Timetracking Entry",
    "Lifetime Assessment",
    "Initial Effort",
    "Date Started",
    "Date Finished",
    "Last Revised",
    "Key Stakeholder",
    "Project Sponsor",
    "Project Manager",
    "Jira Project",
    "Versions",
    "Feature Requests",
    "Migration Plan",
    "Rollback plan",
    "Support Team",
    "Announcements",
)

INVALID_FILENAME = r'[<>:"/\\|?*]'

# ==========================================================
# Helpers
# ==========================================================

def clean_filename(name: str) -> str:
    name = re.sub(INVALID_FILENAME, "_", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()[:180]


def clean_text(text: str):

    if not text:
        return ""

    # -----------------------------------------
    # Remove URLs
    # -----------------------------------------
    text = re.sub(r"http\S+", "", text)

    # -----------------------------------------
    # Remove Windows paths
    # -----------------------------------------
    text = re.sub(r"[A-Za-z]:\\[^\s]+", "", text)

    cleaned = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # Remove table markers
        if line in ("[TABLE]", "[/TABLE]"):
            continue

        # Remove all table rows
        if "|" in line:
            continue

        # Remove "1 flat", "2 flat"
        if re.fullmatch(r"\d+\s+flat", line):
            continue

        # Remove exact headings
        if line in REMOVE_EXACT:
            continue

        # Remove common prefixes
        if any(line.startswith(prefix) for prefix in REMOVE_PREFIXES):
            continue

        # Remove XML / HTML tags
        if re.match(r"<.*?>", line):
            continue

        # Remove Jira IDs
        line = re.sub(r"\b[A-Z]+-\d+\b", "", line)

        # Collapse whitespace
        line = re.sub(r"\s+", " ", line).strip()

        if len(line) < 2:
            continue

        cleaned.append(line)

    # Remove excessive blank lines
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ==========================================================
# Convert
# ==========================================================

count = 0

for jsonl_file in Path(INPUT_FOLDER).glob("*.jsonl"):

    print(f"Processing {jsonl_file.name}")

    with open(jsonl_file, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            title = clean_text(obj.get("title", "Untitled"))
            body = clean_text(obj.get("text", ""))

            if len(body) < 50:
                continue
            
            if len(body.encode("utf-8")) < MIN_DOCUMENT_SIZE_BYTES:
                continue
    
            filename = clean_filename(title)

            ext = ".md" if SAVE_AS_MARKDOWN else ".txt"

            outfile = Path(OUTPUT_FOLDER) / f"{filename}{ext}"

            if SAVE_AS_MARKDOWN:

                content = f"# {title}\n\n{body}\n"

            else:

                content = f"{title}\n\n{body}\n"

            with open(outfile, "w", encoding="utf-8") as out:
                out.write(content)

            count += 1

print()
print("=" * 60)
print(f"Converted {count} documents.")
print(f"Saved to: {OUTPUT_FOLDER}")
print("=" * 60)
