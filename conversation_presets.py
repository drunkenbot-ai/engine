from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConversationDatasetPreset:
    """Built-in Hugging Face dataset recipe for conversation/instruction data.

    Attributes:
        dataset_id: Stable UI/config identifier.
        label: User-facing label with size hint.
        hf_path: Hugging Face dataset path.
        config_name: Optional Hugging Face dataset configuration name.
        split: Dataset split to load.
        stage: Recommended training stage: base, instruction, or conversation.
        description: Short user-facing purpose hint.
    """

    dataset_id: str
    label: str
    hf_path: str
    config_name: Optional[str]
    split: str
    stage: str
    description: str


CONVERSATION_DATASET_PRESETS: dict[str, ConversationDatasetPreset] = {
    "tinystories": ConversationDatasetPreset(
        "tinystories",
        "TinyStories (~2M short stories)",
        "roneneldan/TinyStories",
        None,
        "train",
        "base",
        "Language fluency, simple narrative structure, and basic world knowledge.",
    ),
    "wikitext_103": ConversationDatasetPreset(
        "wikitext_103",
        "WikiText-103 (~100M tokens)",
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        "train",
        "base",
        "Clean Wikipedia-style long-form text for grammar, facts, and language modeling.",
    ),
    "wikipedia_en": ConversationDatasetPreset(
        "wikipedia_en",
        "Wikipedia EN 2023 (large encyclopedia)",
        "wikimedia/wikipedia",
        "20250101.en",
        "train",
        "base",
        "Broad encyclopedia prose. Use a row limit unless you intentionally want a large download.",
    ),
    "openwebtext": ConversationDatasetPreset(
        "openwebtext", "OpenWebText (~8M web documents)", "Skylion007/openwebtext",
        None, "train", "base", "Broad web text for general language pretraining.",
    ),
    "bookcorpusopen": ConversationDatasetPreset(
        "bookcorpusopen", "BookCorpusOpen (books)", "kmfoda/bookcorpus",
        None, "train", "base", "Long-form literary text and narrative language.",
    ),
    "scientific_papers": ConversationDatasetPreset(
        "scientific_papers", "Scientific Papers (ArXiv)", "scientific_papers",
        "arxiv", "train", "base", "Scientific writing and technical vocabulary.",
    ),
    "pubmed_qa": ConversationDatasetPreset(
        "pubmed_qa", "PubMed QA", "pubmed_qa", "pqa_labeled",
        "train", "instruction", "Biomedical question answering.",
    ),
    "open_orca": ConversationDatasetPreset(
        "open_orca", "OpenOrca (~1M instructions)", "Open-Orca/OpenOrca",
        None, "train", "instruction", "Diverse instruction and reasoning answers.",
    ),
    "wizardlm_evol_instruct": ConversationDatasetPreset(
        "wizardlm_evol_instruct", "WizardLM Evol-Instruct", "WizardLM/WizardLM_evol_instruct_V2_196k",
        None, "train", "instruction", "Evolved multi-step instruction following.",
    ),
    "no_robots": ConversationDatasetPreset(
        "no_robots", "No Robots (10K conversations)", "HuggingFaceH4/no_robots",
        None, "train", "conversation", "High-quality multi-turn assistant conversations.",
    ),
    "fineweb_edu": ConversationDatasetPreset(
        "fineweb_edu",
        "FineWeb-Edu sample (large educational web)",
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        "train",
        "base",
        "High-quality educational web text for base language pretraining. Use a row limit.",
    ),
    "ultrachat_200k": ConversationDatasetPreset(
        "ultrachat_200k",
        "UltraChat 200K (~200K conversations)",
        "HuggingFaceH4/ultrachat_200k",
        None,
        "train_sft",
        "conversation",
        "Multi-turn assistant conversation and helpful response style.",
    ),
    "dailydialog": ConversationDatasetPreset(
        "dailydialog",
        "DailyDialog (~13K dialogues)",
        "pixelsandpointers/better_daily_dialog",
        None,
        "train",
        "conversation",
        "Natural everyday dialogue and short conversational turns.",
    ),
    "alpaca_52k": ConversationDatasetPreset(
        "alpaca_52k",
        "Alpaca 52K (~52K instructions)",
        "tatsu-lab/alpaca",
        None,
        "train",
        "instruction",
        "Instruction following with concise task-answer pairs.",
    ),
    "dolly_15k": ConversationDatasetPreset(
        "dolly_15k",
        "Dolly 15K (~15K instructions)",
        "databricks/databricks-dolly-15k",
        None,
        "train",
        "instruction",
        "Human-written instruction following, brainstorming, QA, and classification.",
    ),
    "oasst1": ConversationDatasetPreset(
        "oasst1",
        "OpenAssistant OASST1 (~88K messages)",
        "OpenAssistant/oasst1",
        None,
        "train",
        "conversation",
        "Assistant-style conversational messages and preference data text.",
    ),
    "slimorca": ConversationDatasetPreset(
        "slimorca",
        "SlimOrca (~517K examples)",
        "Open-Orca/SlimOrca",
        None,
        "train",
        "instruction",
        "Instruction and reasoning-style assistant answers.",
    ),
    "codealpaca_20k": ConversationDatasetPreset(
        "codealpaca_20k",
        "CodeAlpaca 20K (~20K code instructions)",
        "sahil2801/CodeAlpaca-20k",
        None,
        "train",
        "code",
        "Small code instruction dataset for text-to-code, code explanation, and programming tasks.",
    ),
    "magicoder_oss_75k": ConversationDatasetPreset(
        "magicoder_oss_75k",
        "Magicoder OSS-Instruct 75K (~75K code tasks)",
        "ise-uiuc/Magicoder-OSS-Instruct-75K",
        None,
        "train",
        "code",
        "Code generation instruction data built from open-source code references.",
    ),
    "evol_codealpaca": ConversationDatasetPreset(
        "evol_codealpaca",
        "Evol CodeAlpaca (~evolved code instructions)",
        "theblackcat102/evol-codealpaca-v1",
        None,
        "train",
        "code",
        "Evolved programming instructions for stronger code fine-tuning variety.",
    ),
}

BASE_DATASET_IDS = [dataset_id for dataset_id, preset in CONVERSATION_DATASET_PRESETS.items() if preset.stage == "base"]
INSTRUCTION_DATASET_IDS = [dataset_id for dataset_id, preset in CONVERSATION_DATASET_PRESETS.items() if preset.stage == "instruction"]
CONVERSATION_DATASET_IDS = [dataset_id for dataset_id, preset in CONVERSATION_DATASET_PRESETS.items() if preset.stage == "conversation"]
CODE_DATASET_IDS = [dataset_id for dataset_id, preset in CONVERSATION_DATASET_PRESETS.items() if preset.stage == "code"]


def dataset_ids_for_stage(stage: str) -> list[str]:
    """Return online dataset IDs available for a training stage.

    Args:
        stage: Dataset/training stage.

    Returns:
        Dataset IDs for the selected stage. Base pretraining intentionally
        exposes every built-in source so users can build mixed base corpora.
    """

    if stage == "base":
        return list(CONVERSATION_DATASET_PRESETS)
    if stage == "instruction":
        return INSTRUCTION_DATASET_IDS
    if stage == "conversation":
        return CONVERSATION_DATASET_IDS
    if stage == "code":
        return CODE_DATASET_IDS
    return []


def dataset_stage_label(stage: str) -> str:
    """Return a user-facing stage label.

    Args:
        stage: Dataset/training stage.

    Returns:
        Human-readable stage name.
    """

    return {
        "base": "Base pretraining",
        "instruction": "Instruction fine-tune",
        "conversation": "Conversation fine-tune",
        "code": "Code fine-tune",
        "tool_call": "Tool-call fine-tune",
    }.get(stage, "Custom")



