"""
utils/data_utils.py

Small, reusable helpers shared by multiple pipeline stages.

Kept separate from the numbered stage files so those files stay focused
and readable, per the project's modularity requirement.
"""

from typing import Any, Dict, Optional


# The agentlans/grammar-correction dataset uses columns:
#   "input"  -> ungrammatical sentence
#   "output" -> corrected sentence
# (Verified directly against the dataset card on Hugging Face.)
#
# We still check a small set of fallback column-name candidates defensively,
# in case of future dataset revisions or if this loader is reused against a
# similarly-shaped grammar correction dataset with different column names.
INCORRECT_COLUMN_CANDIDATES = ["input", "incorrect", "source", "original", "wrong"]
CORRECT_COLUMN_CANDIDATES = ["output", "correct", "target", "corrected", "correction"]


def adapt_record_schema(raw_record: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Adapt a single raw Hugging Face dataset record into the project's
    internal schema: {"incorrect": ..., "correct": ...}.

    This function does NOT clean, trim, or normalize the text in any way.
    It only maps whichever column names the dataset actually uses onto
    our consistent internal field names, exactly as required for Stage 1
    (raw extraction).

    Args:
        raw_record: A single row from the Hugging Face dataset, as a dict.

    Returns:
        A dict with "incorrect" and "correct" string keys, or None if the
        record does not contain usable text under any recognized column
        name (the caller is responsible for counting/skipping invalid
        records).
    """
    incorrect_text = _first_present(raw_record, INCORRECT_COLUMN_CANDIDATES)
    correct_text = _first_present(raw_record, CORRECT_COLUMN_CANDIDATES)

    if incorrect_text is None or correct_text is None:
        return None

    # Basic validation only: both fields must be non-empty strings.
    # (Deeper cleaning is explicitly deferred to 02_preprocessing.py.)
    if not isinstance(incorrect_text, str) or not isinstance(correct_text, str):
        return None
    if incorrect_text.strip() == "" or correct_text.strip() == "":
        return None

    return {"incorrect": incorrect_text, "correct": correct_text}


def _first_present(record: Dict[str, Any], candidate_keys: list) -> Optional[Any]:
    """Return the value of the first candidate key that exists in `record`."""
    for key in candidate_keys:
        if key in record:
            return record[key]
    return None


def make_record_id(dataset_name: str, split: str, index: int) -> str:
    """
    Build a deterministic, human-readable record ID.

    Deterministic IDs (rather than random UUIDs) matter later in the
    pipeline: Stage 5 (ChromaDB creation) needs stable IDs so the script
    can be safely rerun without creating duplicate vector store entries.
    """
    safe_dataset_name = dataset_name.replace("/", "_")
    return f"{safe_dataset_name}_{split}_{index:06d}"
