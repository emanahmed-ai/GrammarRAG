"""
02_preprocessing.py

STAGE 2 of the RAG pipeline: Preprocessing (cleaning & validation).

Purpose
-------
Load the raw (incorrect, correct) sentence pairs produced by
01_documents.py, validate their structure, normalize incidental
whitespace/line-break/Unicode noise, remove invalid or duplicate
records, and persist the result as cleaned JSON for the next stage
(03_chunking.py).

This stage deliberately does NOT:
  - correct grammar or spelling
  - rewrite sentences
  - remove punctuation
  - use an LLM
  - generate embeddings
  - create a ChromaDB store

Those responsibilities belong to other stages. This stage only cleans;
it never corrects.

Run independently with:
    python 02_preprocessing.py
"""

import json
import os
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

import config

REQUIRED_FIELDS = ("incorrect", "correct", "metadata")


def load_raw_documents(input_path: str) -> Dict[str, Any]:
    """
    Load the raw JSON payload produced by 01_documents.py.

    Args:
        input_path: Path to the raw JSON file
            (data/raw/grammar_correction_raw.json).

    Returns:
        The parsed JSON payload as a dict, expected to contain at least
        a "records" list.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the file is not valid JSON or lacks the expected
            top-level structure.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Raw input file not found at '{input_path}'. "
            "Run 01_documents.py first to generate it."
        )

    with open(input_path, "r", encoding="utf-8") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Raw input file is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError(
            "Raw input file does not match the expected top-level structure "
            "(missing a 'records' list)."
        )
    if not isinstance(payload["records"], list):
        raise ValueError("'records' in the raw input file must be a list.")

    return payload


def normalize_text(text: str) -> str:
    """
    Clean incidental formatting noise from a string WITHOUT altering its
    grammatical or semantic content.

    Applies:
        - Safe Unicode normalization (NFC).
        - Line-break normalization (\\r\\n / \\r -> \\n, then collapsed
          into single spaces, since these are single sentences).
        - Collapsing repeated whitespace into a single space.
        - Stripping leading/trailing whitespace.

    Does NOT:
        - Remove or alter punctuation.
        - Change casing.
        - Correct spelling or grammar.

    Args:
        text: Raw input string.

    Returns:
        The normalized string.
    """
    if not isinstance(text, str):
        return text

    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\n", " ")
    normalized = " ".join(normalized.split())
    return normalized.strip()


def validate_record(record: Any) -> bool:
    """
    Check whether a raw record has the required structure and non-empty
    content, WITHOUT mutating the input.

    A record is valid if:
        - It is a dict.
        - It contains "incorrect", "correct", and "metadata" keys.
        - "incorrect" and "correct" are strings.
        - "incorrect" and "correct" are non-empty after normalization.
        - "metadata" is a dict.

    Args:
        record: A single raw record.

    Returns:
        True if the record is valid, False otherwise.
    """
    if not isinstance(record, dict):
        return False

    for field in REQUIRED_FIELDS:
        if field not in record:
            return False

    incorrect = record.get("incorrect")
    correct = record.get("correct")
    metadata = record.get("metadata")

    if not isinstance(incorrect, str) or not isinstance(correct, str):
        return False
    if not isinstance(metadata, dict):
        return False

    if normalize_text(incorrect) == "":
        return False
    if normalize_text(correct) == "":
        return False

    return True


def deduplicate_records(
    records: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Remove exact duplicate (incorrect, correct) pairs, comparing the
    NORMALIZED text of each field. The first occurrence of a pair is
    kept; later duplicates are dropped.

    Args:
        records: Already-normalized, valid records.

    Returns:
        A tuple of (deduplicated_records, num_duplicates_removed).
    """
    seen: Set[Tuple[str, str]] = set()
    deduped: List[Dict[str, Any]] = []
    num_duplicates = 0

    for record in records:
        key = (record["incorrect"], record["correct"])
        if key in seen:
            num_duplicates += 1
            continue
        seen.add(key)
        deduped.append(record)

    return deduped, num_duplicates


def preprocess_documents(
    raw_payload: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Run the full preprocessing pipeline over a raw payload's records:
    validate, normalize, and deduplicate.

    Args:
        raw_payload: The parsed raw JSON payload (from load_raw_documents).

    Returns:
        A tuple of (final_records, statistics), where statistics contains
        original_records, invalid_records_removed, duplicate_records_removed,
        and final_records counts.
    """
    raw_records = raw_payload["records"]
    original_count = len(raw_records)

    cleaned_records: List[Dict[str, Any]] = []
    invalid_count = 0

    for record in raw_records:
        if not validate_record(record):
            invalid_count += 1
            continue

        cleaned_records.append(
            {
                "incorrect": normalize_text(record["incorrect"]),
                "correct": normalize_text(record["correct"]),
                # Metadata is preserved exactly as-is; it is traceability
                # information, not text content to be cleaned.
                "metadata": record["metadata"],
            }
        )

    deduped_records, duplicate_count = deduplicate_records(cleaned_records)

    statistics = {
        "original_records": original_count,
        "invalid_records_removed": invalid_count,
        "duplicate_records_removed": duplicate_count,
        "final_records": len(deduped_records),
    }

    return deduped_records, statistics


def save_preprocessed_documents(
    records: List[Dict[str, Any]],
    dataset_name: str,
    split: str,
    statistics: Dict[str, int],
    output_path: str,
) -> None:
    """
    Persist the cleaned records to disk as JSON, in a schema compatible
    with the raw-stage output plus an added "statistics" block.

    Args:
        records: Cleaned, deduplicated records.
        dataset_name: Dataset identifier, carried over from the raw payload.
        split: Split name, carried over from the raw payload.
        statistics: Preprocessing statistics (see preprocess_documents).
        output_path: Full file path to write the JSON output to.
    """
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    payload = {
        "dataset_name": dataset_name,
        "split": split,
        "num_records": len(records),
        "records": records,
        "statistics": statistics,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    """
    Entry point: load the raw JSON, preprocess it, save the cleaned JSON,
    and print a run summary.
    """
    input_path = config.RAW_DATA_PATH
    output_path = config.PROCESSED_DATA_PATH

    print("=" * 70)
    print("STAGE 2: Preprocessing (Cleaning & Validation)")
    print("=" * 70)
    print(f"Input path:   {input_path}")

    try:
        raw_payload = load_raw_documents(input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    dataset_name = raw_payload.get("dataset_name", config.DATASET_NAME)
    split = raw_payload.get("split", config.DATASET_SPLIT)

    records, statistics = preprocess_documents(raw_payload)

    print(f"Original records:            {statistics['original_records']}")
    print(f"Invalid records removed:     {statistics['invalid_records_removed']}")
    print(f"Duplicate records removed:   {statistics['duplicate_records_removed']}")
    print(f"Final records:               {statistics['final_records']}")
    print("-" * 70)

    if statistics["final_records"] == 0:
        print(
            "ERROR: No valid records remain after preprocessing.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        save_preprocessed_documents(
            records, dataset_name, split, statistics, output_path
        )
    except OSError as exc:
        print(f"ERROR: Failed to save output to '{output_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Output saved to:             {output_path}")
    print("-" * 70)
    print("Sample preprocessed record:")
    print(json.dumps(records[0], ensure_ascii=False, indent=2))
    print("=" * 70)
    print("STAGE 2 complete.")


if __name__ == "__main__":
    main()
