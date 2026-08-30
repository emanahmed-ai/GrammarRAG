"""
01_documents.py

STAGE 1 of the RAG pipeline: Raw Document Loading.

Purpose
-------
Load the `agentlans/grammar-correction` dataset programmatically from
Hugging Face, extract valid (incorrect, correct) sentence pairs, attach
traceability metadata, and persist the result as raw structured JSON for
the next stage (02_preprocessing.py).

This stage deliberately does NOT:
  - normalize whitespace
  - remove punctuation
  - deduplicate records
  - generate embeddings
  - create a ChromaDB store

Those responsibilities belong to later, separate stages.

Run independently with:
    python 01_documents.py
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

from datasets import load_dataset

import config
from utils.data_utils import adapt_record_schema, make_record_id


def load_raw_dataset(dataset_name: str, split: str):
    """
    Load the grammar correction dataset from Hugging Face.

    Args:
        dataset_name: Hugging Face dataset identifier, e.g.
            "agentlans/grammar-correction".
        split: Dataset split to load, e.g. "train".

    Returns:
        A Hugging Face `Dataset` object for the requested split.

    Raises:
        RuntimeError: If the dataset or split cannot be loaded.
    """
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as exc:  # noqa: BLE001 - surface any loader failure clearly
        raise RuntimeError(
            f"Failed to load dataset '{dataset_name}' (split='{split}') "
            f"from Hugging Face. Original error: {exc}"
        ) from exc

    return dataset


def extract_valid_records(
    dataset,
    dataset_name: str,
    split: str,
    max_records: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Convert raw Hugging Face rows into the project's structured record
    schema, validating and attaching metadata along the way.

    Args:
        dataset: A Hugging Face `Dataset` object (iterable of dict-like rows).
        dataset_name: Dataset identifier, stored in each record's metadata.
        split: Split name, used to build deterministic record IDs.
        max_records: Optional cap on the number of valid records to keep.
            If None, all valid records are kept.

    Returns:
        A list of structured records:
            {
                "incorrect": str,
                "correct": str,
                "metadata": {
                    "dataset_name": str,
                    "record_id": str,
                    "source": str,
                },
            }

    Note:
        The original incorrect/correct text is preserved EXACTLY as found
        in the source dataset. No cleaning or normalization happens here.
    """
    structured_records: List[Dict[str, Any]] = []

    for index, raw_record in enumerate(dataset):
        adapted = adapt_record_schema(raw_record)
        if adapted is None:
            # Invalid/unusable record (missing or empty fields) - skip it.
            continue

        record = {
            "incorrect": adapted["incorrect"],
            "correct": adapted["correct"],
            "metadata": {
                "dataset_name": dataset_name,
                "record_id": make_record_id(dataset_name, split, index),
                "source": config.SOURCE_LABEL,
            },
        }
        structured_records.append(record)

        if max_records is not None and len(structured_records) >= max_records:
            break

    return structured_records


def save_raw_records(
    records: List[Dict[str, Any]],
    dataset_name: str,
    split: str,
    output_path: str,
) -> None:
    """
    Persist structured raw records to disk as JSON.

    Creates the output directory if it does not already exist.

    Args:
        records: Structured records produced by `extract_valid_records`.
        dataset_name: Dataset identifier, stored at the top level of the file.
        split: Split name, stored at the top level of the file.
        output_path: Full file path to write the JSON output to.
    """
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    payload = {
        "dataset_name": dataset_name,
        "split": split,
        "num_records": len(records),
        "records": records,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    """
    Entry point: load the dataset, extract valid records, save raw JSON,
    and print a run summary.
    """
    dataset_name = config.DATASET_NAME
    split = config.DATASET_SPLIT
    max_records = config.MAX_RECORDS
    output_path = config.RAW_DATA_PATH

    print("=" * 70)
    print("STAGE 1: Raw Document Loading")
    print("=" * 70)
    print(f"Dataset:      {dataset_name}")
    print(f"Split:        {split}")
    print(f"Max records:  {max_records if max_records is not None else 'unlimited'}")
    print("-" * 70)

    try:
        dataset = load_raw_dataset(dataset_name, split)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    num_loaded = len(dataset)
    print(f"Records loaded from Hugging Face: {num_loaded}")

    records = extract_valid_records(
        dataset=dataset,
        dataset_name=dataset_name,
        split=split,
        max_records=max_records,
    )
    num_valid = len(records)
    print(f"Valid records extracted:          {num_valid}")

    if num_valid == 0:
        print(
            "ERROR: No valid records were extracted. Check that the dataset "
            "schema matches the expected column names in utils/data_utils.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        save_raw_records(records, dataset_name, split, output_path)
    except OSError as exc:
        print(f"ERROR: Failed to save output to '{output_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Output saved to:                  {output_path}")
    print("-" * 70)
    print("Sample record:")
    print(json.dumps(records[0], ensure_ascii=False, indent=2))
    print("=" * 70)
    print("STAGE 1 complete.")


if __name__ == "__main__":
    main()
