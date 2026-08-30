"""
03_chunking.py

STAGE 3 of the RAG pipeline: Chunking.

Purpose
-------
Convert cleaned (incorrect, correct) sentence pairs produced by
02_preprocessing.py into semantic retrieval chunks for the vector store.

Design principle
-----------------
ONE GRAMMAR CORRECTION PAIR = ONE SEMANTIC RETRIEVAL UNIT.

This dataset is sentence-level, not document-level. Splitting a single
sentence pair further (by words, tokens, or a generic recursive text
splitter) would destroy the exact thing the RAG system needs to
retrieve: the contrast between an incorrect form and its correction.
So each valid record becomes exactly one chunk, formatted as a clearly
labeled "Original Sentence" / "Corrected Sentence" block.

This stage deliberately does NOT:
  - split sentences into smaller fragments
  - correct grammar or spelling
  - call an LLM
  - generate embeddings
  - create a ChromaDB store

Those responsibilities belong to other stages.

Run independently with:
    python 03_chunking.py
"""

import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import config

REQUIRED_FIELDS = ("incorrect", "correct", "metadata")


def load_preprocessed_documents(input_path: str) -> Dict[str, Any]:
    """
    Load the preprocessed JSON payload produced by 02_preprocessing.py.

    Args:
        input_path: Path to the preprocessed JSON file
            (data/processed/grammar_correction_preprocessed.json).

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
            f"Preprocessed input file not found at '{input_path}'. "
            "Run 02_preprocessing.py first to generate it."
        )

    with open(input_path, "r", encoding="utf-8") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Preprocessed input file is not valid JSON: {exc}"
            ) from exc

    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError(
            "Preprocessed input file does not match the expected top-level "
            "structure (missing a 'records' list)."
        )
    if not isinstance(payload["records"], list):
        raise ValueError("'records' in the preprocessed input file must be a list.")

    return payload


def _is_valid_record(record: Any) -> bool:
    """
    Check whether a preprocessed record has the required structure and
    non-empty incorrect/correct text.

    Args:
        record: A single record from the preprocessed file.

    Returns:
        True if the record is usable for chunking, False otherwise.
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
    if incorrect.strip() == "" or correct.strip() == "":
        return False

    return True


def create_chunk_text(incorrect: str, correct: str) -> str:
    """
    Build the human-readable chunk text used for semantic embedding and
    retrieval, clearly labeling the incorrect and corrected forms so the
    contrast between them is preserved.

    Args:
        incorrect: The ungrammatical sentence.
        correct: The corrected sentence.

    Returns:
        A formatted string:
            "Original Sentence:\\n<incorrect>\\n\\nCorrected Sentence:\\n<correct>"
    """
    return f"Original Sentence:\n{incorrect}\n\nCorrected Sentence:\n{correct}"


def make_chunk_id(metadata: Dict[str, Any], incorrect: str, correct: str) -> str:
    """
    Derive a deterministic, unique, reproducible chunk ID.

    Strategy:
        - If metadata already contains a non-empty "record_id" (as produced
          by 01_documents.py / utils.data_utils.make_record_id), reuse it
          directly. It is already deterministic and unique per record, so
          introducing a second ID would only add redundant complexity.
        - Otherwise, fall back to a deterministic hash of the sentence pair
          itself (content-derived, not random and not time-based), so the
          same input always yields the same chunk ID even without a
          record_id.

    Args:
        metadata: The record's metadata dict.
        incorrect: The incorrect sentence (used only for the fallback).
        correct: The correct sentence (used only for the fallback).

    Returns:
        A deterministic string chunk ID.
    """
    record_id = metadata.get("record_id")
    if isinstance(record_id, str) and record_id.strip() != "":
        return record_id.strip()

    digest = hashlib.sha256(f"{incorrect}|{correct}".encode("utf-8")).hexdigest()
    return f"chunk_{digest[:16]}"


def create_chunk(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a single semantic retrieval chunk from one valid preprocessed
    record.

    Args:
        record: A valid record with "incorrect", "correct", and "metadata".

    Returns:
        A chunk dict:
            {
                "chunk_id": str,
                "text": str,
                "incorrect": str,
                "correct": str,
                "metadata": dict,
            }
    """
    incorrect = record["incorrect"]
    correct = record["correct"]
    metadata = record["metadata"]

    return {
        "chunk_id": make_chunk_id(metadata, incorrect, correct),
        "text": create_chunk_text(incorrect, correct),
        "incorrect": incorrect,
        "correct": correct,
        "metadata": metadata,
    }


def create_chunks(
    records: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Convert a list of preprocessed records into chunks, skipping any
    malformed records safely.

    Args:
        records: Records from the preprocessed JSON payload.

    Returns:
        A tuple of (chunks, num_skipped).
    """
    chunks: List[Dict[str, Any]] = []
    num_skipped = 0

    for record in records:
        if not _is_valid_record(record):
            num_skipped += 1
            continue
        chunks.append(create_chunk(record))

    return chunks, num_skipped


def save_chunks(
    chunks: List[Dict[str, Any]],
    dataset_name: str,
    split: str,
    output_path: str,
    source_statistics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Persist chunks to disk as JSON.

    Args:
        chunks: Chunk records produced by create_chunks().
        dataset_name: Dataset identifier, carried over from the input payload.
        split: Split name, carried over from the input payload.
        output_path: Full file path to write the JSON output to.
        source_statistics: Optional preprocessing statistics carried over
            from the Stage 2 output, kept separate under
            "preprocessing_statistics" so it is preserved without being
            confused with this stage's own counts.
    """
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    payload: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "split": split,
        "num_chunks": len(chunks),
        "chunks": chunks,
    }

    if source_statistics is not None:
        payload["preprocessing_statistics"] = source_statistics

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    """
    Entry point: load the preprocessed JSON, build chunks, save the
    chunked JSON, and print a run summary.
    """
    input_path = config.PROCESSED_DATA_PATH
    output_path = config.CHUNKS_DATA_PATH

    print("=" * 70)
    print("STAGE 3: Chunking")
    print("=" * 70)
    print(f"Input path:   {input_path}")

    try:
        preprocessed_payload = load_preprocessed_documents(input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    dataset_name = preprocessed_payload.get("dataset_name", config.DATASET_NAME)
    split = preprocessed_payload.get("split", config.DATASET_SPLIT)
    source_statistics = preprocessed_payload.get("statistics")

    records = preprocessed_payload["records"]
    input_record_count = len(records)

    chunks, num_skipped = create_chunks(records)

    print(f"Input records:                {input_record_count}")
    print(f"Malformed records skipped:    {num_skipped}")
    print(f"Chunks created:               {len(chunks)}")
    print("-" * 70)

    if len(chunks) == 0:
        print("ERROR: No valid chunks were created.", file=sys.stderr)
        sys.exit(1)

    try:
        save_chunks(chunks, dataset_name, split, output_path, source_statistics)
    except OSError as exc:
        print(f"ERROR: Failed to save output to '{output_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Output saved to:              {output_path}")
    print("-" * 70)
    print("Sample chunk:")
    print(json.dumps(chunks[0], ensure_ascii=False, indent=2))
    print("=" * 70)
    print("STAGE 3 complete.")


if __name__ == "__main__":
    main()
