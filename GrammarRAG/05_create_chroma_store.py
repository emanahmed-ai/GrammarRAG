"""
05_create_chroma_store.py

STAGE 5 of the RAG pipeline: Persistent ChromaDB Vector Store.

Purpose
-------
Load the precomputed sentence embeddings produced by
04_vector_representation.py and upsert them into a persistent local
ChromaDB collection, so 06_retrieve_context.py can later run semantic
search against it without recomputing anything.

Data flow (must stay exactly this direction):

    Precomputed Embeddings (data/processed/grammar_correction_embeddings.json)
    -> ChromaDB Persistent Collection

This stage deliberately does NOT:
    - load the sentence-transformers embedding model
    - recompute embeddings from chunk text
    - perform similarity search / retrieval
    - call an LLM
    - create a Streamlit UI

Those responsibilities belong to other stages. If this file finds itself
importing SentenceTransformer, something has gone wrong.

Run independently with:
    python 05_create_chroma_store.py
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import config

REQUIRED_EMBEDDING_FIELDS = ("chunk_id", "embedding", "text")


def load_embeddings(input_path: str) -> Dict[str, Any]:
    """
    Load the embeddings JSON payload produced by 04_vector_representation.py.

    Args:
        input_path: Path to the embeddings JSON file
            (data/processed/grammar_correction_embeddings.json).

    Returns:
        The parsed JSON payload as a dict, expected to contain at least
        an "embeddings" list.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the file is not valid JSON or lacks the expected
            top-level structure.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Embeddings input file not found at '{input_path}'. "
            "Run 04_vector_representation.py first to generate it."
        )

    with open(input_path, "r", encoding="utf-8") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Embeddings input file is not valid JSON: {exc}"
            ) from exc

    if not isinstance(payload, dict) or "embeddings" not in payload:
        raise ValueError(
            "Embeddings input file does not match the expected top-level "
            "structure (missing an 'embeddings' list)."
        )
    if not isinstance(payload["embeddings"], list):
        raise ValueError(
            "'embeddings' in the embeddings input file must be a list."
        )

    return payload


def _is_valid_embedding_record(
    record: Any, expected_dimension: Optional[int]
) -> Tuple[bool, Optional[int]]:
    """
    Check whether a single embedding record is well-formed.

    Args:
        record: A single entry from the embeddings payload.
        expected_dimension: The dimension established by previously seen
            valid records in this run, or None if this is the first one.

    Returns:
        A tuple of (is_valid, dimension). `dimension` is the record's
        embedding length when valid, otherwise None.
    """
    if not isinstance(record, dict):
        return False, None

    for field in REQUIRED_EMBEDDING_FIELDS:
        if field not in record:
            return False, None

    chunk_id = record["chunk_id"]
    embedding = record["embedding"]
    text = record["text"]

    if not isinstance(chunk_id, str) or chunk_id.strip() == "":
        return False, None
    if not isinstance(text, str) or text.strip() == "":
        return False, None
    if not isinstance(embedding, list) or len(embedding) == 0:
        return False, None
    if not all(isinstance(value, (int, float)) for value in embedding):
        return False, None

    dimension = len(embedding)
    if expected_dimension is not None and dimension != expected_dimension:
        return False, None

    return True, dimension


def validate_embedding_records(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, Optional[int]]:
    """
    Filter embedding records down to well-formed, dimensionally consistent
    entries, rejecting anything malformed rather than inserting it.

    The first valid record encountered establishes the expected embedding
    dimension; any later record with a different dimension is rejected
    (a mixed-dimension collection would silently corrupt similarity
    search, so it is treated as invalid rather than tolerated).

    Args:
        records: Raw list of embedding record dicts.

    Returns:
        A tuple of (valid_records, num_skipped, embedding_dimension).
        embedding_dimension is None if no valid records were found.
    """
    valid_records: List[Dict[str, Any]] = []
    num_skipped = 0
    embedding_dimension: Optional[int] = None

    for record in records:
        is_valid, dimension = _is_valid_embedding_record(
            record, embedding_dimension
        )
        if not is_valid:
            num_skipped += 1
            continue
        if embedding_dimension is None:
            embedding_dimension = dimension
        valid_records.append(record)

    return valid_records, num_skipped, embedding_dimension


def prepare_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a single embedding record's metadata into a ChromaDB-compatible
    flat dict (ChromaDB metadata values must be str, int, float, or bool;
    nested dicts are not allowed).

    The original incorrect/correct sentence pair is pulled up to the
    top level of the flattened metadata (alongside the nested
    dataset_name / record_id / source fields) so the pair remains
    recoverable directly from a ChromaDB query result, without needing
    to re-parse the stored document text.

    Args:
        record: A validated embedding record, expected to optionally
            contain "incorrect", "correct", and a nested "metadata" dict.

    Returns:
        A flat dict containing whichever of the following are available:
        dataset_name, record_id, source, incorrect, correct. Missing or
        non-scalar values are omitted rather than silently coerced into
        something misleading.
    """
    flattened: Dict[str, Any] = {}

    nested_metadata = record.get("metadata")
    if isinstance(nested_metadata, dict):
        for key in ("dataset_name", "record_id", "source"):
            value = nested_metadata.get(key)
            if isinstance(value, (str, int, float, bool)):
                flattened[key] = value

    for key in ("incorrect", "correct"):
        value = record.get(key)
        if isinstance(value, str) and value.strip() != "":
            flattened[key] = value

    return flattened


def initialize_chroma_client(db_path: str):
    """
    Initialize a persistent ChromaDB client rooted at `db_path`.

    Using `PersistentClient` (rather than the default in-memory client)
    is what makes the store survive process restarts: ChromaDB writes its
    index and metadata to `db_path` on disk, and reopening a client with
    the same path reloads exactly what was there before.

    Args:
        db_path: Directory to store the persistent ChromaDB database in.
            Created automatically by ChromaDB if it does not yet exist.

    Returns:
        A `chromadb.PersistentClient` instance.

    Raises:
        RuntimeError: If the `chromadb` package is missing or the client
            cannot be initialized (e.g. a permissions or corrupt-store
            issue).
    """
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "The 'chromadb' package is not installed. "
            "Install it with: pip install chromadb"
        ) from exc

    try:
        os.makedirs(db_path, exist_ok=True)
        client = chromadb.PersistentClient(path=db_path)
    except Exception as exc:  # noqa: BLE001 - surface any init failure clearly
        raise RuntimeError(
            f"Failed to initialize a persistent ChromaDB client at "
            f"'{db_path}'. Original error: {exc}"
        ) from exc

    return client


def get_or_create_collection(client, collection_name: str):
    """
    Open the target collection, creating it only if it does not exist yet.

    Behavior by starting state:
        - Database does not exist yet: `PersistentClient` creates the
          on-disk store at `db_path` the first time it is used.
        - Collection does not exist yet: `get_or_create_collection`
          creates it fresh, configured for cosine distance (matching the
          L2-normalized embeddings produced in Phase 6).
        - Collection already exists: the existing collection is opened
          as-is; nothing about it is reset or recreated.

    Args:
        client: A `chromadb.PersistentClient` instance.
        collection_name: Name of the collection to open or create.

    Returns:
        A ChromaDB `Collection` handle.

    Raises:
        RuntimeError: If the collection cannot be opened or created.
    """
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:  # noqa: BLE001 - surface any creation failure clearly
        raise RuntimeError(
            f"Failed to get or create collection '{collection_name}'. "
            f"Original error: {exc}"
        ) from exc

    return collection


def upsert_embeddings_in_batches(
    collection,
    records: List[Dict[str, Any]],
    batch_size: int,
) -> int:
    """
    Upsert validated embedding records into a ChromaDB collection in
    fixed-size batches.

    Upsert (rather than plain insert) is what makes this safe to run
    repeatedly: ChromaDB's `upsert` treats `chunk_id` as the record's
    stable primary key. If a `chunk_id` is not yet present, it is
    inserted; if it already exists, its embedding/document/metadata are
    overwritten in place rather than duplicated. Running this script
    twice on an unchanged embeddings file therefore leaves the collection
    count unchanged on the second run.

    Args:
        collection: A ChromaDB `Collection` handle.
        records: Validated embedding records (chunk_id, embedding, text,
            plus optional incorrect/correct/metadata).
        batch_size: Number of records written per `upsert()` call.

    Returns:
        The total number of records upserted.

    Raises:
        RuntimeError: If any batch fails to upsert.
    """
    total_upserted = 0

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]

        ids = [record["chunk_id"] for record in batch]
        embeddings = [record["embedding"] for record in batch]
        documents = [record["text"] for record in batch]
        metadatas = [prepare_metadata(record) for record in batch]

        try:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as exc:  # noqa: BLE001 - surface any upsert failure clearly
            raise RuntimeError(
                f"Failed to upsert batch starting at index {start} "
                f"(size {len(batch)}). Original error: {exc}"
            ) from exc

        total_upserted += len(batch)

    return total_upserted


def verify_collection(
    collection,
    expected_count: int,
    sample_chunk_id: Optional[str],
) -> Dict[str, Any]:
    """
    Run a lightweight post-build sanity check against the collection.

    Checks performed:
        1. The collection's reported count matches `expected_count`.
        2. If a sample chunk_id is provided, it can be fetched back by ID,
           with its document text and metadata (including incorrect /
           correct, when present) intact.

    Args:
        collection: A ChromaDB `Collection` handle.
        expected_count: The record count the collection should report
            after upserting (i.e. its own `.count()` after this run).
        sample_chunk_id: A chunk_id to fetch back as a spot-check, or
            None to skip the by-ID fetch check.

    Returns:
        A dict summarizing verification results:
            {
                "count_matches": bool,
                "actual_count": int,
                "sample_fetch_ok": Optional[bool],
                "sample_has_incorrect": Optional[bool],
                "sample_has_correct": Optional[bool],
            }
    """
    actual_count = collection.count()
    result: Dict[str, Any] = {
        "count_matches": actual_count == expected_count,
        "actual_count": actual_count,
        "sample_fetch_ok": None,
        "sample_has_incorrect": None,
        "sample_has_correct": None,
    }

    if sample_chunk_id is not None:
        fetched = collection.get(
            ids=[sample_chunk_id],
            include=["documents", "metadatas", "embeddings"],
        )
        fetched_ids = fetched.get("ids") or []
        result["sample_fetch_ok"] = len(fetched_ids) == 1 and fetched_ids[0] == sample_chunk_id

        if result["sample_fetch_ok"]:
            metadatas = fetched.get("metadatas") or [{}]
            sample_metadata = metadatas[0] or {}
            result["sample_has_incorrect"] = "incorrect" in sample_metadata
            result["sample_has_correct"] = "correct" in sample_metadata

    return result


def main() -> None:
    """
    Entry point: load precomputed embeddings, validate them, upsert them
    into a persistent ChromaDB collection, verify the result, and print a
    run summary.
    """
    input_path = config.EMBEDDINGS_DATA_PATH
    db_path = config.CHROMA_DB_PATH
    collection_name = config.CHROMA_COLLECTION_NAME
    batch_size = config.CHROMA_UPSERT_BATCH_SIZE

    print("=" * 70)
    print("STAGE 5: Persistent ChromaDB Vector Store")
    print("=" * 70)
    print(f"Input path:        {input_path}")
    print(f"ChromaDB path:     {db_path}")
    print(f"Collection name:   {collection_name}")
    print(f"Batch size:        {batch_size}")
    print("-" * 70)

    try:
        payload = load_embeddings(input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    raw_records = payload["embeddings"]
    valid_records, num_skipped, embedding_dimension = validate_embedding_records(
        raw_records
    )

    print(f"Input embeddings:          {len(raw_records)}")
    print(f"Malformed records skipped: {num_skipped}")
    print(f"Valid records:             {len(valid_records)}")
    print(f"Embedding dimension:       {embedding_dimension}")
    print("-" * 70)

    if len(valid_records) == 0:
        print("ERROR: No valid embedding records available to store.", file=sys.stderr)
        sys.exit(1)

    try:
        client = initialize_chroma_client(db_path)
        collection = get_or_create_collection(client, collection_name)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        num_upserted = upsert_embeddings_in_batches(collection, valid_records, batch_size)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    sample_chunk_id = valid_records[0]["chunk_id"]
    verification = verify_collection(collection, len(valid_records), sample_chunk_id)

    print(f"Records upserted:         {num_upserted}")
    print(f"Final collection count:   {verification['actual_count']}")
    print(f"Count matches expected:   {verification['count_matches']}")
    print(f"Sample fetch by ID OK:    {verification['sample_fetch_ok']}")
    print(f"Sample has 'incorrect':   {verification['sample_has_incorrect']}")
    print(f"Sample has 'correct':     {verification['sample_has_correct']}")
    print("-" * 70)

    verification_passed = (
        verification["count_matches"]
        and verification["sample_fetch_ok"]
        and verification["sample_has_incorrect"]
        and verification["sample_has_correct"]
    )

    if not verification_passed:
        print(
            "ERROR: Post-build verification failed. The collection was "
            "written but does not match expectations; inspect the "
            "summary above.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Verification status:      PASSED")
    print("=" * 70)
    print("STAGE 5 complete.")


if __name__ == "__main__":
    main()
