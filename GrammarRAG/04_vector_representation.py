"""
04_vector_representation.py

STAGE 4 of the RAG pipeline: Vector Representation.

Purpose
-------
Load the semantic grammar-correction chunks produced by 03_chunking.py,
convert each chunk's text into a numerical embedding using a local
Sentence-Transformers model, and persist the result as JSON for the next
stage (05_create_chroma_store.py).

Design principle
----------------
EACH CHUNK EMBEDS THE COMPLETE (INCORRECT, CORRECT) PAIR, NOT JUST ONE SIDE.

The chunk text produced by 03_chunking.py already encodes both the
ungrammatical sentence and its correction in one labeled block:

    "Original Sentence:\\n<incorrect>\\n\\nCorrected Sentence:\\n<correct>"

Embedding only the incorrect sentence would let the retriever match
similar-looking errors but lose the correction itself. Embedding only the
corrected sentence would let it match fluent English but lose the error
pattern a user's mistake needs to be matched against. Embedding the whole
pair preserves the contrast between the two, which is exactly what this
RAG system needs to retrieve: "here is an error pattern similar to yours,
and here is how it was fixed."

This stage deliberately does NOT:
    - re-chunk or re-split text
    - correct grammar or spelling
    - call an LLM
    - create a ChromaDB store
    - perform similarity search

Those responsibilities belong to other stages.

Run independently with:
    python 04_vector_representation.py
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

import config

REQUIRED_CHUNK_FIELDS = ("chunk_id", "text", "incorrect", "correct", "metadata")


def load_chunks(input_path: str) -> Dict[str, Any]:
    """
    Load the chunked JSON payload produced by 03_chunking.py.

    Args:
        input_path: Path to the chunks JSON file
            (data/processed/grammar_correction_chunks.json).

    Returns:
        The parsed JSON payload as a dict, expected to contain at least
        a "chunks" list.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the file is not valid JSON or lacks the expected
            top-level structure.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Chunks input file not found at '{input_path}'. "
            "Run 03_chunking.py first to generate it."
        )

    with open(input_path, "r", encoding="utf-8") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Chunks input file is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict) or "chunks" not in payload:
        raise ValueError(
            "Chunks input file does not match the expected top-level "
            "structure (missing a 'chunks' list)."
        )
    if not isinstance(payload["chunks"], list):
        raise ValueError("'chunks' in the chunks input file must be a list.")

    return payload


def _is_valid_chunk(chunk: Any) -> bool:
    """
    Check whether a chunk has the required structure and non-empty text.

    Args:
        chunk: A single entry from the chunks file.

    Returns:
        True if the chunk is usable for embedding, False otherwise.
    """
    if not isinstance(chunk, dict):
        return False

    for field in REQUIRED_CHUNK_FIELDS:
        if field not in chunk:
            return False

    if not isinstance(chunk["chunk_id"], str) or chunk["chunk_id"].strip() == "":
        return False
    if not isinstance(chunk["text"], str) or chunk["text"].strip() == "":
        return False
    if not isinstance(chunk["incorrect"], str) or not isinstance(chunk["correct"], str):
        return False
    if not isinstance(chunk["metadata"], dict):
        return False

    return True


def filter_valid_chunks(chunks: List[Dict[str, Any]]) -> "tuple[List[Dict[str, Any]], int]":
    """
    Keep only well-formed chunks, skipping anything malformed.

    Args:
        chunks: Raw list of chunk dicts from the chunks payload.

    Returns:
        A tuple of (valid_chunks, num_skipped).
    """
    valid_chunks: List[Dict[str, Any]] = []
    num_skipped = 0

    for chunk in chunks:
        if not _is_valid_chunk(chunk):
            num_skipped += 1
            continue
        valid_chunks.append(chunk)

    return valid_chunks, num_skipped


def load_embedding_model(model_name: str, device: str):
    """
    Load the Sentence-Transformers embedding model once.

    Args:
        model_name: Hugging Face model identifier, e.g.
            "sentence-transformers/all-MiniLM-L6-v2".
        device: Device to load the model on (this project uses "cpu").

    Returns:
        A loaded `SentenceTransformer` instance, ready for repeated
        `.encode()` calls.

    Raises:
        RuntimeError: If the `sentence-transformers` package is missing,
            or if the model cannot be loaded (e.g. no network access to
            download model weights).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "The 'sentence-transformers' package is not installed. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as exc:  # noqa: BLE001 - surface any loader failure clearly
        raise RuntimeError(
            f"Failed to load embedding model '{model_name}' on device "
            f"'{device}'. This usually means the model weights could not "
            f"be downloaded (no network access) or the model name is "
            f"invalid. Original error: {exc}"
        ) from exc

    return model


def generate_embeddings(
    model,
    texts: List[str],
    batch_size: int,
    normalize: bool,
) -> List[List[float]]:
    """
    Encode a list of chunk texts into embeddings using a pre-loaded model.

    The model is loaded once by the caller (see load_embedding_model) and
    passed in here; this function never re-instantiates it, so repeated
    calls (or a loop over batches) never pay a reload cost.

    Args:
        model: A loaded `SentenceTransformer` instance.
        texts: List of chunk texts to embed, in the same order as the
            chunks they came from.
        batch_size: Number of texts encoded per internal batch.
        normalize: If True, L2-normalize each embedding vector so cosine
            similarity reduces to a dot product.

    Returns:
        A list of embeddings (one per input text, same order), each a
        plain Python list of floats so it is JSON-serializable.

    Raises:
        RuntimeError: If encoding fails for any reason.
    """
    try:
        raw_embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
    except Exception as exc:  # noqa: BLE001 - surface any encoding failure clearly
        raise RuntimeError(f"Failed to generate embeddings: {exc}") from exc

    return [embedding.tolist() for embedding in raw_embeddings]


def create_embedding_records(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> List[Dict[str, Any]]:
    """
    Zip each chunk together with its embedding, preserving a deterministic
    chunk_id -> embedding mapping.

    Args:
        chunks: Valid chunks, in the same order the texts were encoded in.
        embeddings: Embeddings produced by generate_embeddings(), in the
            same order as `chunks`.

    Returns:
        A list of embedding records:
            {
                "chunk_id": str,
                "embedding": List[float],
                "text": str,
                "incorrect": str,
                "correct": str,
                "metadata": dict,
            }

    Raises:
        ValueError: If the number of chunks and embeddings do not match,
            which would indicate the mapping is no longer trustworthy.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunk/embedding count mismatch: {len(chunks)} chunks vs "
            f"{len(embeddings)} embeddings. Refusing to zip a mapping "
            "that cannot be trusted to be deterministic."
        )

    records: List[Dict[str, Any]] = []
    for chunk, embedding in zip(chunks, embeddings):
        records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "embedding": embedding,
                "text": chunk["text"],
                "incorrect": chunk["incorrect"],
                "correct": chunk["correct"],
                "metadata": chunk["metadata"],
            }
        )

    return records


def save_embeddings(
    embedding_records: List[Dict[str, Any]],
    model_name: str,
    embedding_dimension: int,
    normalize: bool,
    device: str,
    output_path: str,
) -> None:
    """
    Persist embedding records to disk as JSON.

    Args:
        embedding_records: Records produced by create_embedding_records().
        model_name: Embedding model identifier, stored at the top level
            for traceability.
        embedding_dimension: Dimensionality of each embedding vector.
        normalize: Whether embeddings were L2-normalized.
        device: Device embeddings were generated on.
        output_path: Full file path to write the JSON output to.
    """
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    payload = {
        "model_name": model_name,
        "embedding_dimension": embedding_dimension,
        "normalized": normalize,
        "device": device,
        "num_embeddings": len(embedding_records),
        "embeddings": embedding_records,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    """
    Entry point: load chunks, load the embedding model once, generate
    embeddings in batches, save the embedded JSON, and print a run summary.
    """
    input_path = config.CHUNKS_DATA_PATH
    output_path = config.EMBEDDINGS_DATA_PATH
    model_name = config.EMBEDDING_MODEL_NAME
    batch_size = config.EMBEDDING_BATCH_SIZE
    device = config.EMBEDDING_DEVICE
    normalize = config.EMBEDDING_NORMALIZE

    print("=" * 70)
    print("STAGE 4: Vector Representation")
    print("=" * 70)
    print(f"Input path:    {input_path}")
    print(f"Model:         {model_name}")
    print(f"Batch size:    {batch_size}")
    print(f"Device:        {device}")
    print(f"Normalized:    {normalize}")
    print("-" * 70)

    try:
        payload = load_chunks(input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    raw_chunks = payload["chunks"]
    chunks, num_skipped = filter_valid_chunks(raw_chunks)

    print(f"Input chunks:              {len(raw_chunks)}")
    print(f"Malformed chunks skipped:  {num_skipped}")
    print(f"Chunks to embed:           {len(chunks)}")
    print("-" * 70)

    if len(chunks) == 0:
        print("ERROR: No valid chunks available to embed.", file=sys.stderr)
        sys.exit(1)

    try:
        model = load_embedding_model(model_name, device)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    texts = [chunk["text"] for chunk in chunks]

    try:
        embeddings = generate_embeddings(model, texts, batch_size, normalize)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    embedding_dimension = len(embeddings[0]) if embeddings else 0

    try:
        embedding_records = create_embedding_records(chunks, embeddings)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        save_embeddings(
            embedding_records,
            model_name,
            embedding_dimension,
            normalize,
            device,
            output_path,
        )
    except OSError as exc:
        print(f"ERROR: Failed to save output to '{output_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Embedding dimension:       {embedding_dimension}")
    print(f"Embeddings created:       {len(embedding_records)}")
    print(f"Output saved to:           {output_path}")
    print("-" * 70)
    print("Sample embedding record (vector truncated to first 5 values):")
    sample = dict(embedding_records[0])
    sample["embedding"] = sample["embedding"][:5] + ["..."]
    print(json.dumps(sample, ensure_ascii=False, indent=2))
    print("=" * 70)
    print("STAGE 4 complete.")


if __name__ == "__main__":
    main()
