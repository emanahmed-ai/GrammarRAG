"""
06_retrieve_context.py

STAGE 6 of the RAG pipeline: Semantic Retrieval.

Purpose
-------
Given a brand-new, unseen user sentence, embed it with the SAME
Sentence-Transformers model used in Stage 4 (04_vector_representation.py), query the persistent
ChromaDB collection built in Stage 5 (05_create_chroma_store.py), and return the most semantically
similar grammar-correction examples.

This stage deliberately does NOT:
    - use keyword or exact string matching
    - implement BM25 or any hybrid search
    - rebuild the ChromaDB collection
    - regenerate stored document embeddings
    - call an LLM
    - build a prompt template
    - create a Streamlit UI

Those responsibilities belong to other stages (or, for keyword/hybrid
search, were explicitly excluded from this project's architecture).

Run independently with:
    python 06_retrieve_context.py
"""

import sys
from typing import Any, Dict, List, Optional

import config

# Module-level cache so the embedding model and ChromaDB collection are
# each initialized at most once per process, no matter how many times
# retrieve_context() is called (e.g. from a Streamlit app handling many
# user queries in a single session).
_MODEL_CACHE: Dict[str, Any] = {}
_COLLECTION_CACHE: Dict[str, Any] = {}


def load_embedding_model(model_name: str, device: str):
    """
    Load the Sentence-Transformers embedding model once and cache it.

    Reuses the exact same model name and device used in
    04_vector_representation.py, so query embeddings live in the same
    vector space as the stored document embeddings. A second call with
    the same model_name/device returns the cached instance instead of
    reloading from disk.

    Args:
        model_name: Hugging Face model identifier, e.g.
            "sentence-transformers/all-MiniLM-L6-v2".
        device: Device to load the model on (this project uses "cpu").

    Returns:
        A loaded `SentenceTransformer` instance.

    Raises:
        RuntimeError: If the `sentence-transformers` package is missing,
            or if the model cannot be loaded.
    """
    cache_key = f"{model_name}::{device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

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
            f"'{device}'. Original error: {exc}"
        ) from exc

    _MODEL_CACHE[cache_key] = model
    return model


def initialize_chroma_client(db_path: str):
    """
    Initialize a ChromaDB client pointed at the persistent store created
    in Stage 5 (05_create_chroma_store.py).

    Args:
        db_path: Directory the persistent ChromaDB database lives in.

    Returns:
        A `chromadb.PersistentClient` instance.

    Raises:
        RuntimeError: If the `chromadb` package is missing, or the client
            cannot be initialized.
    """
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "The 'chromadb' package is not installed. "
            "Install it with: pip install chromadb"
        ) from exc

    try:
        client = chromadb.PersistentClient(path=db_path)
    except Exception as exc:  # noqa: BLE001 - surface any init failure clearly
        raise RuntimeError(
            f"Failed to open the persistent ChromaDB store at '{db_path}'. "
            f"Original error: {exc}"
        ) from exc

    return client


def get_collection(db_path: str, collection_name: str):
    """
    Open the existing Stage 5 collection, caching it per (db_path,
    collection_name) pair so repeated retrieve_context() calls do not
    reopen the client/collection every time.

    This function deliberately uses `get_collection`, not
    `get_or_create_collection`: retrieval is read-only against a store
    that 05_create_chroma_store.py is responsible for creating. If the collection does not
    exist yet, that is a configuration problem this stage should surface
    clearly rather than silently paper over by creating an empty one.

    Args:
        db_path: Directory the persistent ChromaDB database lives in.
        collection_name: Name of the collection to open.

    Returns:
        A ChromaDB `Collection` handle.

    Raises:
        RuntimeError: If the ChromaDB path/collection does not exist or
            cannot be opened, with a message distinguishing the two cases.
    """
    import os

    cache_key = f"{db_path}::{collection_name}"
    if cache_key in _COLLECTION_CACHE:
        return _COLLECTION_CACHE[cache_key]

    if not os.path.isdir(db_path):
        raise RuntimeError(
            f"ChromaDB path '{db_path}' does not exist. "
            "Run 05_create_chroma_store.py first to build the vector store."
        )

    client = initialize_chroma_client(db_path)

    try:
        collection = client.get_collection(name=collection_name)
    except Exception as exc:  # noqa: BLE001 - chromadb raises varied exceptions
        raise RuntimeError(
            f"Collection '{collection_name}' does not exist in the "
            f"ChromaDB store at '{db_path}'. Run 05_create_chroma_store.py "
            f"first to build it. Original error: {exc}"
        ) from exc

    _COLLECTION_CACHE[cache_key] = collection
    return collection


def embed_query(model, query: str, normalize: bool) -> List[float]:
    """
    Encode a single user query into an embedding vector.

    Uses the same `normalize_embeddings` behavior as
    04_vector_representation.py (config.EMBEDDING_NORMALIZE), so the
    query vector lives in the same normalized space as the stored
    document vectors and cosine similarity via ChromaDB's dot product
    remains valid.

    Args:
        model: A loaded `SentenceTransformer` instance.
        query: The raw user query string.
        normalize: Whether to L2-normalize the resulting embedding.

    Returns:
        The query embedding as a plain Python list of floats.

    Raises:
        RuntimeError: If encoding fails.
    """
    try:
        embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )[0]
    except Exception as exc:  # noqa: BLE001 - surface any encoding failure clearly
        raise RuntimeError(f"Failed to embed query: {exc}") from exc

    return embedding.tolist()


def format_results(chroma_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert ChromaDB's nested, batch-oriented query response into a flat
    list of result dicts, one per retrieved chunk.

    ChromaDB's `.query()` returns each field (ids, documents, metadatas,
    distances) as a list-of-lists, one outer list per query embedding
    submitted. This project only ever submits one query embedding at a
    time, so this function reaches into index 0 of each outer list and
    then zips the inner lists together by position, since ChromaDB
    guarantees they are aligned by index within a single query's results.

    Args:
        chroma_result: The raw dict returned by `collection.query(...)`.

    Returns:
        A list of dicts, ranked in the order ChromaDB returned them
        (most semantically similar first):
            {
                "chunk_id": str,
                "text": str,
                "incorrect": str,
                "correct": str,
                "metadata": {"dataset_name": ..., "record_id": ..., "source": ...},
                "distance": float,
            }
    """
    ids = (chroma_result.get("ids") or [[]])[0]
    documents = (chroma_result.get("documents") or [[]])[0]
    metadatas = (chroma_result.get("metadatas") or [[]])[0]
    distances = (chroma_result.get("distances") or [[]])[0]

    results: List[Dict[str, Any]] = []
    for index, chunk_id in enumerate(ids):
        raw_metadata = metadatas[index] if index < len(metadatas) else {}
        raw_metadata = raw_metadata or {}

        results.append(
            {
                "chunk_id": chunk_id,
                "text": documents[index] if index < len(documents) else "",
                "incorrect": raw_metadata.get("incorrect", ""),
                "correct": raw_metadata.get("correct", ""),
                "metadata": {
                    "dataset_name": raw_metadata.get("dataset_name"),
                    "record_id": raw_metadata.get("record_id"),
                    "source": raw_metadata.get("source"),
                },
                "distance": distances[index] if index < len(distances) else None,
            }
        )

    return results


def retrieve_context(
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Retrieve the most semantically similar grammar-correction examples
    for a new, unseen user sentence.

    Pipeline: query text -> query embedding (same model/normalization as
    Stage 4) -> ChromaDB semantic similarity search against the Stage 5
    collection -> flattened, aligned results.

    Args:
        query: A non-empty user sentence to find similar grammar
            correction examples for.
        top_k: Maximum number of results to return. Must be a positive
            integer. If it exceeds the collection's size, it is
            transparently capped to the collection size (see note below).

    Returns:
        A list of result dicts (see format_results()), ranked by
        semantic similarity, most similar first. Returns an empty list
        if the collection is empty.

    Raises:
        ValueError: If `query` is not a non-empty string, or `top_k` is
            not a positive integer.
        RuntimeError: If the embedding model or ChromaDB collection
            cannot be loaded/opened.

    Note on top_k capping:
        ChromaDB already returns at most `collection.count()` results
        even if a larger `n_results` is requested, so this would not
        crash on its own. This function caps explicitly anyway so the
        returned list length is always predictable up front and the
        collection is queried with a value that reflects what can
        actually be returned, rather than relying on ChromaDB's
        internal leniency.
    """
    if not isinstance(query, str) or query.strip() == "":
        raise ValueError("query must be a non-empty string.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    collection = get_collection(config.CHROMA_DB_PATH, config.CHROMA_COLLECTION_NAME)

    collection_size = collection.count()
    if collection_size == 0:
        return []

    effective_top_k = min(top_k, collection_size)

    model = load_embedding_model(config.EMBEDDING_MODEL_NAME, config.EMBEDDING_DEVICE)
    query_embedding = embed_query(model, query, config.EMBEDDING_NORMALIZE)

    try:
        chroma_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:  # noqa: BLE001 - surface any query failure clearly
        raise RuntimeError(f"ChromaDB query failed: {exc}") from exc

    return format_results(chroma_result)


def _print_retrieval_demo(query: str, top_k: int) -> None:
    """
    Run retrieve_context() for a single query and print a readable
    CLI report of the results (rank, distance, incorrect/correct pair).

    Args:
        query: The demo query to retrieve context for.
        top_k: Number of results to request.
    """
    print("-" * 70)
    print(f"Query:            {query}")

    try:
        results = retrieve_context(query, top_k=top_k)
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR:            {exc}")
        return

    print(f"Results returned: {len(results)}")

    if not results:
        print("No results (the collection is empty).")
        return

    for rank, result in enumerate(results, start=1):
        print(f"  Rank {rank}:")
        print(f"    Distance:    {result['distance']}")
        print(f"    Incorrect:   {result['incorrect']}")
        print(f"    Correct:     {result['correct']}")


def main() -> None:
    """
    Entry point: demonstrate semantic retrieval against a couple of
    unseen example queries when run directly.
    """
    print("=" * 70)
    print("STAGE 6: Semantic Retrieval")
    print("=" * 70)
    print(f"ChromaDB path:     {config.CHROMA_DB_PATH}")
    print(f"Collection name:   {config.CHROMA_COLLECTION_NAME}")
    print(f"Embedding model:   {config.EMBEDDING_MODEL_NAME}")
    print(f"Default top_k:     {config.DEFAULT_TOP_K}")

    try:
        collection = get_collection(config.CHROMA_DB_PATH, config.CHROMA_COLLECTION_NAME)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Collection size:   {collection.count()}")

    example_queries = [
        "He go to school every day.",
        "She have finished her homework.",
    ]

    for query in example_queries:
        _print_retrieval_demo(query, config.DEFAULT_TOP_K)

    print("-" * 70)
    print("=" * 70)
    print("STAGE 6 complete.")


if __name__ == "__main__":
    main()
