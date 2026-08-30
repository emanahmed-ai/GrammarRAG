"""
config.py

Central configuration for the Grammar & Syntax RAG pipeline.

This file holds constants that are shared across MULTIPLE pipeline stages
(01_documents.py, 05_create_chroma_store.py, 06_retrieve_context.py, etc.)
so that behavior stays consistent without editing every file individually.

Central configuration for the complete Grammar & Syntax RAG pipeline.

This file is the single source of truth for shared configuration values
used across all pipeline stages, including dataset ingestion, preprocessing,
chunking, embeddings, ChromaDB retrieval, OpenRouter prompting, and
Streamlit deployment.
"""

import os

# ---------------------------------------------------------------------------
# Dataset settings
# ---------------------------------------------------------------------------

# Hugging Face dataset identifier (must be loaded programmatically, never
# hardcoded record-by-record).
DATASET_NAME: str = "agentlans/grammar-correction"

# Default split used for building the retrieval knowledge base.
DATASET_SPLIT: str = "train"

# Maximum number of records to load from the dataset.
#
# Development mode:
#   Set this to a large number (or None) to use the full dataset locally,
#   where storage and compute are not constrained.
#
# Deployment mode (e.g. Streamlit Cloud):
#   Keep this capped (e.g. 5,000-10,000) so that downstream embedding
#   generation and the persisted ChromaDB stay small enough to ship in a
#   repository and run within free-tier compute/memory limits.
#
# Set to None to disable the cap and load every available record.
MAX_RECORDS: int = 10000

# ---------------------------------------------------------------------------
# Path settings
# ---------------------------------------------------------------------------

# Root directory of the project (folder containing this config.py file).
PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))

# Raw ingestion output (Stage 1 output).
RAW_DATA_DIR: str = os.path.join(PROJECT_ROOT, "data", "raw")
RAW_DATA_PATH: str = os.path.join(RAW_DATA_DIR, "grammar_correction_raw.json")

# Preprocessed / cleaned output (Stage 2 output).
PROCESSED_DATA_DIR: str = os.path.join(PROJECT_ROOT, "data", "processed")
PROCESSED_DATA_PATH: str = os.path.join(
    PROCESSED_DATA_DIR, "grammar_correction_preprocessed.json"
)

# Chunked output (Stage 3 output). Chunks live alongside the preprocessed
# file in the same "processed" directory.
CHUNKS_DATA_PATH: str = os.path.join(
    PROCESSED_DATA_DIR, "grammar_correction_chunks.json"
)

# Embedded output (Stage 4 output). Embeddings live alongside the chunks
# file in the same "processed" directory.
EMBEDDINGS_DATA_PATH: str = os.path.join(
    PROCESSED_DATA_DIR, "grammar_correction_embeddings.json"
)

# ---------------------------------------------------------------------------
# Embedding settings (Stage 4 - Vector Representation)
# ---------------------------------------------------------------------------

# Sentence-Transformers model used to embed grammar-correction chunks.
# Kept as a single source of truth so no other file hardcodes the model
# name (04_vector_representation.py and, later, 06_retrieve_context.py
# both read this value).
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# Number of chunk texts encoded per model.encode() call. Batching keeps
# memory bounded on CPU while still avoiding a separate model call per
# chunk.
EMBEDDING_BATCH_SIZE: int = 32

# Device used for embedding generation. This project targets CPU-only
# environments (e.g. Streamlit Cloud free tier), so this stays "cpu"
# rather than auto-detecting a GPU.
EMBEDDING_DEVICE: str = "cpu"

# Whether embeddings are L2-normalized after encoding. Normalized
# embeddings turn cosine similarity into a dot product, which is what
# ChromaDB's default "cosine" distance space expects. This must stay
# consistent between Stage 4 (here) and Stage 5 (ChromaDB creation) and
# Stage 6 (query-time embedding), so it lives here as a single flag.
EMBEDDING_NORMALIZE: bool = True

# ---------------------------------------------------------------------------
# ChromaDB settings (Stage 5 - Persistent Vector Store)
# ---------------------------------------------------------------------------

# On-disk location of the persistent ChromaDB store. Using a directory
# under the project root (rather than a temp dir) means the store survives
# process restarts, matching the "persistent local storage" requirement.
CHROMA_DB_PATH: str = os.path.join(PROJECT_ROOT, "chroma_db")

# Name of the ChromaDB collection holding grammar-correction chunks.
# Kept as a single source of truth so 05_create_chroma_store.py and,
# later, 06_retrieve_context.py always open the same collection.
CHROMA_COLLECTION_NAME: str = "grammar_correction_chunks"

# Number of records written per collection.upsert() call. Batches keep a
# single call bounded even when the dataset grows into the thousands.
CHROMA_UPSERT_BATCH_SIZE: int = 500

# ---------------------------------------------------------------------------
# Retrieval settings (Stage 6 - Semantic Retrieval)
# ---------------------------------------------------------------------------

# Default number of results retrieve_context() returns when the caller
# does not specify top_k explicitly.
DEFAULT_TOP_K: int = 5

# ---------------------------------------------------------------------------
# OpenRouter settings (Stage 7 - RAG Prompting + LLM Generation)
# ---------------------------------------------------------------------------

# OpenRouter chat completions endpoint. Not a secret, safe to hardcode.
OPENROUTER_API_URL: str = "https://openrouter.ai/api/v1/chat/completions"

# LLM used for grammar analysis and correction. Matches the model already
# named in the project's submission instructions (Streamlit secrets
# template: OPENROUTER_MODEL = "openai/gpt-4o-mini"). Configurable via the
# OPENROUTER_MODEL environment variable / Streamlit secret without editing
# code, but defaults here so every stage that needs it reads one source of
# truth instead of hardcoding the name separately.
OPENROUTER_MODEL: str = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# Request timeout for OpenRouter API calls, in seconds.
OPENROUTER_TIMEOUT_SECONDS: int = 30

# NOTE: OPENROUTER_API_KEY is deliberately NOT defined here. Per the
# project's API key rules, it must come only from the OPENROUTER_API_KEY
# environment variable (or Streamlit secrets at deploy time), never from
# a hardcoded value in this file.

# ---------------------------------------------------------------------------
# Metadata settings
# ---------------------------------------------------------------------------

SOURCE_LABEL: str = "huggingface"
