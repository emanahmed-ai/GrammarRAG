# Grammar RAG Assistant

An advanced Retrieval-Augmented Generation (RAG) system built to provide intelligent grammar correction and explanations, leveraging vector databases and large language models within an interactive Streamlit web interface.

## 🚀 Features
* **Semantic Context Retrieval:** Utilizes FAISS and Sentence Transformers to retrieve relevant grammatical rules and contextual data.
* **Interactive Web App:** Built with Streamlit to offer a clean, user-friendly interface for real-time grammar checking.
* **Modular Pipeline:** Structured workflows spanning document preprocessing, chunking, vector representation, context retrieval, and response prompting.

## 🛠️ Tech Stack
* **Language:** Python
* **Web Framework:** Streamlit
* **Vector Indexing & Embeddings:** FAISS, Sentence Transformers
* **Environment Management:** VS Code, Virtual Environments (`venv`)

## 📂 Project Structure
```text
GrammarRAG/
│
├── data/                    # Dataset and reference documents
├── utils/                   # Helper scripts and pipelines
├── chroma_db/               # Local vector storage (ignored in git)
├── 01_documents.py          # Document ingestion
├── 02_preprocessing.py      # Text cleaning and preparation
├── 03_chunking.py           # Text splitting strategies
├── 04_vector_representation.py # Embedding generation
├── 05_create_chroma_...py   # Vector database setup
├── 06_retrieve_context.py   # Retrieval logic
├── 07_prompting.py          # LLM prompt construction
├── 08_evaluate_ifleq.py     # Evaluation scripts
├── config.py                # Configuration settings
├── requirements.txt         # Project dependencies
└── streamlit_app.py         # Main Streamlit application entry point