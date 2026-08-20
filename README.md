# EMVAssist — AI-Powered Knowledge Assistant

**Field:** E-Payment Systems (EMV Specifications)

EMVAssist is an AI-powered **Retrieval-Augmented Generation (RAG)** assistant designed to help technical teams search and understand EMV specifications through natural-language questions.

It combines document processing, hybrid retrieval, local LLM inference, and source-grounded answers in a secure, containerized environment.

## ✨ Key Features

- 📄 **EMV Document Management** — Upload, process, and manage EMV specification documents.
- 🧩 **Section-Based Chunking** — Preserves the structure and context of technical specifications.
- 🔎 **Hybrid Retrieval** — Combines semantic search and BM25 with RRF and Cross-Encoder reranking.
- 🤖 **Grounded Q&A** — Generates answers based on retrieved EMV content with source information.
- 💬 **Conversational Chat** — Persistent conversations with contextual follow-up questions.
- 🔐 **Authentication & Roles** — Separate user and administrator capabilities.
- 🎙️ **Voice Interaction** — Speech-to-text and text-to-speech support.
- 📊 **Retrieval Monitoring** — Stores retrieval information for evaluation and analysis.
- 🏠 **Local AI Execution** — Uses local embeddings and LLM inference to keep EMV data within the local environment.

## 🏗️ Architecture

The solution is organized into five main layers:

1. **Data Sources** — EMV specification documents
2. **Document Processing** — Extraction, cleaning, section-based chunking, and embeddings
3. **Knowledge Store** — ChromaDB + PostgreSQL
4. **RAG Layer** — Hybrid retrieval → reranking → LLM
5. **User Interface** — Web application with text and voice interaction

### Query Flow

```text
User → Web UI
     → FastAPI
     → Hybrid Retrieval
     → RRF + Reranking
     → Top-K Chunks
     → Local LLM
     → Grounded Answer + Sources
```

## Tech Stack
- Python
- FastAPI + Uvicorn
- HTML / CSS / JavaScript
- Docling
- LangChain
- ChromaDB
- PostgreSQL
- Sentence Transformers
- BM25
- Cross-Encoder
- Ollama + Qwen3:8B
- Faster-Whisper
- Docker + Docker Compose


## Quickstart
### Install Docker Desktop

Before cloning the repository, download and install **Docker Desktop**:

👉 https://www.docker.com/products/docker-desktop/

Make sure Docker is running before proceeding.


### 1) Clone the repository
```bash
git clone https://github.com/ayoubrekik/EMV-Knowledge-assistant.git
cd EMV-Knowledge-assistant
```

### 2) Add your documents
Put your PDFs into:
```
src/data/
```

### 3) Start the stack
```bash
docker compose up --build
```

- API will run at:  
  `http://localhost:8000`



## 🔒 Data Privacy
- Runs **fully locally** (Docker)
- EMV specs remain on your machine
- No external cloud API required


## 🙋 Author
**Ayoub Rekik**

Data Science & Artificial Intelligence Engineering

AI Knowledge Assistant for EMV E-Payment Systems.
