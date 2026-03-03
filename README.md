# 📘 AI-Powered Knowledge Assistant (EMV)
**Field:** E‑Payment Systems (EMV Specifications)  

An AI-powered **Retrieval-Augmented Generation (RAG)** assistant that centralizes technical knowledge for project teams by aggregating content from **specification documents** (PDF/Word) into a **vector database** for semantic search and high-quality Q&A with **source citations**.

---

## ✨ Key Features
- **Automated knowledge ingestion** from:
  - Specification documents (PDF/Word technical specs)
- **Cleaning + chunking** to improve retrieval accuracy
- **Embeddings generation** for semantic understanding
- **Vector database storage** with metadata (source type, URL, date, section/page)
- **Natural language Q&A** with grounded answers + citations (links to sources)

---

## 🏗 Architecture Overview
The solution is structured into 5 layers:

1. **Data Sources** (Spec Docs)  
2. **Ingestion & Processing** (load → clean → chunk → embed)  
3. **Vector Database** (Knowledge Store)  
4. **AI / RAG Layer** (retrieve top‑K → prompt LLM → cite sources)  
5. **User Interface** (API / chat / web UI)

### Query Flow
```
User → UI/API
UI → RAG Engine
RAG Engine → Vector DB (semantic search)
Vector DB → Top‑K chunks
RAG Engine → LLM (question + chunks)
LLM → Answer + citations
```
---

## 🧰 Tech Stack
- **Python** (core development)
- **FastAPI + Uvicorn** (API service)
- **ChromaDB** (vector database; persisted via volume)
- **Sentence-Transformers / Embeddings** (local embedding generation)
- **Ollama** (local LLM runtime)
- **Docker + Docker Compose** (reproducible local environment)

---

## 🚀 Quickstart
## 🐳 Install Docker Desktop

Before cloning the repository, download and install **Docker Desktop**:

👉 https://www.docker.com/products/docker-desktop/

Make sure Docker is running before proceeding.

---

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


---

## 🔒 Data Privacy
- Runs **fully locally** (Docker)
- EMV specs remain on your machine
- No external cloud API required

---

## 🙋 Author
Internship / project work on an AI Knowledge Assistant for EMV E‑Payment Systems.
# EMV-Knowledge-assistant
