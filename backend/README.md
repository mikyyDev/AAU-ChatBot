# AAU Chatbot Backend (Strict Document-Based RAG)

A production-oriented FastAPI backend for the **Addis Ababa University (AAU) Chatbot**, implementing strict document-grounded Retrieval-Augmented Generation (RAG).

The application answers factual questions **exclusively** using evidence from uploaded AAU documents and refuses to answer unsupported questions without guessing.

---

## 🏛️ Architecture Overview

```text
backend/
├── app/
│   ├── main.py                  # FastAPI application with lifecycle auto-ingestion & CORS
│   ├── api/
│   │   ├── chat.py              # Chat QA endpoint (/api/chat, /api) and health check (/api/health)
│   │   └── documents.py         # Document upload (/api/documents/upload), listing, deletion, bulk ingest
│   ├── core/
│   │   ├── config.py            # Pydantic Settings & calibrated retrieval thresholds
│   │   └── security.py          # Input sanitization, path-traversal prevention, RBAC, prompt-injection defense
│   ├── schemas/
│   │   ├── chat.py              # Strict ChatRequest/ChatResponse with dual frontend compatibility
│   │   └── documents.py         # Document upload, metadata, and ingestion schemas
│   ├── services/
│   │   ├── ingestion.py         # Multi-format document parser (PDF, DOCX, TXT, MD), chunker & vector indexer
│   │   ├── retriever.py         # Chroma vector store retriever with RBAC enforcement
│   │   ├── evidence_checker.py  # Standalone evidence validator, predicate verifier & refusal logic
│   │   ├── answer_generator.py  # LLM answering service (Groq / extractive fallback) with untrusted data wrappers
│   │   └── citation_validator.py# Citation grounding verifier and hallucination filter
│   ├── prompts/
│   │   └── system_prompt.py     # AAU strict document-only system prompt and canonical refusal message
│   └── db/
│       ├── models.py            # DocumentRecord & DocumentChunk models
│       ├── session.py           # ChromaDB client & high-performance zero-network embedding function
│       └── repositories.py      # Thread-safe persistent document metadata store
├── data/                        # AAU documents (.docx, .pdf, .txt, .md)
├── tests/
│   ├── conftest.py              # Isolated test fixtures and FastAPI test client
│   ├── test_ingestion.py        # Upload validation, corruption handling, path traversal defense
│   ├── test_retrieval.py        # Vector retrieval & RBAC permissions enforcement
│   ├── test_refusal.py          # Insufficient evidence, topic mismatch, general knowledge refusal
│   └── test_chat.py             # Directly answered, partial answers, conflicting docs, prompt injection
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🛡️ Core Rules & Refusal Guarantee

1. **Document-Grounded Only**: Answers are constructed solely from retrieved AAU document chunks.
2. **Prescribed Refusal**: If evidence is insufficient, absent, or out of scope, the system strictly returns:
   > *"I'm sorry, I couldn't find sufficient information about this in the available AAU documents."*
3. **Evidence Validation Component**: An independent evidence checker verifies that factual predicates (dates, deadlines, numbers, criteria) exist in the retrieved text before LLM generation.
4. **Prompt Injection Defense**: Retrieved text is treated as untrusted data, enclosed in strict boundary tags, and sanitized of adversarial directives.
5. **Role-Based Access Control (RBAC)**: Documents restricted to specific roles (e.g., `faculty`, `dean`, `admin`) are filtered at retrieval time and never leaked to unauthorized users.

---

## 🚀 Quick Start

### 1. Setup Virtual Environment

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and configure your API keys (optional; an offline extractive fallback is provided):

```powershell
copy .env.example .env
```

| Key | Description | Default |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | Optional Groq API Key for fast LLM inference | `None` (uses extractive fallback) |
| `GROQ_MODEL` | Groq LLM model name | `llama-3.3-70b-versatile` |
| `PORT` | API server port | `8000` |
| `ALLOWED_ORIGINS` | CORS allowed origins | `["http://localhost:3000","http://localhost:5173"]` |

### 3. Run Development Server

```powershell
uvicorn app.main:app --reload --port 8000
```

On startup, the server automatically scans and ingests documents located in `data/`.

---

## 📡 API Endpoints

### Chat
- **`POST /api/chat`** (alias: **`POST /api`**, **`POST /chat`**)
  - Request: `{"question": "When was AAU established?"}` (or `{"message": "..."}`)
  - Response:
    ```json
    {
      "answer": "Addis Ababa University (AAU) was established in 1950...",
      "sources": [
        {
          "document_id": "9aabb35b-...",
          "document_name": "About_AAU.docx",
          "page": null,
          "source": "About_AAU.docx",
          "score": 0.85
        }
      ],
      "grounded": true,
      "refusal": false,
      "found": true
    }
    ```

- **`GET /api/health`** (alias: **`GET /chat/health`**)
  - Returns service status, indexed document count, and vector chunk counts.

### Documents & Ingestion
- **`POST /api/documents/upload`**: Upload a new `.pdf`, `.docx`, `.txt`, or `.md` document.
- **`GET /api/documents`**: List metadata for all authorized documents.
- **`DELETE /api/documents/{document_id}`**: Delete a document and purge all its vector chunks.
- **`POST /api/ingest`**: Bulk ingest all documents from `backend/data` or root `data`.

---

## 🧪 Automated Test Suite

The test suite covers all 12 required scenarios from the Master Project Prompt:

```powershell
pytest tests/ -v
```

### Verified Test Scenarios
1. ✅ Question directly answered by an uploaded document.
2. ✅ Question with no relevant retrieved chunks (refused).
3. ✅ Question where retrieved chunks discuss the topic but do not answer it (refused).
4. ✅ Partially answerable question (answers supported part, explains missing info).
5. ✅ Unrelated general-knowledge questions (refused).
6. ✅ Conflicting information across documents (transparently reported).
7. ✅ Invalid or nonexistent citations (hallucinated sources rejected).
8. ✅ Prompt injection inside retrieved document text (neutralized).
9. ✅ Unauthorized access to restricted documents (RBAC enforcement).
10. ✅ Empty, corrupted, or unsupported upload (clean HTTP 400 errors).
11. ✅ Retrieval or model failure (returns safe fallback refusal).
12. ✅ Unsupported factual claim in response (grounding failure detected).
