# AAU Chatbot Assistance

Production-style starter for an Addis Ababa University chatbot using:

- FastAPI backend with Retrieval-Augmented Generation (RAG)
- Next.js frontend with a clean chat experience
- Chroma vector database for local document retrieval

## Project Structure

```text
AAU-Chatbot-Assistance/
├── backend/
├── frontend/
├── data/                 # Put AAU documents here (pdf, txt, md)
├── docs/
├── .gitignore
├── README.md
└── docker-compose.yml
```

## Core Rule

The system answers only from your uploaded AAU documents.

If no relevant information is found, it returns:

`Sorry  i cant provide you from External Resource`

## Quick Start

### 1. Backend

```bash
cd backend
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on http://localhost:3000 and backend on http://localhost:8000.

## Ingestion Flow

1. Put AAU files in `/data`.
2. Call ingestion endpoint:

```bash
curl -X POST http://localhost:8000/chat/ingest
```

3. Start chatting from frontend.

## API Endpoints

- `GET /chat/health`
- `POST /chat/ingest`
- `POST /chat`
