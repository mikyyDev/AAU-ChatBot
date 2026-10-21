import logging
from contextlib import asynccontextmanager

from chromadb.errors import ChromaError
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.core.config import settings
from app.db.session import get_vector_collection
from app.services.ingestion import get_ingestion_service

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("aau_chatbot")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Application startup and shutdown lifecycle.
    Automatically checks and indexes AAU documents from data folder if vector store is empty.
    """
    logger.info("Starting AAU Chatbot Backend API...")
    try:
        collection = get_vector_collection()
        doc_count = collection.count()
        logger.info(f"Vector store connected. Current indexed chunks: {doc_count}")

        # If empty, perform initial directory ingestion from data directories
        if doc_count == 0:
            logger.info("Vector index is empty. Scanning data directories for initial ingestion...")
            ingestion_service = get_ingestion_service()
            for target_dir in [settings.DATA_DIR, settings.ROOT_DATA_DIR]:
                if target_dir.exists():
                    total, succ, _ = ingestion_service.ingest_directory(target_dir)
                    if succ > 0:
                        logger.info(f"Auto-ingested {succ}/{total} documents from {target_dir}")
                        break
    except (ChromaError, HTTPException, OSError, RuntimeError, ValueError):
        logger.exception("Startup vector-store check or ingestion failed")

    yield
    logger.info("Shutting down AAU Chatbot Backend API.")


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Production-oriented RAG Chatbot for Addis Ababa University (AAU) with strict document grounding.",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS if settings.ALLOWED_ORIGINS else ["*"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core API routes under /api
app.include_router(chat_router, prefix="/api")
app.include_router(documents_router, prefix="/api")

# Legacy compatibility routes under /chat and /api
app.include_router(chat_router, prefix="/chat")
app.include_router(documents_router, prefix="/chat")


@app.get("/", tags=["Root"])
def root():
    return {
        "service": settings.APP_NAME,
        "status": "online",
        "docs_url": "/docs",
        "chat_endpoint": "/api/chat",
        "health_endpoint": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
