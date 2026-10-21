from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "AAU Chatbot API"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = Path(__file__).resolve().parent.parent.parent / "data"
    ROOT_DATA_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent / "data"
    CHROMA_PERSIST_DIR: Path = Path(__file__).resolve().parent.parent.parent / "chroma_db"
    CHROMA_COLLECTION_NAME: str = "aau_documents"
    METADATA_STORE_PATH: Path = Path(__file__).resolve().parent.parent.parent / "documents_metadata.json"

    # LLM Settings
    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0

    # Ingestion & Chunking
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100
    MAX_UPLOAD_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
    ALLOWED_EXTENSIONS: set[str] = {".pdf", ".docx", ".txt", ".md"}

    # Retrieval & Evidence Verification Thresholds
    RETRIEVAL_TOP_K: int = 4
    SIMILARITY_DISTANCE_THRESHOLD: float = 1.35  # Chroma L2 or cosine distance threshold
    MIN_RELEVANCE_SCORE: float = 0.30
    LEXICAL_OVERLAP_MIN_TOKENS: int = 2

    # Refusal Message
    REFUSAL_MESSAGE: str = "I'm sorry, I couldn't find sufficient information about this in the available AAU documents."

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
