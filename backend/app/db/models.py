from datetime import datetime, timezone

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    """Metadata record for an uploaded document."""
    id: str = Field(..., description="Unique document ID (UUID).")
    filename: str = Field(..., description="Original or sanitized filename.")
    file_hash: str = Field(..., description="SHA-256 content hash for duplicate detection.")
    title: str = Field(..., description="Human-readable title or basename.")
    file_type: str = Field(..., description="Extension (.pdf, .docx, .txt, .md).")
    file_size: int = Field(..., description="File size in bytes.")
    chunk_count: int = Field(0, description="Total chunks indexed for this document.")
    access_roles: list[str] = Field(default_factory=lambda: ["public"], description="Allowed access roles.")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of document ingestion."
    )
    status: str = Field("ingested", description="Status: 'ingested', 'processing', 'failed'.")
    error_message: str | None = Field(None, description="Detailed error description if failed.")


class DocumentChunk(BaseModel):
    """Text chunk extracted and stored in the vector index."""
    chunk_id: str
    document_id: str
    document_name: str
    page: int | None = None
    chunk_index: int = 0
    content: str
    access_roles: list[str] = Field(default_factory=lambda: ["public"])
