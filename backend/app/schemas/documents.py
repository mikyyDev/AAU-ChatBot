from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    status: str
    message: str


class DocumentMetadataResponse(BaseModel):
    document_id: str
    filename: str
    title: str
    file_type: str
    file_size: int
    chunk_count: int
    access_roles: list[str]
    created_at: str
    status: str
    error_message: str | None = None


class IngestAllResponse(BaseModel):
    total_files: int
    successful: int
    failed: int
    details: list[dict] = Field(default_factory=list)


class DeleteDocumentResponse(BaseModel):
    document_id: str
    filename: str
    deleted: bool
    message: str
