from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from app.core.config import settings
from app.db.repositories import DocumentRepository, get_document_repository
from app.schemas.documents import (
    DeleteDocumentResponse,
    DocumentMetadataResponse,
    DocumentUploadResponse,
    IngestAllResponse,
)
from app.services.ingestion import DocumentIngestionService, get_ingestion_service

router = APIRouter(tags=["Documents"])


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: Annotated[UploadFile, File()],
    ingestion_service: Annotated[DocumentIngestionService, Depends(get_ingestion_service)],
    roles: Annotated[
        str | None,
        Form(description="Comma-separated roles authorized to view this document."),
    ] = "public",
) -> DocumentUploadResponse:
    """
    Validate, ingest, chunk, and embed an uploaded AAU document (.pdf, .docx, .txt, .md).
    """
    role_list = [r.strip() for r in (roles or "public").split(",") if r.strip()]
    record = await ingestion_service.ingest_upload(file=file, access_roles=role_list)
    return DocumentUploadResponse(
        document_id=record.id,
        filename=record.filename,
        chunk_count=record.chunk_count,
        status=record.status,
        message=f"Document '{record.filename}' ingested successfully with {record.chunk_count} chunks.",
    )


@router.get("/documents", response_model=list[DocumentMetadataResponse])
def list_documents(
    repository: Annotated[DocumentRepository, Depends(get_document_repository)],
    roles: Annotated[str | None, Query(description="User roles to filter viewable documents.")] = "public",
) -> list[DocumentMetadataResponse]:
    """List metadata for documents that the current user has access to."""
    role_list = [r.strip() for r in (roles or "public").split(",") if r.strip()]
    docs = repository.list_documents(user_roles=role_list)
    return [
        DocumentMetadataResponse(
            document_id=d.id,
            filename=d.filename,
            title=d.title,
            file_type=d.file_type,
            file_size=d.file_size,
            chunk_count=d.chunk_count,
            access_roles=d.access_roles,
            created_at=d.created_at,
            status=d.status,
            error_message=d.error_message,
        )
        for d in docs
    ]


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(
    document_id: str,
    repository: Annotated[DocumentRepository, Depends(get_document_repository)],
) -> DeleteDocumentResponse:
    """Delete a document and purge all associated indexed chunks from vector store."""
    doc = repository.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    filename = doc.filename
    success = repository.delete_document(document_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document from storage.",
        )

    return DeleteDocumentResponse(
        document_id=document_id,
        filename=filename,
        deleted=True,
        message=f"Document '{filename}' and all vector embeddings were permanently deleted.",
    )


@router.post("/ingest", response_model=IngestAllResponse)
def ingest_directory(
    ingestion_service: Annotated[DocumentIngestionService, Depends(get_ingestion_service)],
) -> IngestAllResponse:
    """
    Ingest all AAU documents located in the data folders (e.g. backend/data or root data).
    """
    targets = [settings.DATA_DIR, settings.ROOT_DATA_DIR]
    total_files = 0
    total_succ = 0
    total_fail = 0
    all_details = []

    seen_filenames = set()

    for data_dir in targets:
        if data_dir.exists() and data_dir.is_dir():
            files = [f for f in data_dir.iterdir() if f.is_file() and f.suffix.lower() in settings.ALLOWED_EXTENSIONS]
            for file_path in files:
                if file_path.name in seen_filenames:
                    continue
                seen_filenames.add(file_path.name)
                total_files += 1
                try:
                    content = file_path.read_bytes()
                    record = ingestion_service.ingest_bytes(
                        file_bytes=content,
                        original_filename=file_path.name,
                        access_roles=["public"],
                    )
                    total_succ += 1
                    all_details.append({
                        "filename": file_path.name,
                        "status": "ingested",
                        "document_id": record.id,
                        "chunks": record.chunk_count,
                    })
                except (OSError, HTTPException) as error:
                    total_fail += 1
                    all_details.append({
                        "filename": file_path.name,
                        "status": "failed",
                        "error": str(error),
                    })

    return IngestAllResponse(
        total_files=total_files,
        successful=total_succ,
        failed=total_fail,
        details=all_details,
    )
