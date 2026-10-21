import hashlib
import io
import logging
import uuid
import zipfile
from pathlib import Path

from chromadb.errors import ChromaError
from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from fastapi import HTTPException, UploadFile, status
from pydantic import ValidationError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.config import settings
from app.core.security import (
    sanitize_filename,
    validate_file_extension,
    validate_file_size,
)
from app.db.models import DocumentChunk, DocumentRecord
from app.db.repositories import DocumentRepository, get_document_repository
from app.db.session import get_vector_collection

logger = logging.getLogger(__name__)


def compute_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def split_text_into_chunks(
    text: str,
    chunk_size: int = 600,
    overlap: int = 100,
) -> list[str]:
    """
    Split text into chunks with overlap, breaking on paragraph or sentence boundaries.
    """
    if not text:
        return []
    
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]
    
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for para in paragraphs:
        para_len = len(para)
        # If single paragraph exceeds chunk size, split by lines or sentences
        if para_len > chunk_size:
            words = para.split()
            buf = []
            buf_len = 0
            for w in words:
                if buf_len + len(w) + 1 > chunk_size:
                    chunk_str = " ".join(buf)
                    if chunk_str:
                        chunks.append(chunk_str)
                    # keep overlap from end of words
                    overlap_words = buf[-int(overlap / 10):] if len(buf) > 10 else []
                    buf = list(overlap_words)
                    buf.append(w)
                    buf_len = sum(len(x) + 1 for x in buf)
                else:
                    buf.append(w)
                    buf_len += len(w) + 1
            if buf:
                chunks.append(" ".join(buf))
            continue

        if current_length + para_len + 2 <= chunk_size:
            current_chunk.append(para)
            current_length += para_len + 2
        else:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_length = para_len

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return [c.strip() for c in chunks if c.strip()]


class DocumentIngestionService:
    def __init__(self, repository: DocumentRepository | None = None):
        self.repository = repository or get_document_repository()

    def extract_text_from_bytes(
        self, file_bytes: bytes, file_ext: str
    ) -> list[tuple[str, int | None]]:
        """
        Extract text chunks with page numbers: returns list of (page_or_section_text, page_number).
        """
        results: list[tuple[str, int | None]] = []

        if file_ext == ".pdf":
            try:
                reader = PdfReader(io.BytesIO(file_bytes))
                if len(reader.pages) == 0:
                    raise ValueError("PDF has 0 pages.")
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        results.append((page_text.strip(), i + 1))
            except (OSError, PdfReadError, ValueError) as error:
                raise ValueError(f"Corrupted or unreadable PDF: {error}") from error

        elif file_ext == ".docx":
            try:
                doc = Document(io.BytesIO(file_bytes))
                paras = [p.text for p in doc.paragraphs if p.text.strip()]
                # Include text from tables if any
                for table in doc.tables:
                    for row in table.rows:
                        row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                        if row_text:
                            paras.append(row_text)
                
                full_doc_text = "\n\n".join(paras)
                if not full_doc_text.strip():
                    raise ValueError("DOCX document contains no text.")
                results.append((full_doc_text.strip(), None))
            except (OSError, KeyError, ValueError, zipfile.BadZipFile, PackageNotFoundError) as error:
                raise ValueError(f"Corrupted or unreadable DOCX: {error}") from error

        elif file_ext in {".txt", ".md"}:
            try:
                text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = file_bytes.decode("latin-1")
            
            if not text.strip():
                raise ValueError("File is empty or contains only whitespace.")
            results.append((text.strip(), None))

        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        if not results or all(not text.strip() for text, _ in results):
            raise ValueError("No readable text could be extracted from the document.")

        return results

    def ingest_bytes(
        self,
        file_bytes: bytes,
        original_filename: str,
        access_roles: list[str] | None = None,
        force_reindex: bool = False,
    ) -> DocumentRecord:
        """
        Process, chunk, embed, and store document in vector store.
        """
        clean_name = sanitize_filename(original_filename)
        file_ext = validate_file_extension(clean_name)
        file_size = len(file_bytes)
        validate_file_size(file_size)

        file_hash = compute_sha256(file_bytes)
        roles = access_roles or ["public"]

        # Duplicate detection
        existing_doc = self.repository.get_by_hash(file_hash)
        if existing_doc and not force_reindex:
            logger.info(f"File {clean_name} already ingested with doc_id {existing_doc.id}")
            return existing_doc

        # If a document with the same filename already exists, remove stale chunks first
        existing_by_name = self.repository.get_by_filename(clean_name)
        if existing_by_name:
            logger.info(f"Updating existing document: {clean_name}")
            self.repository.delete_document(existing_by_name.id)

        doc_id = str(uuid.uuid4())
        title = Path(clean_name).stem.replace("_", " ").title()

        try:
            pages_text = self.extract_text_from_bytes(file_bytes, file_ext)
            
            # Chunking
            chunks: list[DocumentChunk] = []
            chunk_idx = 0
            for page_text, page_num in pages_text:
                text_splits = split_text_into_chunks(
                    page_text,
                    chunk_size=settings.CHUNK_SIZE,
                    overlap=settings.CHUNK_OVERLAP,
                )
                for split in text_splits:
                    chunk = DocumentChunk(
                        chunk_id=f"{doc_id}_c{chunk_idx}",
                        document_id=doc_id,
                        document_name=clean_name,
                        page=page_num,
                        chunk_index=chunk_idx,
                        content=split,
                        access_roles=roles,
                    )
                    chunks.append(chunk)
                    chunk_idx += 1

            if not chunks:
                raise ValueError("No extractable chunks found in document.")

            # Index into Chroma vector store
            collection = get_vector_collection()
            
            ids = [c.chunk_id for c in chunks]
            documents = [c.content for c in chunks]
            metadatas = [
                {
                    "document_id": c.document_id,
                    "document_name": c.document_name,
                    "page": c.page if c.page is not None else -1,
                    "chunk_index": c.chunk_index,
                    "access_roles": ",".join(c.access_roles),
                    "title": title,
                }
                for c in chunks
            ]

            collection.add(ids=ids, documents=documents, metadatas=metadatas)

            record = DocumentRecord(
                id=doc_id,
                filename=clean_name,
                file_hash=file_hash,
                title=title,
                file_type=file_ext,
                file_size=file_size,
                chunk_count=len(chunks),
                access_roles=roles,
                status="ingested",
            )
            self.repository.save_document(record)
            logger.info(f"Successfully ingested {clean_name}: {len(chunks)} chunks created.")
            return record

        except (ChromaError, HTTPException, OSError, RuntimeError, ValidationError, ValueError) as error:
            logger.exception("Failed to ingest %s", clean_name)
            failed_record = DocumentRecord(
                id=doc_id,
                filename=clean_name,
                file_hash=file_hash,
                title=title,
                file_type=file_ext,
                file_size=file_size,
                chunk_count=0,
                access_roles=roles,
                status="failed",
                error_message=str(error),
            )
            self.repository.save_document(failed_record)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document ingestion failed: {error}",
            ) from error

    async def ingest_upload(
        self,
        file: UploadFile,
        access_roles: list[str] | None = None,
    ) -> DocumentRecord:
        content = await file.read()
        return self.ingest_bytes(
            file_bytes=content,
            original_filename=file.filename or "uploaded_file",
            access_roles=access_roles,
        )

    def ingest_directory(
        self,
        directory_path: Path,
        access_roles: list[str] | None = None,
    ) -> tuple[int, int, list[dict]]:
        """
        Scan and ingest all supported documents in a directory.
        """
        if not directory_path.exists() or not directory_path.is_dir():
            return 0, 0, []

        successful = 0
        failed = 0
        details: list[dict] = []

        files = sorted(
            [f for f in directory_path.iterdir() if f.is_file() and f.suffix.lower() in settings.ALLOWED_EXTENSIONS]
        )

        for file_path in files:
            try:
                content = file_path.read_bytes()
                record = self.ingest_bytes(
                    file_bytes=content,
                    original_filename=file_path.name,
                    access_roles=access_roles or ["public"],
                )
                successful += 1
                details.append({
                    "filename": file_path.name,
                    "status": "ingested",
                    "document_id": record.id,
                    "chunks": record.chunk_count,
                })
            except (ChromaError, HTTPException, OSError, RuntimeError, ValidationError, ValueError) as error:
                failed += 1
                details.append({
                    "filename": file_path.name,
                    "status": "failed",
                    "error": str(error),
                })

        return len(files), successful, details


def get_ingestion_service() -> DocumentIngestionService:
    return DocumentIngestionService()
