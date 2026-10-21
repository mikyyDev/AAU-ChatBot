import io

from app.db.repositories import DocumentRepository
from app.db.session import get_vector_collection
from app.services.ingestion import DocumentIngestionService
from fastapi.testclient import TestClient


def test_successful_txt_ingestion(clean_db_state: DocumentRepository):
    service = DocumentIngestionService(repository=clean_db_state)
    content = b"Addis Ababa University offers programs across humanities, social sciences, and engineering."
    record = service.ingest_bytes(
        file_bytes=content,
        original_filename="aau_programs.txt",
        access_roles=["public"],
    )

    assert record.id is not None
    assert record.filename == "aau_programs.txt"
    assert record.status == "ingested"
    assert record.chunk_count > 0

    # Verify vector store has the chunks
    col = get_vector_collection()
    res = col.get(where={"document_id": record.id})
    assert len(res["ids"]) == record.chunk_count


def test_duplicate_ingestion_prevention(clean_db_state: DocumentRepository):
    service = DocumentIngestionService(repository=clean_db_state)
    content = b"Addis Ababa University was founded in 1950."
    rec1 = service.ingest_bytes(content, "history.txt")
    rec2 = service.ingest_bytes(content, "history.txt")

    assert rec1.id == rec2.id
    # Verify no duplicate vectors added
    col = get_vector_collection()
    res = col.get(where={"document_id": rec1.id})
    assert len(res["ids"]) == rec1.chunk_count


def test_document_deletion_removes_vectors(clean_db_state: DocumentRepository):
    service = DocumentIngestionService(repository=clean_db_state)
    content = b"Research guidelines for AAU postgraduate students."
    record = service.ingest_bytes(content, "guidelines.txt")
    
    col = get_vector_collection()
    res_before = col.get(where={"document_id": record.id})
    assert len(res_before["ids"]) > 0

    # Delete
    deleted = clean_db_state.delete_document(record.id)
    assert deleted is True

    # Vectors must be purged
    res_after = col.get(where={"document_id": record.id})
    assert len(res_after["ids"]) == 0
    assert clean_db_state.get_document(record.id) is None


# Scenario 10: An empty, corrupted, or unsupported upload
def test_empty_file_upload_rejected(client: TestClient):
    response = client.post(
        "/api/documents/upload",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_unsupported_file_extension_rejected(client: TestClient):
    response = client.post(
        "/api/documents/upload",
        files={"file": ("malicious.exe", io.BytesIO(b"binary content"), "application/x-msdownload")},
    )
    assert response.status_code == 400
    assert "unsupported file type" in response.json()["detail"].lower()


def test_corrupted_pdf_upload_handled_gracefully(client: TestClient):
    corrupted_pdf_bytes = b"%PDF-1.4 CORRUPTED CONTENT NOT A VALID PDF STRUCTURE %%%EOF"
    response = client.post(
        "/api/documents/upload",
        files={"file": ("corrupted.pdf", io.BytesIO(corrupted_pdf_bytes), "application/pdf")},
    )
    assert response.status_code == 400
    assert "corrupted" in response.json()["detail"].lower() or "failed" in response.json()["detail"].lower()


def test_path_traversal_filename_sanitized(client: TestClient):
    content = b"Safe content for university policy."
    response = client.post(
        "/api/documents/upload",
        files={"file": ("../../../../etc/passwd.txt", io.BytesIO(content), "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["filename"] == "passwd.txt"
