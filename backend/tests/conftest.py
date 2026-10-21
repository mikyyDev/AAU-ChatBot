from pathlib import Path

import pytest
from app.db.repositories import DocumentRepository
from app.db.session import get_vector_collection
from app.main import app
from app.services.ingestion import DocumentIngestionService
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def clean_db_state(tmp_path: Path):
    """
    Ensure every test starts with an isolated, clean repository and collection.
    """
    test_meta_path = tmp_path / "test_metadata.json"
    repo = DocumentRepository(store_path=test_meta_path)
    DocumentRepository._instance = repo

    # Clean collection before each test. Fail setup if cleanup fails rather than
    # silently running against contaminated vector-store state.
    col = get_vector_collection()
    all_ids = col.get()["ids"]
    if all_ids:
        col.delete(ids=all_ids)

    try:
        yield repo
    finally:
        # Cleanup after each test and always reset the repository singleton.
        try:
            col = get_vector_collection()
            all_ids = col.get()["ids"]
            if all_ids:
                col.delete(ids=all_ids)
        finally:
            DocumentRepository._instance = None


@pytest.fixture
def client():
    """FastAPI TestClient instance."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_aau_text():
    return (
        "Addis Ababa University (AAU) was established in 1950 as the University College of Addis Ababa. "
        "The main campus is located at Sidist Kilo in Addis Ababa, Ethiopia. "
        "Undergraduate admission requires completion of the Ethiopian University Entrance Examination (EUEE) "
        "with a minimum competitive passing score set annually by the Ministry of Education. "
        "The regular academic year consists of two semesters: Semester I starts in October, and Semester II starts in March. "
        "The deadline for first semester student registration is October 25."
    )


@pytest.fixture
def ingested_sample_doc(clean_db_state, sample_aau_text):
    service = DocumentIngestionService(repository=clean_db_state)
    record = service.ingest_bytes(
        file_bytes=sample_aau_text.encode("utf-8"),
        original_filename="aau_academic_calendar.txt",
        access_roles=["public"],
    )
    return record
