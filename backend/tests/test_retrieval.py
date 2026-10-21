from app.db.repositories import DocumentRepository
from app.services.ingestion import DocumentIngestionService
from app.services.retriever import DocumentRetriever


def test_retrieval_finds_relevant_chunks(clean_db_state: DocumentRepository, sample_aau_text: str):
    service = DocumentIngestionService(repository=clean_db_state)
    service.ingest_bytes(
        file_bytes=sample_aau_text.encode("utf-8"),
        original_filename="aau_info.txt",
        access_roles=["public"],
    )

    retriever = DocumentRetriever()
    results = retriever.retrieve(query="When was Addis Ababa University established?", user_roles=["public"])

    assert len(results) > 0
    top = results[0]
    assert top.document_name == "aau_info.txt"
    assert "1950" in top.content
    assert top.score > 0.3


# Scenario 9: Unauthorized access to restricted documents
def test_unauthorized_access_to_restricted_documents(clean_db_state: DocumentRepository):
    service = DocumentIngestionService(repository=clean_db_state)
    
    # Ingest restricted document for faculty only
    faculty_doc = b"Faculty confidential: Annual salary review and faculty promotion board schedules for 2026."
    service.ingest_bytes(
        file_bytes=faculty_doc,
        original_filename="confidential_faculty_salaries.txt",
        access_roles=["faculty"],
    )

    # Ingest public student document
    public_doc = b"Student library hours: The AAU central library is open Monday to Saturday from 8:00 AM to 8:00 PM."
    service.ingest_bytes(
        file_bytes=public_doc,
        original_filename="library_hours.txt",
        access_roles=["public"],
    )

    retriever = DocumentRetriever()

    # Public user queries for faculty salaries
    public_results = retriever.retrieve(
        query="What are the faculty salary reviews?",
        user_roles=["public"],
    )
    # The confidential faculty document MUST NOT be retrieved for public role!
    retrieved_names = [c.document_name for c in public_results]
    assert "confidential_faculty_salaries.txt" not in retrieved_names

    # Faculty member queries for the same document
    faculty_results = retriever.retrieve(
        query="What are the faculty salary reviews?",
        user_roles=["faculty"],
    )
    assert len(faculty_results) > 0
    assert faculty_results[0].document_name == "confidential_faculty_salaries.txt"


def test_admin_has_access_to_all_documents(clean_db_state: DocumentRepository):
    service = DocumentIngestionService(repository=clean_db_state)
    service.ingest_bytes(
        file_bytes=b"Restricted dean minutes: internal restructuring plan.",
        original_filename="dean_minutes.txt",
        access_roles=["dean"],
    )

    retriever = DocumentRetriever()
    admin_results = retriever.retrieve(
        query="internal restructuring plan",
        user_roles=["admin"],
    )
    assert len(admin_results) > 0
    assert admin_results[0].document_name == "dean_minutes.txt"
