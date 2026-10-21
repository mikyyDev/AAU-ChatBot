import pytest
from app.db.repositories import DocumentRepository
from app.prompts.system_prompt import PRESCRIBED_REFUSAL_MESSAGE
from app.services.answer_generator import AnswerGenerator
from app.services.citation_validator import CitationValidator
from app.services.evidence_checker import EvidenceChecker
from app.services.ingestion import DocumentIngestionService
from app.services.retriever import DocumentRetriever, RetrievedChunk


# Scenario 2: A question with no relevant retrieved chunks
def test_question_with_no_relevant_retrieved_chunks(clean_db_state: DocumentRepository):
    # Vector store is empty or has unrelated docs
    service = DocumentIngestionService(repository=clean_db_state)
    service.ingest_bytes(
        file_bytes=b"Addis Ababa University offers courses in engineering, mathematics, and physics.",
        original_filename="courses.txt",
    )

    retriever = DocumentRetriever()
    evidence_checker = EvidenceChecker()
    answer_gen = AnswerGenerator()

    query = "What is the flight schedule for Ethiopian Airlines to Paris?"
    chunks = retriever.retrieve(query=query)
    evidence = evidence_checker.evaluate_evidence(query=query, retrieved_chunks=chunks)
    
    assert evidence.is_sufficient is False
    answer, sources, grounded, refusal = answer_gen.generate_answer(query=query, evidence=evidence)

    assert refusal is True
    assert grounded is False
    assert len(sources) == 0
    assert answer == PRESCRIBED_REFUSAL_MESSAGE


# Scenario 3: A question where retrieved chunks discuss the topic but do not answer it
def test_topic_overlap_without_answer_predicate_refusal(clean_db_state: DocumentRepository):
    service = DocumentIngestionService(repository=clean_db_state)
    # The document talks about AAU library branches, but contains NO information about overdue fines
    library_doc = (
        "The Addis Ababa University Library System includes the John F. Kennedy Memorial Library "
        "at Sidist Kilo campus, along with specialized branch libraries at the Technology and Health campuses. "
        "The libraries provide study spaces and computer stations for registered university students."
    )
    service.ingest_bytes(file_bytes=library_doc.encode("utf-8"), original_filename="library_info.txt")

    retriever = DocumentRetriever()
    evidence_checker = EvidenceChecker()
    answer_gen = AnswerGenerator()

    # User asks for fine fee (numeric predicate), which is NOT in the text
    query = "What is the fee or penalty cost for overdue library books at AAU?"
    chunks = retriever.retrieve(query=query)
    evidence = evidence_checker.evaluate_evidence(query=query, retrieved_chunks=chunks)

    # Must be marked insufficient!
    assert evidence.is_sufficient is False
    answer, _, grounded, refusal = answer_gen.generate_answer(query=query, evidence=evidence)

    assert refusal is True
    assert grounded is False
    assert answer == PRESCRIBED_REFUSAL_MESSAGE


# Scenario 5: An unrelated general-knowledge question
@pytest.mark.parametrize("general_query", [
    "Who won the 2022 World Cup in Qatar?",
    "Explain Python decorators and metaclasses.",
    "What is the capital of Australia?",
    "How does quantum computing work?",
])
def test_unrelated_general_knowledge_questions_refused(
    clean_db_state: DocumentRepository, sample_aau_text: str, general_query: str
):
    service = DocumentIngestionService(repository=clean_db_state)
    service.ingest_bytes(sample_aau_text.encode("utf-8"), "aau_guide.txt")

    retriever = DocumentRetriever()
    evidence_checker = EvidenceChecker()
    answer_gen = AnswerGenerator()

    chunks = retriever.retrieve(query=general_query)
    evidence = evidence_checker.evaluate_evidence(query=general_query, retrieved_chunks=chunks)
    answer, _, grounded, refusal = answer_gen.generate_answer(query=general_query, evidence=evidence)

    assert refusal is True
    assert grounded is False
    assert answer == PRESCRIBED_REFUSAL_MESSAGE


# Scenario 12: A response containing an unsupported factual claim
def test_unsupported_factual_claim_rejected():
    validator = CitationValidator()
    
    retrieved_chunk = RetrievedChunk(
        chunk_id="c1",
        document_id="doc-123",
        document_name="aau_tuition.txt",
        page=1,
        score=0.85,
        content="Undergraduate tuition for Ethiopian regular students is fully subsidized by the government.",
        access_roles=["public"],
    )

    # Hypothesize an answer with unsupported factual claims and no matching vocabulary
    hallucinated_answer = (
        "The president of France visited Addis Ababa University in 1998 and donated a luxury yacht to the lake campus."
    )

    citations, is_grounded = validator.validate_citations(
        answer_text=hallucinated_answer,
        retrieved_chunks=[retrieved_chunk],
    )

    assert is_grounded is False
    assert len(citations) == 0
