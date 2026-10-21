import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import settings
from app.db.repositories import DocumentRepository, get_document_repository
from app.db.session import get_vector_collection
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.answer_generator import AnswerGenerator, get_answer_generator
from app.services.evidence_checker import EvidenceChecker, get_evidence_checker
from app.services.retriever import DocumentRetriever, get_retriever

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
@router.post("", response_model=ChatResponse)  # Alias for /api or /chat
def chat_endpoint(
    request: ChatRequest,
    retriever: Annotated[DocumentRetriever, Depends(get_retriever)],
    evidence_checker: Annotated[EvidenceChecker, Depends(get_evidence_checker)],
    answer_generator: Annotated[AnswerGenerator, Depends(get_answer_generator)],
) -> ChatResponse:
    """
    Core Question-Answering endpoint.
    Retrieves evidence, verifies sufficiency, and returns strictly grounded answers.
    """
    query = request.query
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    try:
        # 1. Retrieve chunks applying user access roles
        retrieved_chunks = retriever.retrieve(query=query, user_roles=request.roles)

        # 2. Evaluate evidence sufficiency and relevance
        evidence = evidence_checker.evaluate_evidence(query=query, retrieved_chunks=retrieved_chunks)

        # 3. Generate grounded answer or refusal
        answer_text, citations, is_grounded, is_refusal = answer_generator.generate_answer(
            query=query, evidence=evidence
        )

        return ChatResponse(
            answer=answer_text,
            sources=citations,
            grounded=is_grounded,
            refusal=is_refusal,
            found=not is_refusal and is_grounded,
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected chat processing error")
        # Safe fallback without exposing internal stack trace
        return ChatResponse(
            answer=settings.REFUSAL_MESSAGE,
            sources=[],
            grounded=False,
            refusal=True,
            found=False,
        )


@router.get("/health")
def health_check(
    repository: Annotated[DocumentRepository, Depends(get_document_repository)],
):
    """Health check endpoint reporting API and vector store status."""
    try:
        collection = get_vector_collection()
        total_vectors = collection.count()
        total_docs = len(repository.list_documents(user_roles=["admin"]))
        return {
            "status": "healthy",
            "app_name": settings.APP_NAME,
            "environment": settings.ENVIRONMENT,
            "indexed_documents": total_docs,
            "indexed_chunks": total_vectors,
            "groq_configured": bool(settings.GROQ_API_KEY),
        }
    except Exception as error:
        logger.exception("Health check failed")
        return {
            "status": "degraded",
            "error": str(error),
        }
