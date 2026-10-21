import logging

from chromadb.errors import ChromaError
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.security import check_document_access
from app.db.session import get_vector_collection

logger = logging.getLogger(__name__)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page: int | None = None
    score: float = Field(0.0, description="Normalized relevance score between 0.0 and 1.0")
    distance: float = Field(0.0, description="Raw distance metric from vector search")
    content: str
    access_roles: list[str] = Field(default_factory=lambda: ["public"])


class DocumentRetriever:
    def __init__(self):
        self.collection = get_vector_collection()

    def retrieve(
        self,
        query: str,
        user_roles: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Query Chroma vector store, apply role-based access filtering,
        calculate calibrated similarity scores, and return ranked chunks.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        roles = user_roles or ["public"]
        k = top_k if top_k is not None else settings.RETRIEVAL_TOP_K
        if k <= 0:
            return []

        try:
            # Query vector store
            count = self.collection.count()
            if count == 0:
                logger.info("Vector collection is currently empty.")
                return []

            n_results = min(k * 3, count)  # Over-fetch slightly to allow post-filtering by permissions
            results = self.collection.query(
                query_texts=[cleaned_query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            if not results or not results["documents"] or not results["documents"][0]:
                return []

            docs = results["documents"][0]
            metas = results["metadatas"][0]
            ids = results["ids"][0]
            distance_results = results.get("distances")
            distances = distance_results[0] if distance_results else [0.0] * len(docs)

            retrieved: list[RetrievedChunk] = []

            for chunk_id, content, meta, dist in zip(ids, docs, metas, distances):
                doc_roles = meta.get("access_roles", "public").split(",")
                
                # Enforce RBAC at retrieval time
                if not check_document_access(doc_roles, roles):
                    continue

                # Calibrate distance to a 0.0 - 1.0 similarity score
                # For cosine or L2 normalized embeddings, dist typically in [0.0, 2.0]
                similarity = max(0.0, 1.0 - (dist / 2.0))

                page_val = meta.get("page", -1)
                page_num = page_val if page_val > 0 else None

                chunk = RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=meta.get("document_id", "unknown"),
                    document_name=meta.get("document_name", "unknown_doc"),
                    page=page_num,
                    score=round(similarity, 4),
                    distance=round(dist, 4),
                    content=content,
                    access_roles=doc_roles,
                )
                retrieved.append(chunk)

            # Return top-k authorized results
            return retrieved[:k]

        except (ChromaError, OSError, RuntimeError, ValueError) as error:
            logger.exception("Vector retrieval failed")
            raise RuntimeError(f"Vector search service failure: {error}") from error


def get_retriever() -> DocumentRetriever:
    return DocumentRetriever()
