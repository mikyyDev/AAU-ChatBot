import json
import logging
import threading
from pathlib import Path

from app.core.config import settings
from app.core.security import check_document_access
from app.db.models import DocumentRecord
from app.db.session import get_vector_collection

logger = logging.getLogger(__name__)


class DocumentRepository:
    """
    Repository for managing document metadata and synchronizing with the vector store.
    Uses thread-safe persistent JSON store for document metadata.
    """
    _instance: "DocumentRepository | None" = None
    _lock = threading.Lock()

    def __init__(self, store_path: Path | None = None):
        self.store_path = store_path or settings.METADATA_STORE_PATH
        self._documents: dict[str, DocumentRecord] = {}
        self._load()

    @classmethod
    def get_instance(cls) -> "DocumentRepository":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load(self) -> None:
        """Load document records from persistent disk store."""
        if self.store_path.exists():
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for doc_id, doc_data in data.items():
                        self._documents[doc_id] = DocumentRecord(**doc_data)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                logger.exception("Error loading document metadata from %s", self.store_path)
                self._documents = {}

    def _save(self) -> None:
        """Atomically persist document records to disk."""
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.store_path.with_suffix(".tmp")
            data = {doc_id: doc.model_dump() for doc_id, doc in self._documents.items()}
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            temp_path.replace(self.store_path)
        except (OSError, TypeError, ValueError):
            logger.exception("Failed to persist document metadata to %s", self.store_path)

    def save_document(self, record: DocumentRecord) -> DocumentRecord:
        with self._lock:
            self._documents[record.id] = record
            self._save()
            return record

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        return self._documents.get(doc_id)

    def get_by_hash(self, file_hash: str) -> DocumentRecord | None:
        for doc in self._documents.values():
            if doc.file_hash == file_hash and doc.status == "ingested":
                return doc
        return None

    def get_by_filename(self, filename: str) -> DocumentRecord | None:
        for doc in self._documents.values():
            if doc.filename.lower() == filename.lower() and doc.status == "ingested":
                return doc
        return None

    def list_documents(self, user_roles: list[str] | None = None) -> list[DocumentRecord]:
        """List all documents that the user has permission to view."""
        roles = user_roles or ["public"]
        results = []
        for doc in self._documents.values():
            if check_document_access(doc.access_roles, roles):
                results.append(doc)
        return results

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete document metadata and remove all associated chunks from vector store.
        Ensures no stale indexed content is left behind.
        """
        with self._lock:
            if doc_id not in self._documents:
                return False
            
            # Remove from vector store
            try:
                collection = get_vector_collection()
                collection.delete(where={"document_id": doc_id})
                logger.info(f"Deleted vector chunks for document_id: {doc_id}")
            except Exception:
                logger.exception("Error removing vectors for doc_id %s", doc_id)

            del self._documents[doc_id]
            self._save()
            return True

    def clear(self) -> None:
        """Reset repository (used for testing)."""
        with self._lock:
            self._documents.clear()
            self._save()
            try:
                collection = get_vector_collection()
                # Clear all vectors from Chroma collection
                all_ids = collection.get()["ids"]
                if all_ids:
                    collection.delete(ids=all_ids)
            except Exception:
                logger.exception("Failed to clear vector store during repository reset")


def get_document_repository() -> DocumentRepository:
    return DocumentRepository.get_instance()
