import hashlib
import math

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from app.core.config import settings


class NormalizedHashingEmbedding(EmbeddingFunction):
    """
    High-performance, zero-network-dependency embedding function.
    Generates deterministic normalized subword/word n-gram vector representations
    suitable for offline RAG, local search, unit tests, and production environments
    without risk of HTTP download timeouts.
    """
    def __init__(self, dim: int = 256):
        self.dim = dim

    def name(self) -> str:
        return "normalized_hashing_embedding"

    def get_config(self) -> dict:
        return {"dim": self.dim}

    @classmethod
    def build_from_config(cls, config: dict) -> "NormalizedHashingEmbedding":
        return cls(dim=config.get("dim", 256))

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: list[list[float]] = []
        for text in input:
            tokens = text.lower().replace("\n", " ").split()
            vec = [0.0] * self.dim
            if not tokens:
                embeddings.append(vec)
                continue
            
            for token in tokens:
                # Word token hash
                w_idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
                vec[w_idx] += 1.0
                
                # Subword n-grams (3-grams) for robust morphological matching
                if len(token) >= 3:
                    for i in range(len(token) - 2):
                        ngram = token[i:i+3]
                        ng_idx = int(hashlib.sha256(ngram.encode("utf-8")).hexdigest(), 16) % self.dim
                        vec[ng_idx] += 0.5
            
            norm = math.sqrt(sum(val * val for val in vec))
            if norm > 0:
                vec = [val / norm for val in vec]
            embeddings.append(vec)
        return embeddings


class VectorStoreManager:
    _instance: "VectorStoreManager | None" = None

    def __init__(self):
        settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(settings.CHROMA_PERSIST_DIR))
        self.embedding_function = NormalizedHashingEmbedding(dim=256)
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"description": "Addis Ababa University Knowledge Base"}
        )

    @classmethod
    def get_instance(cls) -> "VectorStoreManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def get_vector_collection():
    """Dependency injection helper to obtain the Chroma collection."""
    return VectorStoreManager.get_instance().collection
