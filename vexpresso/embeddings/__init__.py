from vexpresso.embeddings.base import EmbeddingFunction
from vexpresso.embeddings.sentence_transformers import (
    SentenceTransformerEmbeddingFunction,
)
from vexpresso.utils import Transformation, transformation


def get_embedding_fn(embedding_fn: Transformation) -> Transformation:
    # langchain check
    if getattr(embedding_fn, "embed_documents", None) is not None:
        return transformation(embedding_fn, function="embed_documents")
    return transformation(embedding_fn)


__all__ = [
    "EmbeddingFunction",
    "SentenceTransformerEmbeddingFunction",
    "get_embeddings_fn",
]
