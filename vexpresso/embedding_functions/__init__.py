from vexpresso.embedding_functions.base import EmbeddingFunction, get_embedding_fn
from vexpresso.embedding_functions.clip import ClipEmbeddingsFunction
from vexpresso.embedding_functions.openai import OpenAIEmbeddingFunction
from vexpresso.embedding_functions.sentence_transformers import (
    SentenceTransformerEmbeddingFunction,
)

EMBEDDING_FUNCTIONS_MAP = {
    "clip": ClipEmbeddingsFunction,
    "sentence-transformers": SentenceTransformerEmbeddingFunction,
    "openai": OpenAIEmbeddingFunction,
}

DEFAULT_EMBEDDING_FUNCTION = "sentence-transformers"

__all__ = [
    "EmbeddingFunction",
    "SentenceTransformerEmbeddingFunction",
    "ClipEmbeddingsFunction",
    "OpenAIEmbeddingFunction",
    "get_embedding_fn",
    "EMBEDDING_FUNCTIONS_MAP",
    "DEFAULT_EMBEDDING_FUNCTION",
]
