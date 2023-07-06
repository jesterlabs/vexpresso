from vexpresso.retrievers.base import BaseRetriever, RetrievalOutput
from vexpresso.retrievers.faiss import FaissRetriever
from vexpresso.retrievers.np import Retriever

RETRIEVERS_MAP = {"npy": Retriever, "faiss": FaissRetriever}

DEFAULT_RETRIEVER = "npy"

__all__ = [
    "BaseRetriever",
    "Retriever",
    "RetrievalOutput",
    "FaissRetriever",
    "RETRIEVERS_MAP",
    "DEFAULT_RETRIEVER",
]
