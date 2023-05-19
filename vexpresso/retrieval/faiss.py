from typing import List

import numpy as np

from vexpresso.retrieval.strategy import RetrievalOutput, RetrievalStrategy


class FaissRetrievalStrategy(RetrievalStrategy):
    def __init__(self):
        self._faiss = None
        try:
            import faiss  # noqa

            self._faiss = faiss
        except ImportError:
            raise ImportError(
                "Could not import faiss python package."
                "Please install it with `pip install faiss-cpu` or `faiss-gpu`."
            )
        self.index = None

    def _setup_index(self, embeddings: np.ndarray):
        self.index = self._faiss.IndexFlatL2(embeddings.shape[1])  # noqa
        self.index.add(embeddings.astype(np.float32))

    def retrieve(
        self,
        query_embeddings: np.ndarray,
        embeddings: np.ndarray,
        k: int = 4,
    ) -> List[RetrievalOutput]:
        self._setup_index(embeddings)
        _, indices = self.index.search(query_embeddings.astype(np.float32), k=k)
        out = []
        for indices in indices:
            query_output = RetrievalOutput(
                embeddings[indices], indices, query_embeddings
            )
            out.append(query_output)
        return out
