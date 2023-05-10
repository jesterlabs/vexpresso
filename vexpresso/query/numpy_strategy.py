from typing import Any, Iterable

import numpy as np

from vexpresso.query.strategy import BatchedQueryOutput, QueryOutput, QueryStrategy


def is_batched(arr: np.ndarray) -> bool:
    if len(arr.shape) > 1:
        return True
    return False


def get_norm_vector(vector: np.ndarray) -> np.array:
    if len(vector.shape) == 1:
        return vector / np.linalg.norm(vector)
    return vector / np.linalg.norm(vector, axis=-1)[:, np.newaxis]


def cosine_similarity(query_vector, vectors):
    norm_vectors = get_norm_vector(vectors)
    norm_query_vector = get_norm_vector(query_vector)
    similarities = np.dot(norm_vectors, norm_query_vector.T)
    return similarities


def euclidean_metric(query_vector, vectors, get_similarity_score=True):
    if not is_batched(query_vector):
        similarities = np.linalg.norm(vectors - query_vector, axis=1)
    else:
        similarities = np.linalg.norm(vectors - query_vector[:, np.newaxis], axis=1)
    if get_similarity_score:
        similarities = 1 / (1 + similarities)
    return similarities


def get_similarity_fn(name: str):
    functions = {
        "euclidian": euclidean_metric,
        "cosine": cosine_similarity,
    }  # prolly move this to enums
    return functions.get(name, "euclidian")


class NumpyStrategy(QueryStrategy):
    def __init__(self, similarity_fn: str = "euclidian"):
        self.similarity_fn = get_similarity_fn(similarity_fn)

    def _get_top_k(
        self,
        query_embeddings: np.ndarray,
        embeddings: np.ndarray,
        k: int = 1,
    ):
        similarities = self.distance_fn(query_embeddings, embeddings)
        if not is_batched(similarities):
            similarities = np.expand_dims(similarities, axis=0)
        top_indices = np.argsort(similarities, axis=-1)[:, -k:][::-1]  # B X k
        return top_indices

    def query(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        k: int = 1,
    ) -> QueryOutput:
        top_indices = np.squeeze(self._get_top_k(query_embedding, embeddings, k))
        return QueryOutput(embeddings[top_indices], top_indices, query_embedding)

    def batch_query(
        self, query_embeddings: np.ndarray, embeddings: np.ndarray, k: int = 1
    ) -> BatchedQueryOutput:
        top_indices = self._get_top_k(query_embeddings, embeddings, k)

        out_embeddings = []

        for indices in top_indices:
            batch_embeddings = embeddings[indices]
            out_embeddings.append(batch_embeddings)

        out_embeddings = np.squeeze(np.stack(batch_embeddings))
        out_indices = np.squeeze(top_indices)
        return BatchedQueryOutput(out_embeddings, out_indices, query_embeddings)
