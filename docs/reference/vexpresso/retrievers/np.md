# Module vexpresso.retrievers.np

??? example "View Source"
        from typing import Any, List

        import numpy as np

        from vexpresso.retrievers.base import BaseRetriever, RetrievalOutput

        

        def is_batched(arr: np.ndarray) -> bool:

            if len(arr.shape) > 1:

                return True

            return False

        

        def get_norm_vector(vector: np.ndarray) -> np.array:

            if len(vector.shape) == 1:

                return vector / np.linalg.norm(vector)

            return vector / np.linalg.norm(vector, axis=-1)[:, np.newaxis] + 1e-4

        

        def cosine_similarity(query_vector, vectors):

            norm_vectors = get_norm_vector(vectors)

            norm_query_vector = get_norm_vector(query_vector)

            similarities = np.dot(norm_query_vector, norm_vectors.T)

            return similarities

        

        def euclidean_metric(query_vector, vectors, get_similarity_score=True):

            similarities = np.linalg.norm(

                vectors[np.newaxis, :, :] - query_vector[:, np.newaxis, :], axis=-1

            )

            if get_similarity_score:

                similarities = 1 / (1 + similarities)

            return similarities

        

        def get_similarity_fn(name: str):

            functions = {

                "euclidian": euclidean_metric,

                "cosine": cosine_similarity,

            }  # prolly move this to enums

            return functions.get(name, functions["cosine"])

        

        class Retriever(BaseRetriever):

            def __init__(self, similarity_fn: str = "cosine"):

                self.similarity_fn = get_similarity_fn(similarity_fn)

            def _get_similarities(

                self,

                query_embeddings: np.ndarray,

                embeddings: np.ndarray,

            ):

                if not is_batched(query_embeddings):

                    query_embeddings = np.expand_dims(query_embeddings, axis=0)

                similarities = self.similarity_fn(query_embeddings, embeddings)

                if not is_batched(similarities):

                    similarities = np.expand_dims(similarities, 0)

                return similarities

            def _get_top_k(

                self,

                query_embeddings: np.ndarray,

                embeddings: np.ndarray,

                k: int = 1,

            ):

                similarities = self._get_similarities(query_embeddings, embeddings)

                top_indices = np.flip(

                    np.argsort(similarities, axis=-1)[:, -k:], axis=-1

                )  # B X k

                return top_indices, similarities

            def retrieve(

                self,

                query_embeddings: np.ndarray,

                embeddings: List[Any],

                k: int = 4,

            ) -> List[RetrievalOutput]:

                embeddings = np.array(embeddings)

                query_embeddings = np.array(query_embeddings)

                top_indices, similarities = self._get_top_k(query_embeddings, embeddings, k)

                # move to list for consistency w/ single and batch calls

                out = []

                for idx in range(top_indices.shape[0]):

                    query_output = RetrievalOutput(

                        embeddings[top_indices[idx]],

                        top_indices[idx],

                        scores=similarities[idx],

                        query_embeddings=query_embeddings[idx],

                    )

                    out.append(query_output)

                return out

## Functions

    
### cosine_similarity

```python3
def cosine_similarity(
    query_vector,
    vectors
)
```

??? example "View Source"
        def cosine_similarity(query_vector, vectors):

            norm_vectors = get_norm_vector(vectors)

            norm_query_vector = get_norm_vector(query_vector)

            similarities = np.dot(norm_query_vector, norm_vectors.T)

            return similarities

    
### euclidean_metric

```python3
def euclidean_metric(
    query_vector,
    vectors,
    get_similarity_score=True
)
```

??? example "View Source"
        def euclidean_metric(query_vector, vectors, get_similarity_score=True):

            similarities = np.linalg.norm(

                vectors[np.newaxis, :, :] - query_vector[:, np.newaxis, :], axis=-1

            )

            if get_similarity_score:

                similarities = 1 / (1 + similarities)

            return similarities

    
### get_norm_vector

```python3
def get_norm_vector(
    vector: numpy.ndarray
) -> <built-in function array>
```

??? example "View Source"
        def get_norm_vector(vector: np.ndarray) -> np.array:

            if len(vector.shape) == 1:

                return vector / np.linalg.norm(vector)

            return vector / np.linalg.norm(vector, axis=-1)[:, np.newaxis] + 1e-4

    
### get_similarity_fn

```python3
def get_similarity_fn(
    name: str
)
```

??? example "View Source"
        def get_similarity_fn(name: str):

            functions = {

                "euclidian": euclidean_metric,

                "cosine": cosine_similarity,

            }  # prolly move this to enums

            return functions.get(name, functions["cosine"])

    
### is_batched

```python3
def is_batched(
    arr: numpy.ndarray
) -> bool
```

??? example "View Source"
        def is_batched(arr: np.ndarray) -> bool:

            if len(arr.shape) > 1:

                return True

            return False

## Classes

### Retriever

```python3
class Retriever(
    similarity_fn: str = 'cosine'
)
```

??? example "View Source"
        class Retriever(BaseRetriever):

            def __init__(self, similarity_fn: str = "cosine"):

                self.similarity_fn = get_similarity_fn(similarity_fn)

            def _get_similarities(

                self,

                query_embeddings: np.ndarray,

                embeddings: np.ndarray,

            ):

                if not is_batched(query_embeddings):

                    query_embeddings = np.expand_dims(query_embeddings, axis=0)

                similarities = self.similarity_fn(query_embeddings, embeddings)

                if not is_batched(similarities):

                    similarities = np.expand_dims(similarities, 0)

                return similarities

            def _get_top_k(

                self,

                query_embeddings: np.ndarray,

                embeddings: np.ndarray,

                k: int = 1,

            ):

                similarities = self._get_similarities(query_embeddings, embeddings)

                top_indices = np.flip(

                    np.argsort(similarities, axis=-1)[:, -k:], axis=-1

                )  # B X k

                return top_indices, similarities

            def retrieve(

                self,

                query_embeddings: np.ndarray,

                embeddings: List[Any],

                k: int = 4,

            ) -> List[RetrievalOutput]:

                embeddings = np.array(embeddings)

                query_embeddings = np.array(query_embeddings)

                top_indices, similarities = self._get_top_k(query_embeddings, embeddings, k)

                # move to list for consistency w/ single and batch calls

                out = []

                for idx in range(top_indices.shape[0]):

                    query_output = RetrievalOutput(

                        embeddings[top_indices[idx]],

                        top_indices[idx],

                        scores=similarities[idx],

                        query_embeddings=query_embeddings[idx],

                    )

                    out.append(query_output)

                return out

------

#### Ancestors (in MRO)

* vexpresso.retrievers.base.BaseRetriever

#### Class variables

```python3
SUPPORTED_TYPES
```

#### Methods

    
#### retrieve

```python3
def retrieve(
    self,
    query_embeddings: numpy.ndarray,
    embeddings: List[Any],
    k: int = 4
) -> List[vexpresso.retrievers.base.RetrievalOutput]
```

Queries embeddings with query embedding vector and returns nearest embeddings and their corresponding ids

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| query_embeddings | np.ndarray | query, used to find nearest embeddings in set. | None |
| embeddings | List[Any] | embeddings set, query is compared to this. | None |

**Returns:**

| Type | Description |
|---|---|
| Union[List[QueryOutput], QueryOutput] | dataclasses containing returned embeddings and corresponding indices.<br>When this has more than one entry, that means that the call was batched |

??? example "View Source"
            def retrieve(

                self,

                query_embeddings: np.ndarray,

                embeddings: List[Any],

                k: int = 4,

            ) -> List[RetrievalOutput]:

                embeddings = np.array(embeddings)

                query_embeddings = np.array(query_embeddings)

                top_indices, similarities = self._get_top_k(query_embeddings, embeddings, k)

                # move to list for consistency w/ single and batch calls

                out = []

                for idx in range(top_indices.shape[0]):

                    query_output = RetrievalOutput(

                        embeddings[top_indices[idx]],

                        top_indices[idx],

                        scores=similarities[idx],

                        query_embeddings=query_embeddings[idx],

                    )

                    out.append(query_output)

                return out