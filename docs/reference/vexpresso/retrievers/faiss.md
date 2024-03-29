# Module vexpresso.retrievers.faiss

??? example "View Source"
        from typing import List

        import numpy as np

        from vexpresso.retrievers.base import BaseRetriever, RetrievalOutput

        

        class FaissRetriever(BaseRetriever):

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

                if not isinstance(embeddings, np.ndarray):

                    embeddings = np.array(embeddings)

                query_embeddings = np.array(query_embeddings)

                self._setup_index(embeddings)

                distances, indices = self.index.search(query_embeddings.astype(np.float32), k=k)

                out = []

                for idx in range(indices.shape[0]):

                    query_output = RetrievalOutput(

                        embeddings[indices[idx]],

                        indices[idx],

                        scores=distances[idx],

                        query_embeddings=query_embeddings,

                    )

                    out.append(query_output)

                return out

## Classes

### FaissRetriever

```python3
class FaissRetriever(
    
)
```

??? example "View Source"
        class FaissRetriever(BaseRetriever):

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

                if not isinstance(embeddings, np.ndarray):

                    embeddings = np.array(embeddings)

                query_embeddings = np.array(query_embeddings)

                self._setup_index(embeddings)

                distances, indices = self.index.search(query_embeddings.astype(np.float32), k=k)

                out = []

                for idx in range(indices.shape[0]):

                    query_output = RetrievalOutput(

                        embeddings[indices[idx]],

                        indices[idx],

                        scores=distances[idx],

                        query_embeddings=query_embeddings,

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
    embeddings: numpy.ndarray,
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

                embeddings: np.ndarray,

                k: int = 4,

            ) -> List[RetrievalOutput]:

                if not isinstance(embeddings, np.ndarray):

                    embeddings = np.array(embeddings)

                query_embeddings = np.array(query_embeddings)

                self._setup_index(embeddings)

                distances, indices = self.index.search(query_embeddings.astype(np.float32), k=k)

                out = []

                for idx in range(indices.shape[0]):

                    query_output = RetrievalOutput(

                        embeddings[indices[idx]],

                        indices[idx],

                        scores=distances[idx],

                        query_embeddings=query_embeddings,

                    )

                    out.append(query_output)

                return out