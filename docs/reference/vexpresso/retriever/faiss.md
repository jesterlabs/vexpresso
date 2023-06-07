# Module vexpresso.retriever.faiss

??? example "View Source"
        from typing import List

        import numpy as np

        from vexpresso.retriever.retriever import RetrievalOutput, Retriever

        

        class FaissRetriever(Retriever):

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

                distances, indices = self.index.search(query_embeddings.astype(np.float32), k=k)

                out = []

                for indices in indices:

                    query_output = RetrievalOutput(

                        embeddings[indices], indices, query_embeddings, scores=distances

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
        class FaissRetriever(Retriever):

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

                distances, indices = self.index.search(query_embeddings.astype(np.float32), k=k)

                out = []

                for indices in indices:

                    query_output = RetrievalOutput(

                        embeddings[indices], indices, query_embeddings, scores=distances

                    )

                    out.append(query_output)

                return out

------

#### Ancestors (in MRO)

* vexpresso.retriever.retriever.Retriever

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
) -> List[vexpresso.retriever.retriever.RetrievalOutput]
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

                self._setup_index(embeddings)

                distances, indices = self.index.search(query_embeddings.astype(np.float32), k=k)

                out = []

                for indices in indices:

                    query_output = RetrievalOutput(

                        embeddings[indices], indices, query_embeddings, scores=distances

                    )

                    out.append(query_output)

                return out