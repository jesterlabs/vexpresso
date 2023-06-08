# Module vexpresso.retriever

??? example "View Source"
        from vexpresso.retriever.faiss import FaissRetriever

        from vexpresso.retriever.np import NumpyRetriever

        from vexpresso.retriever.retriever import RetrievalOutput, Retriever

        __all__ = [

            "Retriever",

            "NumpyRetriever",

            "RetrievalOutput",

            "FaissRetriever",

        ]

## Sub-modules

* [vexpresso.retriever.faiss](faiss/)
* [vexpresso.retriever.np](np/)
* [vexpresso.retriever.retriever](retriever/)

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

                query_embeddings = np.array(query_embeddings)

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

* vexpresso.retriever.Retriever

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

                query_embeddings = np.array(query_embeddings)

                self._setup_index(embeddings)

                distances, indices = self.index.search(query_embeddings.astype(np.float32), k=k)

                out = []

                for indices in indices:

                    query_output = RetrievalOutput(

                        embeddings[indices], indices, query_embeddings, scores=distances

                    )

                    out.append(query_output)

                return out

### NumpyRetriever

```python3
class NumpyRetriever(
    similarity_fn: str = 'cosine'
)
```

??? example "View Source"
        class NumpyRetriever(Retriever):

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

* vexpresso.retriever.Retriever

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

### RetrievalOutput

```python3
class RetrievalOutput(
    embeddings: Any,
    indices: Union[numpy.ndarray, Iterable[int]],
    scores: Union[numpy.ndarray, Iterable[float]],
    query_embeddings: Optional[Any] = None
)
```

RetrievalOutput(embeddings: Any, indices: Union[numpy.ndarray, Iterable[int]], scores: Union[numpy.ndarray, Iterable[float]], query_embeddings: Optional[Any] = None)

??? example "View Source"
        @dataclass

        class RetrievalOutput:

            embeddings: Any

            indices: Union[np.ndarray, Iterable[int]]

            scores: Union[np.ndarray, Iterable[float]]

            query_embeddings: Optional[Any] = None

------

#### Class variables

```python3
query_embeddings
```

### Retriever

```python3
class Retriever(
    /,
    *args,
    **kwargs
)
```

??? example "View Source"
        class Retriever(metaclass=abc.ABCMeta):

            SUPPORTED_TYPES = [np.dtype]

            @abc.abstractmethod

            def retrieve(

                self, query_embeddings: np.ndarray, embeddings: List[Any], *args, **kwargs

            ) -> Union[List[RetrievalOutput], RetrievalOutput]:

                """

                Queries embeddings with query embedding vector and returns nearest embeddings and their corresponding ids

        

                Args:

                    query_embeddings (np.ndarray): query, used to find nearest embeddings in set.

                    embeddings (List[Any]): embeddings set, query is compared to this.

                Returns:

                    Union[List[QueryOutput], QueryOutput]: dataclasses containing returned embeddings and corresponding indices.

                    When this has more than one entry, that means that the call was batched

                """

------

#### Descendants

* vexpresso.retriever.FaissRetriever
* vexpresso.retriever.NumpyRetriever

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
    *args,
    **kwargs
) -> Union[List[vexpresso.retriever.retriever.RetrievalOutput], vexpresso.retriever.retriever.RetrievalOutput]
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
            @abc.abstractmethod

            def retrieve(

                self, query_embeddings: np.ndarray, embeddings: List[Any], *args, **kwargs

            ) -> Union[List[RetrievalOutput], RetrievalOutput]:

                """

                Queries embeddings with query embedding vector and returns nearest embeddings and their corresponding ids

        

                Args:

                    query_embeddings (np.ndarray): query, used to find nearest embeddings in set.

                    embeddings (List[Any]): embeddings set, query is compared to this.

                Returns:

                    Union[List[QueryOutput], QueryOutput]: dataclasses containing returned embeddings and corresponding indices.

                    When this has more than one entry, that means that the call was batched

                """