# Module vexpresso.retriever.base

??? example "View Source"
        import abc

        from dataclasses import asdict, dataclass

        from typing import Any, Dict, Iterable, List, Optional, Union

        import numpy as np

        

        @dataclass

        class RetrievalOutput:

            embeddings: Any

            indices: Union[np.ndarray, Iterable[int]]

            scores: Union[np.ndarray, Iterable[float]]

            query_embeddings: Optional[Any] = None

            def dict(self) -> Dict[str, Any]:

                return {k: str(v) for k, v in asdict(self).items()}

        

        class BaseRetriever(metaclass=abc.ABCMeta):

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

## Classes

### BaseRetriever

```python3
class BaseRetriever(
    /,
    *args,
    **kwargs
)
```

??? example "View Source"
        class BaseRetriever(metaclass=abc.ABCMeta):

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

* vexpresso.retriever.faiss.FaissRetriever
* vexpresso.retriever.np.Retriever

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
) -> Union[List[vexpresso.retriever.base.RetrievalOutput], vexpresso.retriever.base.RetrievalOutput]
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

            def dict(self) -> Dict[str, Any]:

                return {k: str(v) for k, v in asdict(self).items()}

------

#### Class variables

```python3
query_embeddings
```

#### Methods

    
#### dict

```python3
def dict(
    self
) -> Dict[str, Any]
```

??? example "View Source"
            def dict(self) -> Dict[str, Any]:

                return {k: str(v) for k, v in asdict(self).items()}