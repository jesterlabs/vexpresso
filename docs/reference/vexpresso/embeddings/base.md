# Module vexpresso.embeddings.base

??? example "View Source"
        import abc

        from typing import Any, Dict, List

        from vexpresso.utils import DataType, Transformation, transformation

        

        def get_embedding_fn(

            embedding_fn: Transformation,

            datatype: DataType = DataType.python(),

            init_kwargs: Dict[str, Any] = {},

        ) -> Transformation:

            # langchain check

            if getattr(embedding_fn, "embed_documents", None) is not None:

                return transformation(

                    embedding_fn,

                    datatype=datatype,

                    function="embed_documents",

                    init_kwargs=init_kwargs,

                )

            return transformation(embedding_fn, datatype=datatype, init_kwargs=init_kwargs)

        

        class EmbeddingFunction(metaclass=abc.ABCMeta):

            @abc.abstractmethod

            def __call__(self, column: List[Any], *args, **kwargs):

                """

                This is the main function of `embedding function` to be applied on a column

                """

## Functions

    
### get_embedding_fn

```python3
def get_embedding_fn(
    embedding_fn: Callable[[List[Any], Any], List[Any]],
    datatype: daft.datatype.DataType = Python,
    init_kwargs: Dict[str, Any] = {}
) -> Callable[[List[Any], Any], List[Any]]
```

??? example "View Source"
        def get_embedding_fn(

            embedding_fn: Transformation,

            datatype: DataType = DataType.python(),

            init_kwargs: Dict[str, Any] = {},

        ) -> Transformation:

            # langchain check

            if getattr(embedding_fn, "embed_documents", None) is not None:

                return transformation(

                    embedding_fn,

                    datatype=datatype,

                    function="embed_documents",

                    init_kwargs=init_kwargs,

                )

            return transformation(embedding_fn, datatype=datatype, init_kwargs=init_kwargs)

## Classes

### EmbeddingFunction

```python3
class EmbeddingFunction(
    /,
    *args,
    **kwargs
)
```

??? example "View Source"
        class EmbeddingFunction(metaclass=abc.ABCMeta):

            @abc.abstractmethod

            def __call__(self, column: List[Any], *args, **kwargs):

                """

                This is the main function of `embedding function` to be applied on a column

                """

------

#### Descendants

* vexpresso.embeddings.clip.ClipEmbeddingsFunction
* vexpresso.embeddings.sentence_transformers.SentenceTransformerEmbeddingFunction