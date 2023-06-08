# Module vexpresso.embeddings.embedding_fn

??? example "View Source"
        import abc

        from typing import Any, List

        

        class EmbeddingFunction(metaclass=abc.ABCMeta):

            @abc.abstractmethod

            def __call__(self, column: List[Any], *args, **kwargs):

                """

                This is the main function of `embedding function` to be applied on a column

                """

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

* vexpresso.embeddings.sentence_transformers.SentenceTransformerEmbeddingFunction