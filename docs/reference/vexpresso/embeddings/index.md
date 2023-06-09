# Module vexpresso.embeddings

??? example "View Source"
        from vexpresso.embeddings.base import EmbeddingFunction

        from vexpresso.embeddings.sentence_transformers import (

            SentenceTransformerEmbeddingFunction,

        )

        from vexpresso.utils import Transformation, transformation

        

        def get_embedding_fn(embedding_fn: Transformation) -> Transformation:

            # langchain check

            if getattr(embedding_fn, "embed_documents", None) is not None:

                return transformation(embedding_fn, function="embed_documents")

            return transformation(embedding_fn)

        

        __all__ = [

            "EmbeddingFunction",

            "SentenceTransformerEmbeddingFunction",

            "get_embeddings_fn",

        ]

## Sub-modules

* [vexpresso.embeddings.base](base/)
* [vexpresso.embeddings.sentence_transformers](sentence_transformers/)

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

* vexpresso.embeddings.SentenceTransformerEmbeddingFunction

### SentenceTransformerEmbeddingFunction

```python3
class SentenceTransformerEmbeddingFunction(
    model: str = 'sentence-transformers/all-mpnet-base-v2',
    output_type: str = 'np',
    *args,
    **kwargs
)
```

??? example "View Source"
        class SentenceTransformerEmbeddingFunction(EmbeddingFunction):

            def __init__(

                self, model: str = DEFAULT_MODEL, output_type: str = "np", *args, **kwargs

            ):

                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(model, *args, **kwargs)

                self.output_type = output_type

            def __call__(self, list_of_texts: List[str]):

                out = self.model.encode(list_of_texts, convert_to_tensor=True)

                if self.output_type == "np":

                    return out.detach().cpu().numpy()

                if self.output_type == "list":

                    return out.tolist()

                return out

------

#### Ancestors (in MRO)

* vexpresso.embeddings.EmbeddingFunction