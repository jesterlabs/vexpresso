# Module vexpresso.embeddings

??? example "View Source"
        from vexpresso.embeddings.base import EmbeddingFunction, get_embedding_fn

        from vexpresso.embeddings.clip import ClipEmbeddingsFunction

        from vexpresso.embeddings.sentence_transformers import (

            SentenceTransformerEmbeddingFunction,

        )

        __all__ = [

            "EmbeddingFunction",

            "SentenceTransformerEmbeddingFunction",

            "ClipEmbeddingsFunction",

            "get_embedding_fn",

        ]

## Sub-modules

* [vexpresso.embeddings.base](base/)
* [vexpresso.embeddings.clip](clip/)
* [vexpresso.embeddings.sentence_transformers](sentence_transformers/)

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

### ClipEmbeddingsFunction

```python3
class ClipEmbeddingsFunction(
    model: str = 'openai/clip-vit-base-patch32'
)
```

??? example "View Source"
        class ClipEmbeddingsFunction(EmbeddingFunction):

            def __init__(self, model: str = DEFAULT_MODEL):

                import torch

                from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

                self.model = CLIPModel.from_pretrained(model)

                self.processor = CLIPProcessor.from_pretrained(model)

                self.tokenizer = CLIPTokenizerFast.from_pretrained(model)

                self.device = torch.device("cpu")

                if torch.cuda.is_available():

                    self.device = torch.device("cuda")

                    self.model = self.model.to(self.device)

            def __call__(self, inp, inp_type: str):

                if inp_type == "image":

                    inputs = self.processor(images=inp, return_tensors="pt", padding=True)[

                        "pixel_values"

                    ].to(self.device)

                    return self.model.get_image_features(inputs).detach().cpu().numpy()

                if inp_type == "text":

                    inputs = self.tokenizer(inp, padding=True, return_tensors="pt")

                    inputs["input_ids"] = inputs["input_ids"].to(self.device)

                    inputs["attention_mask"] = inputs["attention_mask"].to(self.device)

                    return self.model.get_text_features(**inputs).detach().cpu().numpy()

------

#### Ancestors (in MRO)

* vexpresso.embeddings.EmbeddingFunction

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

* vexpresso.embeddings.ClipEmbeddingsFunction
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