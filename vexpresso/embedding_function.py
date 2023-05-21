from typing import Any, Callable, List


class EmbeddingFunction:
    def __init__(self, embedding_fn: Callable[[Any], Any]):
        self.embedding_fn = embedding_fn

    def embed(self, inp: Any, *args, **kwargs) -> Any:
        return self.embedding_fn(inp, *args, **kwargs)

    def batch_embed(self, inputs: List[Any], *args, **kwargs) -> Any:
        return [self.embed(i, *args, **kwargs) for i in inputs]

    def __call__(self, inputs: Any, *args, **kwargs) -> Any:
        return self.embed(inputs, *args, **kwargs)


class LangChainEmbeddingsFunction(EmbeddingFunction):
    def __init__(self, embedding_fn):
        try:
            from langchain.embeddings.base import Embeddings  # noqa
            self.embedding_fn = embedding_fn
        except ImportError:
            raise ImportError(
                "Could not import langchain python package."
                "Please install it with `pip install langchain`."
            )

    def embed(self, inp: Any, *args, **kwargs):
        return self.embedding_fn.embed_query(inp, *args, **kwargs)

    def batch_embed(self, inputs: List[Any], *args, **kwargs) -> Any:
        return self.embedding_fn.embed_documents(inputs, *args, **kwargs)
