from typing import Any, Callable, List, Union

from langchain.embeddings.base import Embeddings


class EmbeddingFunction:
    def __init__(self, embedding_fn: Union[Callable[[Any], Any], Embeddings]):
        self.embedding_fn = embedding_fn

    def embed(self, inp: Any, *args, **kwargs):
        if isinstance(self.embedding_fn, Embeddings):
            return self.embedding_fn.embed_query(inp)
        return self.embedding_fn(inp, *args, **kwargs)

    def batch_embed(self, inputs: List[Any], *args, **kwargs) -> Any:
        if isinstance(self.embedding_fn, Embeddings):
            return self.embedding_fn.embed_documents(inputs)
        return [self.embed(i, *args, **kwargs) for i in inputs]

    def __call__(self, inputs: Any, *args, **kwargs) -> Any:
        return self.embed(inputs, *args, **kwargs)
