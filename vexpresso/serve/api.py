try:
    from fastapi import FastAPI
    from ray import serve
except Exception:
    pass

from typing import Any, Dict, List, Optional, Union

from vexpresso.collection import Collection
from vexpresso.retrievers import BaseRetriever
from vexpresso.utils import Transformation


def serve_collection(collection: Collection):
    app = FastAPI()

    @serve.deployment(route_prefix="/")
    @serve.ingress(app)
    class CollectionRayServe:
        def __init__(self, collection: Collection):
            self.collection = collection

        @app.get("/query")
        def query(
            self,
            column: str,
            query: List[Any] = None,
            query_embedding: List[Any] = None,
            filter_conditions: Optional[Dict[str, Dict[str, str]]] = None,
            k: int = None,
            sort: bool = True,
            embedding_fn: Optional[Union[Transformation, str]] = None,
            return_scores: bool = False,
            score_column_name: Optional[str] = None,
            retriever: Optional[Union[BaseRetriever, str]] = None,
        ) -> Dict[str, Any]:
            return self.collection.query(
                column,
                query,
                query_embedding,
                filter_conditions=filter_conditions,
                k=k,
                sort=sort,
                embedding_fn=embedding_fn,
                return_scores=return_scores,
                score_column_name=score_column_name,
                retriever=retriever,
            ).to_dict()

    serve.run(CollectionRayServe.bind(collection))
