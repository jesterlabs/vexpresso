from re import M
from vexpresso.collection import Collection
import numpy as np
from vexpresso.retrieval import TopKRetrievalStrategy

seed = np.random.seed(1337)

def assert_list_equals(actual, expected):
    assert all([a == b for a, b in zip(actual, expected)])

def mock_embedding_function(texts):
        o = []
        for text in texts:
            if "test" in text:
                split_val = float(text.split("test")[-1])
                embedding = np.ones((768,))*split_val
            else:
                embedding = np.zeros((768,))
            o.append(embedding)
        return np.stack(o)


texts = [f"test{i}" for i in range(100)]
random_ints = np.random.randint(10, size=(len(texts),))
embeddings = mock_embedding_function(texts)
metadata = {"ints":random_ints}


def test_collection_with_single_query():
    strategy = TopKRetrievalStrategy('euclidian')
    _collection = Collection(
        content=texts,
        embeddings=embeddings,
        embedding_fn=mock_embedding_function,
        retrieval_strategy=strategy,
        metadata=metadata
    )

    query = ["test1"]
    out = _collection.query(query=query, k=3)
    expected = ["test1", "test0", "test2"]
    assert_list_equals(out.content, expected)

