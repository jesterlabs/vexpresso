from vexpresso.daft.collection import DaftCollection
import numpy as np

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

def test_default_collection_creation():
    data = {"test":list(range(100))}
    collection = DaftCollection(data=data)
    assert collection.to_dict() == data

    selected = collection["test"].to_list()[0]
    assert selected == list(range(100))


