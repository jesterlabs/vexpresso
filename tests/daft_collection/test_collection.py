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


def test_iloc():
    data = {"test":list(range(100))}
    collection = DaftCollection(data=data)
    actual = collection.iloc(10).to_dict()['test'][0]
    assert actual == 10

def test_sort():
    data = {"test":[10,4,9,10,0,0,1]}
    collection = DaftCollection(data=data)
    actual = collection.sort("test", desc=False).to_dict()["test"]
    assert actual == sorted([10,4,9,10,0,0,1])

def test_from_data():
    data = {"test":list(range(100))}
    collection = DaftCollection(data=data)
    new_data = {"test":list(range(10))}
    new_collection = collection.from_data(new_data)

    assert new_collection.retriever is collection.retriever
    assert new_collection.embedding_functions is collection.embedding_functions
    assert new_collection["test"].to_list()[0] == list(range(10))

def test_add_rows():
    data = {"test":[10,4,9], "test2":["1", "2", "3"]}
    collection = DaftCollection(data=data)
    new_data = [{"test":2, "test2":"test2"}, {"test":32, "test2":"test32"}]
    new_collection = collection.add_rows(new_data).collect()
    expected = {"test":[10,4,9,2,32], "test2":["1", "2", "3", "test2", "test32"]}
    assert new_collection.to_dict() == expected
