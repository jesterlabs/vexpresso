import json
import os
import numpy as np
import pandas as pd
from tempfile import NamedTemporaryFile
import vexpresso

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


def test_create_collection():
    data = {"test":list(range(100))}
    collection = vexpresso.create(data, lazy=False)
    assert collection.to_dict() == data

    # from json
    tf = NamedTemporaryFile(delete=False, suffix=".json")
    with open(tf.name, 'w') as f:
        json.dump(data, f)
    collection = vexpresso.create(f.name)
    assert collection.to_dict() == data

    tf.close()
    os.unlink(tf.name)

    # from pandas
    df = pd.DataFrame(data)
    collection = vexpresso.create(df)
    assert collection.to_dict() == data
    assert collection.df is collection.daft_df

def test_repr():
    data = {"test":list(range(100))}
    collection = vexpresso.create(data, lazy=False)
    assert collection.__repr__() == collection.df.__repr__()

def test_len():
    data = {"test":list(range(100))}
    collection = vexpresso.create(data, lazy=False)
    assert len(collection) == 100

def test_cast():
    data = {"test":list(range(100)), "test2":list(range(100))}
    collection = vexpresso.create(data, lazy=False)
    collection = collection.cast("test", vexpresso.DataType.float64()).execute()
    assert np.array(collection.to_dict()["test"]).dtype == np.dtype("float64")

    collection = collection.cast(datatype=vexpresso.DataType.float64()).execute()
    assert np.array(collection.to_dict()["test"]).dtype == np.dtype("float64")
    assert np.array(collection.to_dict()["test2"]).dtype == np.dtype("float64")

def test_iloc():
    data = {"test":list(range(100))}
    collection = vexpresso.create(data=data)
    actual = collection.iloc(10).to_dict()['test'][0]
    assert actual == 10

def test_rename():
    data = {"test":list(range(100)), "test2":list(range(100,200)), "test3":list(range(200,300))}
    collection = vexpresso.create(data=data)
    renamed = collection.rename({"test":"new-name"}, lazy=False)
    assert renamed.column_names == ["new-name", "test2", "test3"]
    assert renamed.to_dict()["new-name"] == data["test"]

def test_add_column():
    data = {"test":list(range(100))}
    collection = vexpresso.create(data=data)

    data_2 = list(range(100))
    added = collection.add_column("new", data_2)
    assert added.to_dict()["new"] == data_2

def test_add_rows():
    data = {"test":list(range(100)), "test2":["s"]*100}
    collection = vexpresso.create(data=data)

    new_rows = [{"test":[1,2], "test2":["te", "p"]}]

    data_ = {"test":list(range(100)) + [1,2], "test2": ["s"]*100 + ["te", "p"]}

    added = collection.add_rows(new_rows)
    assert added.to_dict()["new"] == data_


def test_sort():
    data = {"test":[10,4,9,10,0,0,1]}
    collection = vexpresso.create(data=data)
    actual = collection.sort("test", desc=False).to_dict()["test"]
    assert actual == sorted([10,4,9,10,0,0,1])

def test_from_data():
    data = {"test":list(range(100))}
    collection = vexpresso.create(data=data)
    new_data = {"test":list(range(10))}
    new_collection = collection.from_data(new_data)

    assert new_collection.retriever is collection.retriever
    assert new_collection.embedding_functions is collection.embedding_functions
    assert new_collection["test"].to_list() == list(range(10))

def test_add_rows():
    data = {"test":[10,4,9], "test2":["1", "2", "3"]}
    collection = vexpresso.create(data=data)
    new_data = [{"test":2, "test2":"test2"}, {"test":32, "test2":"test32"}]
    new_collection = collection.add_rows(new_data).collect()
    expected = {"test":[10,4,9,2,32], "test2":["1", "2", "3", "test2", "test32"]}
    assert new_collection.to_dict() == expected
