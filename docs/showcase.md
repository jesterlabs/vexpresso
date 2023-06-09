# Vexpresso Showcase

#### This is a simple tutorial to go over vexpresso capabilities

Imports


```python
import vexpresso
import numpy as np
from vexpresso.retriever import Retriever
```

## Collection Creation

#### First we'll create some sample data. Here we're using just strings, but because `vexpresso` uses `daft`, you can use any datatype!


```python
data = {"numbers":list(range(1, 100)), "strings":[f"test_{i}" for i in range(1, 100)]}
```

#### To create the collection, use the `create` method. Lets also use a NumpyRetriever that uses euclidian distance. This by default is lazy execution, meaning that we actually don't load in any data until `execute` or `show` is called. (Or if `lazy` is passed)


```python
collection = vexpresso.create(data=data, retriever=Retriever(similarity_fn="euclidian"))
collection.daft_df
```

    2023-06-09 13:41:54.170 | INFO     | daft.context:runner:80 - Using PyRunner





<div>
    <table class="dataframe">
<tbody>
<tr><td>numbers<br>Int64</td><td>strings<br>Utf8</td><td>vexpresso_index<br>Int64</td></tr>
</tbody>
</table>
    <small>(No data to display: Dataframe not materialized)</small>
</div>



#### Lets see what's in the collection now!


```python
collection.show(5)
```




<div>
    <table class="dataframe">
<thead>
<tr><th style="text-align: right;">  numbers<br>Int64</th><th>strings<br>Utf8  </th><th style="text-align: right;">  vexpresso_index<br>Int64</th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">                 1</td><td>test_1           </td><td style="text-align: right;">                         0</td></tr>
<tr><td style="text-align: right;">                 2</td><td>test_2           </td><td style="text-align: right;">                         1</td></tr>
<tr><td style="text-align: right;">                 3</td><td>test_3           </td><td style="text-align: right;">                         2</td></tr>
<tr><td style="text-align: right;">                 4</td><td>test_4           </td><td style="text-align: right;">                         3</td></tr>
<tr><td style="text-align: right;">                 5</td><td>test_5           </td><td style="text-align: right;">                         4</td></tr>
</tbody>
</table>
    <small>(Showing first 5 rows)</small>
</div>



#### vexpresso's `Collection` methods return `Collection` objects, allowing for complex chaining of calls

## Querying

#### Lets embed the data using a simple "fake" embedding function


```python
import numpy as np

def embed_fn(strings):
    return [np.array([float(s.split("_")[-1])]*100) for s in strings]
```


```python
collection = collection.embed("strings", embedding_fn=embed_fn) # returns a new collection
```

#### By default vexpresso is "lazy", meaning that nothing is executed until `.execute` is called
Note: this can be bypassed by passing `lazy=False`

```python
collection = collection.embed("strings", embedding_fn=embed_fn, lazy=False)
```


```python
collection.daft_df
```




<div>
    <table class="dataframe">
<tbody>
<tr><td>numbers<br>Int64</td><td>strings<br>Utf8</td><td>vexpresso_index<br>Int64</td><td>embeddings_strings<br>Python</td></tr>
</tbody>
</table>
    <small>(No data to display: Dataframe not materialized)</small>
</div>



#### Let's execute it to get embeddings


```python
collection = collection.execute()
```


```python
collection.show(5)
```




<div>
    <table class="dataframe">
<thead>
<tr><th style="text-align: right;">  numbers<br>Int64</th><th>strings<br>Utf8  </th><th style="text-align: right;">  vexpresso_index<br>Int64</th><th>embeddings_strings<br>Python                     </th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">                 1</td><td>test_1           </td><td style="text-align: right;">                         0</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td></tr>
<tr><td style="text-align: right;">                 2</td><td>test_2           </td><td style="text-align: right;">                         1</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td></tr>
<tr><td style="text-align: right;">                 3</td><td>test_3           </td><td style="text-align: right;">                         2</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td></tr>
<tr><td style="text-align: right;">                 4</td><td>test_4           </td><td style="text-align: right;">                         3</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td></tr>
<tr><td style="text-align: right;">                 5</td><td>test_5           </td><td style="text-align: right;">                         4</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td></tr>
</tbody>
</table>
    <small>(Showing first 5 rows)</small>
</div>




```python
collection.to_dict()["embeddings_strings"][:3]
```




    [array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]),
     array([2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
            2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
            2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
            2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
            2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
            2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]),
     array([3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
            3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
            3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
            3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
            3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
            3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])]



#### as you can see we now have an `embeddings_strings` column, let's query it and return the top 5 results!


```python
embed_fn(["test_3"])
```




    [array([3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
            3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
            3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
            3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
            3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
            3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])]




```python
queried = collection.query("embeddings_strings", query="test_3", k=5).execute()
```

#### As expected, the closest strings to `test_3` (according to our embedding function above) are `test_3`, `test_2`, `test_4`, `test_1`, `test_5`.
#### In addition, we can see the actual similarity scores in `embeddings_strings_score` column


```python
queried.show(5)
```




<div>
    <table class="dataframe">
<thead>
<tr><th style="text-align: right;">  numbers<br>Int64</th><th>strings<br>Utf8  </th><th style="text-align: right;">  vexpresso_index<br>Int64</th><th>embeddings_strings<br>Python                     </th><th style="text-align: right;">  embeddings_strings_score<br>Float64</th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">                 3</td><td>test_3           </td><td style="text-align: right;">                         2</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td style="text-align: right;">                            1        </td></tr>
<tr><td style="text-align: right;">                 2</td><td>test_2           </td><td style="text-align: right;">                         1</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td style="text-align: right;">                            0.0909091</td></tr>
<tr><td style="text-align: right;">                 4</td><td>test_4           </td><td style="text-align: right;">                         3</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td style="text-align: right;">                            0.0909091</td></tr>
<tr><td style="text-align: right;">                 1</td><td>test_1           </td><td style="text-align: right;">                         0</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td style="text-align: right;">                            0.047619 </td></tr>
<tr><td style="text-align: right;">                 5</td><td>test_5           </td><td style="text-align: right;">                         4</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td style="text-align: right;">                            0.047619 </td></tr>
</tbody>
</table>
    <small>(Showing first 5 rows)</small>
</div>



#### Sometimes you will want to batch queries together into a single call. vexpresso has a convenient `batch_query` function. This will return a list of Collections


```python
queries = ["test_1", "test_5", "test_10"]
```


```python
batch_queried = collection.batch_query("embeddings_strings", queries=queries, k=2)
```

#### We now have collections for each query


```python
batch_queried[0].show(2)
```




<div>
    <table class="dataframe">
<thead>
<tr><th style="text-align: right;">  numbers<br>Int64</th><th>strings<br>Utf8  </th><th style="text-align: right;">  vexpresso_index<br>Int64</th><th>embeddings_strings<br>Python                     </th><th style="text-align: right;">  embeddings_strings_score<br>Float64</th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">                10</td><td>test_10          </td><td style="text-align: right;">                         9</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td style="text-align: right;">                            1        </td></tr>
<tr><td style="text-align: right;">                11</td><td>test_11          </td><td style="text-align: right;">                        10</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td style="text-align: right;">                            0.0909091</td></tr>
</tbody>
</table>
    <small>(Showing first 2 rows)</small>
</div>




```python
batch_queried[1].show(2)
```




<div>
    <table class="dataframe">
<thead>
<tr><th style="text-align: right;">  numbers<br>Int64</th><th>strings<br>Utf8  </th><th style="text-align: right;">  vexpresso_index<br>Int64</th><th>embeddings_strings<br>Python                     </th><th style="text-align: right;">  embeddings_strings_score<br>Float64</th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">                10</td><td>test_10          </td><td style="text-align: right;">                         9</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td style="text-align: right;">                            1        </td></tr>
<tr><td style="text-align: right;">                11</td><td>test_11          </td><td style="text-align: right;">                        10</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td style="text-align: right;">                            0.0909091</td></tr>
</tbody>
</table>
    <small>(Showing first 2 rows)</small>
</div>




```python
batch_queried[2].show(2)
```




<div>
    <table class="dataframe">
<thead>
<tr><th style="text-align: right;">  numbers<br>Int64</th><th>strings<br>Utf8  </th><th style="text-align: right;">  vexpresso_index<br>Int64</th><th>embeddings_strings<br>Python                     </th><th style="text-align: right;">  embeddings_strings_score<br>Float64</th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">                10</td><td>test_10          </td><td style="text-align: right;">                         9</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td style="text-align: right;">                            1        </td></tr>
<tr><td style="text-align: right;">                11</td><td>test_11          </td><td style="text-align: right;">                        10</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td style="text-align: right;">                            0.0909091</td></tr>
</tbody>
</table>
    <small>(Showing first 2 rows)</small>
</div>



## Filtering

#### With `vexpresso`, filtering is super easy. The syntax is similar to `chromadb`

#### Filter dictionary must have the following structure:

```python
{
    <field>: {
        <filter_method>: <value>
    },
    <field>: {
        <filter_method>: <value>
    },
}

```

Let's filter the original collection to only include rows with `numbers` > 95


```python
filtered_collection = collection.filter(
    {
        "numbers":{
            "gt":95
        }
    }
).execute()
```


```python
filtered_collection.show(5)
```




<div>
    <table class="dataframe">
<thead>
<tr><th style="text-align: right;">  numbers<br>Int64</th><th>strings<br>Utf8  </th><th style="text-align: right;">  vexpresso_index<br>Int64</th><th>embeddings_strings<br>Python                     </th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">                96</td><td>test_96          </td><td style="text-align: right;">                        95</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td></tr>
<tr><td style="text-align: right;">                97</td><td>test_97          </td><td style="text-align: right;">                        96</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td></tr>
<tr><td style="text-align: right;">                98</td><td>test_98          </td><td style="text-align: right;">                        97</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td></tr>
<tr><td style="text-align: right;">                99</td><td>test_99          </td><td style="text-align: right;">                        98</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td></tr>
</tbody>
</table>
    <small>(Showing first 4 rows)</small>
</div>



#### We can use multiple filter conditions as well
Let's filter the collection to only return rows with numbers <= 50 and strings with "0" in them


```python
filtered_collection = collection.filter(
    {
        "numbers":{
            "lte":50
        },
        "strings":{
            "contains":"0"
        }
    }
).execute()
```


```python
filtered_collection.show(5)
```




<div>
    <table class="dataframe">
<thead>
<tr><th style="text-align: right;">  numbers<br>Int64</th><th>strings<br>Utf8  </th><th style="text-align: right;">  vexpresso_index<br>Int64</th><th>embeddings_strings<br>Python                     </th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">                10</td><td>test_10          </td><td style="text-align: right;">                         9</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td></tr>
<tr><td style="text-align: right;">                20</td><td>test_20          </td><td style="text-align: right;">                        19</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td></tr>
<tr><td style="text-align: right;">                30</td><td>test_30          </td><td style="text-align: right;">                        29</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td></tr>
<tr><td style="text-align: right;">                40</td><td>test_40          </td><td style="text-align: right;">                        39</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td></tr>
<tr><td style="text-align: right;">                50</td><td>test_50          </td><td style="text-align: right;">                        49</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td></tr>
</tbody>
</table>
    <small>(Showing first 5 rows)</small>
</div>



#### Sometimes you need a custom filtering function, with vexpresso its easy to do that with the `custom` filter keyword!
Lets filter a collection to only return rows with even `numbers` and `strings` that contain a "3"


```python
def custom_filter(number, mod_val) -> bool:
    return number % mod_val == 0
```


```python
filtered_collection = collection.filter(
    {
        "numbers":{
            "custom":{"function":custom_filter, "function_kwargs":{"mod_val":2}}
        },
        "strings":{
            "contains":"3"
        }
    }
).execute()
```


```python
filtered_collection.show(5)
```




<div>
    <table class="dataframe">
<thead>
<tr><th style="text-align: right;">  numbers<br>Int64</th><th>strings<br>Utf8  </th><th style="text-align: right;">  vexpresso_index<br>Int64</th><th>embeddings_strings<br>Python                     </th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">                30</td><td>test_30          </td><td style="text-align: right;">                        29</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td></tr>
<tr><td style="text-align: right;">                32</td><td>test_32          </td><td style="text-align: right;">                        31</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td></tr>
<tr><td style="text-align: right;">                34</td><td>test_34          </td><td style="text-align: right;">                        33</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td></tr>
<tr><td style="text-align: right;">                36</td><td>test_36          </td><td style="text-align: right;">                        35</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td></tr>
<tr><td style="text-align: right;">                38</td><td>test_38          </td><td style="text-align: right;">                        37</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td></tr>
</tbody>
</table>
    <small>(Showing first 5 rows)</small>
</div>



#### You can also combine filters + queries in the same call

 Lets query the collection with "test_10" and filter only even numbers


```python
even_filter = {
    "numbers":{
        "custom":{"function":custom_filter, "function_kwargs":{"mod_val":2}}
    }
}
```


```python
query_filtered_collection = collection.query("embeddings_strings", "test_10", k=10, filter_conditions=even_filter).execute()
```


```python
query_filtered_collection.show(5)
```




<div>
    <table class="dataframe">
<thead>
<tr><th style="text-align: right;">  numbers<br>Int64</th><th>strings<br>Utf8  </th><th style="text-align: right;">  vexpresso_index<br>Int64</th><th>embeddings_strings<br>Python                     </th><th style="text-align: right;">  embeddings_strings_score<br>Float64</th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">                10</td><td>test_10          </td><td style="text-align: right;">                         9</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td style="text-align: right;">                            1        </td></tr>
<tr><td style="text-align: right;">                 8</td><td>test_8           </td><td style="text-align: right;">                         7</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td style="text-align: right;">                            0.047619 </td></tr>
<tr><td style="text-align: right;">                12</td><td>test_12          </td><td style="text-align: right;">                        11</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td style="text-align: right;">                            0.047619 </td></tr>
<tr><td style="text-align: right;">                 6</td><td>test_6           </td><td style="text-align: right;">                         5</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td style="text-align: right;">                            0.0243902</td></tr>
<tr><td style="text-align: right;">                14</td><td>test_14          </td><td style="text-align: right;">                        13</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td style="text-align: right;">                            0.0243902</td></tr>
</tbody>
</table>
    <small>(Showing first 5 rows)</small>
</div>



## Chaining Functions

#### We can chain functions lazily easily

For instance, lets query and filter multiple times


```python
even_filter = {
    "numbers":{
        "custom":{"function":custom_filter, "function_kwargs":{"mod_val":2}}
    }
}
```


```python
chained_collection = collection.query("embeddings_strings", "test_10", k=50) \
                               .filter(even_filter) \
                               .query("embeddings_strings", "test_30", k=50) \
                               .filter({"numbers":{"lte":30}})
```


```python
chained_collection.daft_df
```




<div>
    <table class="dataframe">
<tbody>
<tr><td>numbers<br>Int64</td><td>strings<br>Utf8</td><td>vexpresso_index<br>Int64</td><td>embeddings_strings<br>Python</td><td>embeddings_strings_score<br>Float64</td></tr>
</tbody>
</table>
    <small>(No data to display: Dataframe not materialized)</small>
</div>



Here we queried for the closest 50 elements to "test_10", filtered for only even numbers, queried top 50 of "test_30", then filtered for numbers <= 30


```python
chained_collection = chained_collection.execute()
```


```python
chained_collection.show(5)
```




<div>
    <table class="dataframe">
<thead>
<tr><th style="text-align: right;">  numbers<br>Int64</th><th>strings<br>Utf8  </th><th style="text-align: right;">  vexpresso_index<br>Int64</th><th>embeddings_strings<br>Python                     </th><th style="text-align: right;">  embeddings_strings_score<br>Float64</th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">                30</td><td>test_30          </td><td style="text-align: right;">                        29</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td style="text-align: right;">                            1        </td></tr>
<tr><td style="text-align: right;">                28</td><td>test_28          </td><td style="text-align: right;">                        27</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td style="text-align: right;">                            0.047619 </td></tr>
<tr><td style="text-align: right;">                26</td><td>test_26          </td><td style="text-align: right;">                        25</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td style="text-align: right;">                            0.0243902</td></tr>
<tr><td style="text-align: right;">                24</td><td>test_24          </td><td style="text-align: right;">                        23</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td style="text-align: right;">                            0.0163934</td></tr>
<tr><td style="text-align: right;">                22</td><td>test_22          </td><td style="text-align: right;">                        21</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td style="text-align: right;">                            0.0123457</td></tr>
</tbody>
</table>
    <small>(Showing first 5 rows)</small>
</div>



## Transforms

#### Sometimes you want to transform your data. Because of `daft`, you can use `vexpresso` to do this easily! 

#### For example, lets add a new column where we change "test" to "example" in the strings column. Lets specify that this output is also a string type

For a full list of datatypes, visit daft documentation: https://www.getdaft.io/projects/docs/en/latest/api_docs/datatype.html


```python
def simple_apply_fn(strings):
    return [
        s.replace("test", "example") for s in strings
    ]
```


```python
transformed_collection = collection.apply(simple_apply_fn, collection["strings"], datatype=vexpresso.DataType.string()).execute()
```


```python
transformed_collection.show(5)
```




<div>
    <table class="dataframe">
<thead>
<tr><th style="text-align: right;">  numbers<br>Int64</th><th>strings<br>Utf8  </th><th style="text-align: right;">  vexpresso_index<br>Int64</th><th>embeddings_strings<br>Python                     </th><th>tranformed_strings<br>Utf8  </th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">                 1</td><td>test_1           </td><td style="text-align: right;">                         0</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td>example_1                   </td></tr>
<tr><td style="text-align: right;">                 2</td><td>test_2           </td><td style="text-align: right;">                         1</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td>example_2                   </td></tr>
<tr><td style="text-align: right;">                 3</td><td>test_3           </td><td style="text-align: right;">                         2</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td>example_3                   </td></tr>
<tr><td style="text-align: right;">                 4</td><td>test_4           </td><td style="text-align: right;">                         3</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td>example_4                   </td></tr>
<tr><td style="text-align: right;">                 5</td><td>test_5           </td><td style="text-align: right;">                         4</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td>example_5                   </td></tr>
</tbody>
</table>
    <small>(Showing first 5 rows)</small>
</div>



#### We can also pass in args, kwargs, and multiple columns into the apply function

For instance, lets replace the "test" chars in string column with "modified" and also replace the suffix with `number` times 1000. In addition lets name the column `modified`


```python
def multi_column_apply_fn(string_columns, numbers):
    out = []
    for string, num in zip(string_columns, numbers):
        replaced = string.replace("test", "modified").split("_")[0]
        modified = f"{replaced}_{num*1000}"
        out.append(modified)
    return out
```


```python
transformed_collection = collection.apply(
    multi_column_apply_fn,
    collection["strings"],
    numbers=collection["numbers"],
    to="modified",
    datatype=vexpresso.DataType.string()
).execute()
```


```python
transformed_collection.show(5)
```




<div>
    <table class="dataframe">
<thead>
<tr><th style="text-align: right;">  numbers<br>Int64</th><th>strings<br>Utf8  </th><th style="text-align: right;">  vexpresso_index<br>Int64</th><th>embeddings_strings<br>Python                     </th><th>modified<br>Utf8  </th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">                 1</td><td>test_1           </td><td style="text-align: right;">                         0</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td>modified_1000     </td></tr>
<tr><td style="text-align: right;">                 2</td><td>test_2           </td><td style="text-align: right;">                         1</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td>modified_2000     </td></tr>
<tr><td style="text-align: right;">                 3</td><td>test_3           </td><td style="text-align: right;">                         2</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td>modified_3000     </td></tr>
<tr><td style="text-align: right;">                 4</td><td>test_4           </td><td style="text-align: right;">                         3</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td>modified_4000     </td></tr>
<tr><td style="text-align: right;">                 5</td><td>test_5           </td><td style="text-align: right;">                         4</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td>modified_5000     </td></tr>
</tbody>
</table>
    <small>(Showing first 5 rows)</small>
</div>



## Saving + Loading

#### Once you've done a bunch of processing on a collection, you probably want to save it somewhere. Vexpresso supports local file saving + huggingface datasets

Lets save the `transformed_collection` above to a directory `saved_transformed_collection`


```python
transformed_collection.save("./saved_transformed_collection")
```

    saving to ./saved_transformed_collection


We can then load the collection with the same `create` function. Make sure to also include the embedding functions that were used on the original collection!


```python
loaded_collection = vexpresso.create(
    directory_or_repo_id = "saved_transformed_collection",
    embedding_functions = {"embeddings_strings":embed_fn}
)
```


```python
loaded_collection.show(5)
```




<div>
    <table class="dataframe">
<thead>
<tr><th style="text-align: right;">  numbers<br>Int64</th><th>strings<br>Utf8  </th><th style="text-align: right;">  vexpresso_index<br>Int64</th><th>embeddings_strings<br>Python                     </th><th>modified<br>Utf8  </th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">                 1</td><td>test_1           </td><td style="text-align: right;">                         0</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td>modified_1000     </td></tr>
<tr><td style="text-align: right;">                 2</td><td>test_2           </td><td style="text-align: right;">                         1</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td>modified_2000     </td></tr>
<tr><td style="text-align: right;">                 3</td><td>test_3           </td><td style="text-align: right;">                         2</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td>modified_3000     </td></tr>
<tr><td style="text-align: right;">                 4</td><td>test_4           </td><td style="text-align: right;">                         3</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td>modified_4000     </td></tr>
<tr><td style="text-align: right;">                 5</td><td>test_5           </td><td style="text-align: right;">                         4</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td>modified_5000     </td></tr>
</tbody>
</table>
    <small>(Showing first 5 rows)</small>
</div>



#### Now let's upload to huggingface!

For this you'll need to install huggingfacehub


```python
# !pip install huggingface-hub
```

Automatically gets token from env variable: HUGGINGFACEHUB_API_TOKEN = ...

or you can pass in token directly via `collection.save(token=...)`


```python
# username = "shyamsn97"
# repo_name = "vexpresso_test_showcase"
username = "REPLACE"
repo_name = "REPLACE"
```


```python
loaded_collection.save(hf_username = username, repo_name = repo_name, to_hub=True, )
```

    Uploading collection to None


    /home/shyam/miniconda3/envs/py39/lib/python3.9/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


    Upload to shyamsn97/vexpresso_test_showcase complete!





    'shyamsn97/vexpresso_test_showcase'



The example is private by default, but this can be changed by the `private` flag


```python
# loaded_collection.save(hf_username = username, repo_name = repo_name, to_hub=True, private=False)
```

You can see an example of the above data: https://huggingface.co/datasets/shyamsn97/vexpresso_test_showcase

#### Now lets load it!


```python
loaded_collection = vexpresso.create(
    hf_username = username,
    repo_name = repo_name,
    embedding_functions = {"embeddings_strings":embed_fn}
)
```

    Retrieving from hf repo: shyamsn97/vexpresso_test_showcase


    Fetching 2 files: 100%|███████████████████████████████████████████████████| 2/2 [00:00<00:00,  7.04it/s]



```python
loaded_collection.show(5)
```




<div>
    <table class="dataframe">
<thead>
<tr><th style="text-align: right;">  numbers<br>Int64</th><th>strings<br>Utf8  </th><th style="text-align: right;">  vexpresso_index<br>Int64</th><th>embeddings_strings<br>Python                     </th><th>modified<br>Utf8  </th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">                 1</td><td>test_1           </td><td style="text-align: right;">                         0</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td>modified_1000     </td></tr>
<tr><td style="text-align: right;">                 2</td><td>test_2           </td><td style="text-align: right;">                         1</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td>modified_2000     </td></tr>
<tr><td style="text-align: right;">                 3</td><td>test_3           </td><td style="text-align: right;">                         2</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td>modified_3000     </td></tr>
<tr><td style="text-align: right;">                 4</td><td>test_4           </td><td style="text-align: right;">                         3</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td>modified_4000     </td></tr>
<tr><td style="text-align: right;">                 5</td><td>test_5           </td><td style="text-align: right;">                         4</td><td>&ltnp.ndarray<br>shape=(100,)<br>dtype=float64&gt</td><td>modified_5000     </td></tr>
</tbody>
</table>
    <small>(Showing first 5 rows)</small>
</div>


