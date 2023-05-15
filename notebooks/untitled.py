import json
import pandas as pd
import duckdb

documents = []
with open("./data/pokemon.json", "r") as f:
    for line in f:
        documents.append(json.loads(line))

df = pd.DataFrame(documents)

con = duckdb.connect()
rel = con.from_df(df)

rel.filter("name == Abra").df()