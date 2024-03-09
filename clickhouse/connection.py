import time

import clickhouse_connect
from scipy.spatial import distance

start = time.time()
client = clickhouse_connect.get_client(
    host="0e42-81-5-106-50.ngrok-free.app", username="test", password="test", port="80"
)
print("Ping:", client.ping())

client.command(
    """CREATE TABLE IF NOT EXISTS table4 (
    Name String,
    Url String,
    Date String,
    Number String,
    Embedding Array(Float64)
) ENGINE = MergeTree PRIMARY KEY tuple();"""
)


# client.command(
#     """INSERT INTO "table4"("Name","Url","Date","Number","Embedding") VALUES('alo','alo','alo','alo',[1,3.2]);"""
# )
search = [1, 2]
print(f"Serach for vector: {search}")
res = []
fetch = client.query("SELECT * FROM table4")

with fetch.rows_stream as stream:  # Тут есть коменты
    for item in stream:
        item = item[-1]
        dist = distance.cosine(search, item)
        res.append(dist)

res.sort()
print(res)
print("Work: ", time.time() - start)
