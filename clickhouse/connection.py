# client.command(
#     """INSERT INTO "table4"("Name","Url","Date","Number","Embedding") VALUES('alo','alo','alo','alo',[1,3.2]);"""
# )

import time

import clickhouse_connect
from scipy.spatial import distance

# from numpy import dot
# from numpy.linalg import norm

search = []  # search vector here


def search_result(connection, vector: list[float], limit: int = 5):
    start = time.time()
    print(f"Serach for vector: {search}")
    res = []
    fetch = client.query("SELECT * FROM TinyLlamaEmb")

    with fetch.rows_stream as stream:  # Тут есть коменты
        for item in stream:
            vec = item[-1]

            dist = distance.cosine(search, vec)
            # dist = dot(search, vec) / (norm(search) * norm(vec))
            res.append(
                {
                    "name": item[0],
                    "url": item[1],
                    "date": item[2],
                    "num": item[3],
                    "emb": item[4],
                }
            )
    res.sort(reverse=False, key=lambda x: x["emb"])
    print("Work: ", time.time() - start)
    return res[0:limit]


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

print(search_result(client, search, limit=2)[0]["name"])
