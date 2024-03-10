import clickhouse_connect
import json

def create_table(client, table_name: str) -> None:
    client.command(
        f"""CREATE TABLE IF NOT EXISTS {table_name} (
            Name String,
            Url String,
            Date String,
            Number String,
            Text String,
            Embedding Array(Float32)
        ) ENGINE = MergeTree PRIMARY KEY tuple();"""
    )


client = clickhouse_connect.get_client(
    host="dafa-81-5-106-50.ngrok-free.app", port="80"
)

TABLE_NAME = "SbertEmb"

create_table(client, TABLE_NAME)

with open("server/RAG/data_emb.json", "r", encoding="utf-8") as file:
    data = json.load(file)

batch_size = 5
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]  
    values = ""
    for item in batch:
        values += f"('{item['title']}','{item['link']}','{item['date']}','{item['number']}','{item['text']}','[{item['emeddings']}]'),"
    query = f"""INSERT INTO "{TABLE_NAME}"("Name","Url","Date","Number", "Text", "Embedding") VALUES {values}"""
    print("отправка запроса")
    try:
        client.command(query)
    except Exception as e:
        print(f"Ошибка при отправке запроса: {e}")