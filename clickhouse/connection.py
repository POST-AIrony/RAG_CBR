import clickhouse_connect

client = clickhouse_connect.get_client(
    host="127.0.0.1", username="test", password="test"
)

# client.command("SET allow_experimental_annoy_index = 1")
client.command(
    """CREATE TABLE table1 (Name String, embedding Array(Float32)) ENGINE = MergeTree ORDER BY Name;"""
)
