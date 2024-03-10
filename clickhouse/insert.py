import clickhouse_connect

client = clickhouse_connect.get_client(
    host="dafa-81-5-106-50.ngrok-free.app", port="80"
)
print("Ping:", client.ping())
client.command(
    """CREATE TABLE IF NOT EXISTS table5 (
    Name String,
    Url String,
    Date String,
    Number String,
    Text String,
    Embedding Array(Float64)
) ENGINE = MergeTree PRIMARY KEY tuple();"""
)
client.command(
    """INSERT INTO "table5"("Name","Url","Date","Number", "Text", "Embedding") VALUES
    ('alo','alo','alo','alo','alo',[1,3.2]),
    ('alo','alo','alo','alo','alo',[1,3.2]),
    ('alo','alo','alo','alo','alo',[1,3.2]),
    ('alo','alo','alo','alo','alo',[1,3.2]),
    ('alo','alo','alo','alo','alo',[1,3.2]),
    ('alo','alo','alo','alo','alo',[1,3.2]),
"""
)
"""INSERT INTO "table4"("Name","Url","Date","Number","Embedding") VALUES
    ('alo','alo','alo','alo',[1,3.2]),
    ('alo','alo','alo','alo',[1,3.2]),
    ('alo','alo','alo','alo',[1,3.2]),
    ('alo','alo','alo','alo',[1,3.2]),
    ('alo','alo','alo','alo',[1,3.2])
"""
"""INSERT INTO "table4"("Name","Url","Date","Number","Embedding") VALUES('alo','alo','alo','alo',[1,3.2]);"""
