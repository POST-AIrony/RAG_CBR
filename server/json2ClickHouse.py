import clickhouse_connect
import json
import sys

sys.path.append(".")
from server.config import HOST, PORT, TABLE_NAME


def create_table(client, table_name: str) -> None:
    """
    Создает таблицу в ClickHouse, если она не существует.

    Parameters:
    - client: Клиент ClickHouse для выполнения запроса.
    - table_name (str): Имя таблицы для создания.

    Returns:
    - None

    Examples:
    >>> create_table(client, "my_table")
    """
    # Выполняем запрос на создание таблицы
    float32_type = "Float32," * 1024
    client.command(
        f"""CREATE TABLE IF NOT EXISTS {table_name} (
            Name String,
            Url String,
            Date String,
            Number String,
            Text String,
            Embedding Tuple({float32_type})
        ) ENGINE = MergeTree PRIMARY KEY tuple();"""
    )


def append_to_clickhouse(client, table_name: str, data: list[dict]) -> None:
    """
    Добавляет данные в таблицу ClickHouse.

    Parameters:
    - client: Клиент ClickHouse для выполнения запроса.
    - table_name (str): Имя таблицы, в которую добавляются данные.
    - data (list[dict]): Список словарей с данными для добавления.

    Returns:
    - None

    Examples:
    >>> data = [
    >>>     {"title": "Заголовок", "link": "http://example.com", "date": "2022-01-01", "number": "123", "text": "Текст", "emeddings": [1.0, 2.0, 3.0]},
    >>>     {"title": "Заголовок2", "link": "http://example2.com", "date": "2022-01-02", "number": "124", "text": "Текст2", "emeddings": [4.0, 5.0, 6.0]},
    >>> ]
    >>> append_to_clickhouse(client, "my_table", data)
    """
    # Формируем строку значений для запроса INSERT
    values = ""
    for item in data:
        values += f"('{item['title']}','{item['link']}','{item['date']}','{item['number']}','{item['text']}','[{item['emeddings']}]'),"

    # Формируем запрос INSERT
    query = f"""INSERT INTO "{table_name}"("Name","Url","Date","Number", "Text", "Embedding") VALUES {values[:-1]}"""

    # Выполняем запрос
    client.command(query)

if __name__ == "__main__":
    client = clickhouse_connect.get_client(host=HOST, port=PORT)
    create_table(client, TABLE_NAME)
    exit()
    with open("server/RAG/data_emb_true.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    batch_size = 2
    for i in range(0, len(data), batch_size):
        append_to_clickhouse(client, TABLE_NAME, data[i : i + batch_size])
