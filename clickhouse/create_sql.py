def generate_sql_row(
    name: str,
    url: str,
    date: str,
    number: str,
    text: str,
    vector: list[float],
    table_name: str,
):
    """функция для генерирования sql запроса"""
    numbers = ",".join([str(i) for i in vector])
    return f"""INSERT INTO "{table_name}"("Name","Date","Text","Url","Number","Embedding") VALUES('{name}','{date}','{text}','{url}','{number}',({numbers}));"""


sql = generate_sql_row("b", "a", "a", "a", "a", [i for i in range(1024)], "table7")
print(sql)
