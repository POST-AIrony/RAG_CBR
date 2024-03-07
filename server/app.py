from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from server import database, schemas

try:
    connection = database.get_connection()
    cursor = connection.cursor()
    cursor.execute(
        """CREATE TABLE documents1 (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    vector VECTOR NOT NULL
);"""
    )
    connection.commit()
    cursor.execute(
        "CREATE INDEX vector_index ON documents USING ivfflat (vector) WITH (dim=768);"
    )
    connection.commit()
    print("Database connected")
except Exception as e:
    print("Dataabase error: ", str(e))


async def lifespan(router: FastAPI):
    router.state.model = object()
    print("ML model loaded")
    yield
    print("ML model unloaded")


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="server/static"), name="static")


@app.get("/ping")
async def ping():
    return {"message": "pong!"}


@app.post("/message")
async def new_message(messages: schemas.ChatInfo):
    return  # res =  ml(app.state.model, messages)


@app.get("/", response_class=HTMLResponse)
def index():
    html_content = "hello"
    return HTMLResponse(content=html_content, status_code=200)


async def add_vector(text, vector: list[int | float]):
    connection = database.get_connection()
    cursor = connection.cursor()
    vectors = ",".join([str(float(i)) for i in vector])
    id = 0
    sql = "SELECT * FROM documents1"
    cursor.execute(sql)
    for row in cursor.fetchall():
        id += 1
    id += 1
    sql = f"""INSERT INTO documents1 (id, text, vector) VALUES ({id}, '{text}', '[{vectors}]')"""
    cursor.execute(sql)
    connection.commit()


from psycopg2 import sql


async def get_vectors(your_query_vector, your_k):
    connection = database.get_connection()
    cursor = connection.cursor()

    vectors = ",".join([str(float(i)) for i in your_query_vector])
    query = f"SELECT id, text, vector, vector <-> '[{vectors}]' AS distance FROM documents1 ORDER BY vector <-> '[{vectors}]' LIMIT {your_k};"
    cursor.execute(query)
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    connection.commit()
    connection.close()


@app.get("/test")
async def test():
    await get_vectors([i for i in range(1, 769)], 5)
    await add_vector("test add", [i for i in range(1, 769)])
    return
