from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from server import database, schemas

try:
    connection = database.get_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT version();")
    print("Database connected")
except Exception as e:
    print("Dataabase error: ", str(e))


async def lifespan(router: FastAPI):
    router.state.model = object()
    print("ML model loaded")
    yield
    print("ML model unloaded")


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="/server/static"), name="static")


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


async def add_vector():
    connection = database.get_connection()
    cursor = connection.cursor()


async def get_vector():
    connection = database.get_connection()
    cursor = connection.cursor()
