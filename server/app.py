import clickhouse_connect
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from server.RAG import pipeline

client = clickhouse_connect.get_client(
    host="dafa-81-5-106-50.ngrok-free.app", port="80"
)


async def lifespan(router: FastAPI):
    tokenizer, model = pipeline.load_models()
    router.state.tokenizer = tokenizer
    router.state.model = model
    print("ML model loaded")
    yield
    print("ML model unloaded")


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="server/static"), name="static")


@app.get("/ping")
async def ping():
    return {"message": "pong!"}


@app.post("/message")
async def new_message(msg: str):
    return pipeline.get_result(
        client,
        pipeline.get_embembeddings(msg, app.state.model, app.state.tokenizer),
        TABLE_NAME=pipeline.TABLE_NAME,
    )


@app.post("/create/row")
async def create_row(name: str, url: str, date: str, num: str, text: str):
    embembedding = pipeline.get_embembeddings(
        text, app.state.model, app.state.tokenizer
    )
    vectors = ",".join([str(float(i)) for i in embembedding])
    # TODO: add to DataBase


@app.get("/", response_class=HTMLResponse)
def index():
    html_content = "hello"
    return HTMLResponse(content=html_content, status_code=200)
