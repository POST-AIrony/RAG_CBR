import clickhouse_connect
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from server.RAG import pipeline
from server.config import HOST, PORT, TABLE_NAME, MODEL_EMB_NAME, MODEL_CHAT_NAME
from typing import List, Dict

client = clickhouse_connect.get_client(host=HOST, port=PORT)


async def lifespan(router: FastAPI):
    tokenizer, model = pipeline.load_models(MODEL_EMB_NAME)
    chatbot = pipeline.load_chatbot(MODEL_CHAT_NAME)
    router.state.tokenizer = tokenizer
    router.state.model = model
    router.state.chatbot = chatbot
    print("ML model loaded")
    yield
    print("ML model unloaded")


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="server/static"), name="static")


@app.get("/ping")
async def ping():
    return {"message": "pong!"}


@app.post("/chat-bot")
async def new_message(chat: List[Dict[str, str]], k: int = 10):
    if len(chat) == 1:
        search = pipeline.search_results(
            client,
            TABLE_NAME,
            pipeline.txt2embeddings(
                chat[0]["content"], app.state.model, app.state.tokenizer
            )[0],
            k,
        )
        document = search[0]["text"]
    else:
        search = []
        document = ""
    chatbot_answer = pipeline.generate_answer(app.state.chatbot, chat, document)
    return {
        "message": chatbot_answer,
        "documents": search,
    }


@app.post("/append-row")
async def create_row(name: str, url: str, date: str, num: str, text: str):
    embedding = pipeline.txt2embeddings(text, app.state.model, app.state.tokenizer)[0]
    vectors = ",".join([str(float(i)) for i in embedding])
    values = f"('{name}','{url}','{date}','{num}','{text}','[{vectors}]'),"
    query = f"""INSERT INTO "{TABLE_NAME}"("Name","Url","Date","Number", "Text", "Embedding") VALUES {values}"""
    client.command(query)


@app.get("/", response_class=HTMLResponse)
def index():
    html_content = "hello"
    return HTMLResponse(content=html_content, status_code=200)
