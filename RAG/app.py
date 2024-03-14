import clickhouse_connect
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from RAG import utilities
from RAG.config import HOST, MODEL_CHAT_NAME, MODEL_EMB_NAME, PORT, TABLE_NAME
from RAG.schemas import ChatInfo

client = clickhouse_connect.get_client(host=HOST, port=PORT)


async def lifespan(router: FastAPI):
    """Подгрузка данных при старте приложения (подгрузка моделей)"""
    tokenizer, model = utilities.load_models(MODEL_EMB_NAME)
    chatbot = utilities.load_chatbot(MODEL_CHAT_NAME)
    router.state.tokenizer = tokenizer
    router.state.model = model
    router.state.chatbot = chatbot
    print("ML model loaded")
    yield
    # При завершении работы приложения
    del router.state.tokenizer
    del router.state.model
    del router.state.chatbot
    print("ML model unloaded")


app = FastAPI(lifespan=lifespan)


@app.post("/chat-bot")
async def new_message(chat: ChatInfo, k: int = 10):
    """Новое сообщение"""
    if len(chat.messages) == 1:
        search = utilities.search_results(
            client,
            TABLE_NAME,
            utilities.txt2embeddings(
                chat.messages[0].content, app.state.tokenizer, app.state.model
            )[0],
            k,
        )
        document = search[0]["text"]
    else:
        search = []
        document = ""
    chatbot_answer = utilities.generate_answer(
        app.state.chatbot, chat.model_dump().get("messages"), document
    )
    return {
        "message": chatbot_answer,
        "documents": search,
    }


def json2sql(data: dict, table_name: str):

    return [
        f"""INSERT INTO "{table_name}"("Text","Number","Date","Title","FileType","Url","ChunkType", "Embedding") VALUES('{item.get('page_content')}', '{item.get('number')}', '{item.get('date')}', '{item.get('title')}', '{item.get('file_type')}', '{item.get('url')}', '{item.get('chunk_type')}', ({item.get("emeddings")}))"""
        for item in data
    ]


@app.post("/append-row")
async def create_row(
    page_content: str,
    number: str,
    date: str,
    title: str,
    file_type: str,
    url: str,
    chunk_type: str,
    table_name: str,
):
    """Создание записи в базе"""
    embedding = utilities.txt2embeddings(
        page_content, app.state.model, app.state.tokenizer
    )[0]

    client.command(
        f"""INSERT INTO "{table_name}"("Text","Number","Date","Title","FileType","Url","ChunkType", "Embedding") VALUES('{page_content}', '{number}', '{date}', '{title}', '{file_type}', '{url}', '{chunk_type}', ({embedding}))"""
    )


@app.get("/", response_class=HTMLResponse)
def index():
    """Приветственная страница с текстом hello"""
    html_content = "hello"
    return HTMLResponse(content=html_content, status_code=200)
