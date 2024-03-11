import clickhouse_connect
import torch
from scipy.spatial import distance
from transformers import AutoModel, AutoTokenizer, pipeline, Conversation
from server.config import (
    TABLE_NAME,
    HOST,
    PORT,
    MODEL_EMB_NAME,
    MODEL_CHAT_NAME,
    SYSTEM_PROMPT,
)


def search_results(connection, table_name: str, vector: list[float], limit: int = 5):
    res = []
    with connection.query(f"SELECT * FROM {table_name}").rows_stream as stream:
        for item in stream:
            name, url, date, num, text, emb = item

            dist = distance.cosine(vector, emb)
            res.append(
                {
                    "name": name,
                    "url": url,
                    "date": date,
                    "num": num,
                    "text": text,
                    "dist": dist,
                }
            )
    res.sort(key=lambda x: x["dist"])
    return res[:limit]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def txt2embeddings(text: str, tokenizer, model, device="cpu"):
    encoded_input = tokenizer(
        [text],
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**encoded_input)

    return mean_pooling(model_output, encoded_input["attention_mask"])


def load_chatbot(model):
    chatbot = pipeline(
        model=model,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cuda",
        task="conversational",
    )
    return chatbot


def generate_answer(chatbot, chat, document):
    conversation = Conversation()
    conversation.add_message({"role": "system", "content": SYSTEM_PROMPT})
    if document:
        document_template = """
        CONTEXT:
        {document}

        Отвечай только на русском языке.
        ВОПРОС:
        """
        conversation.add_message({"role": "user", "content": document_template})

    for message in chat:
        conversation.add_message(
            {"role": message["role"], "content": message["content"]}
        )

    conversation = chatbot(conversation, max_new_tokens=512)

    return conversation[-1]


def load_models(model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)
    return tokenizer, model


if __name__ == "__main__":
    client = clickhouse_connect.get_client(host=HOST, port=PORT)
    print("Ping:", client.ping())

    tokenizer, model = load_models(MODEL_EMB_NAME)
    chatbot = load_chatbot(MODEL_CHAT_NAME)
    chat = [{
        "role": "user",
        "content": "Федеральная антимонопольная служба и Центральный банк Российской Федерации рекомендуют кредитным организациям прозрачно информировать клиентов о бонусных программах, включая условия начисления кешбэка, чтобы избежать ввода потребителей в заблуждение и нарушения законодательства о конкуренции и рекламе.",
    }]
    embeddings = (pipeline.txt2embeddings(request, model, tokenizer)[0],)
    search = pipeline.search_results(client, TABLE_NAME, embeddings, 10)
    document = search[0]["text"]
    chatbot_answer = generate_answer(chatbot, chat, document)
    print("ML model unloaded")
