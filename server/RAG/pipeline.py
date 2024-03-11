import clickhouse_connect
import torch
from scipy.spatial import distance
from transformers import AutoModel, AutoTokenizer



TABLE_NAME = "SbertEmb"
MODEL_NAME = "ai-forever/sbert_large_nlu_ru"

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

    return mean_pooling(model_output, encoded_input["attention_mask"])[0]


def load_models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    return tokenizer, model


def get_result(client, embeddings, TABLE_NAME):
    results = search_results(client, TABLE_NAME, embeddings, 10)
    if __name__ == "__main__":
        return results
    return [results]

if __name__ == "__main__":
    client = clickhouse_connect.get_client(
    host="dafa-81-5-106-50.ngrok-free.app", port="80"
    )
    print("Ping:", client.ping())

    tokenizer, model = load_models()
    text_data = "Федеральная антимонопольная служба и Центральный банк Российской Федерации рекомендуют кредитным организациям прозрачно информировать клиентов о бонусных программах, включая условия начисления кешбэка, чтобы избежать ввода потребителей в заблуждение и нарушения законодательства о конкуренции и рекламе."
    print(get_result(client, txt2embeddings(text_data, model, tokenizer), TABLE_NAME))
    print("ML model unloaded")
