import clickhouse_connect

import json
from transformers import AutoTokenizer, AutoModel
import torch


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def create_table(client, table_name: str) -> None:
    client.command(
        f"""CREATE TABLE IF NOT EXISTS {table_name} (
            Name String,
            Url String,
            Date String,
            Number String,
            Text String,
            Embedding Array(Float32)
        ) ENGINE = MergeTree PRIMARY KEY tuple();"""
    )


client = clickhouse_connect.get_client(
    host="dafa-81-5-106-50.ngrok-free.app", port="80"
)

TABLE_NAME = "SbertEmb"
MODEL_NAME = "ai-forever/sbert_large_nlu_ru"

create_table(client, TABLE_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, device_map="cuda", torch_dtype="auto")
model = AutoModel.from_pretrained(MODEL_NAME, device_map="cuda", torch_dtype="auto")

with open("server/RAG/data_text.json", "r", encoding="utf-8") as file:
    data = json.load(file)
new_data = []
text_data = [item["text"] for item in data]

encoded_input = tokenizer(
    text_data,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
encoded_input = {k: v.to('cuda') for k, v in encoded_input.items()} 

with torch.no_grad():
    model_output = model(**encoded_input)

sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
vectors = [
    ",".join([str(float(i)) for i in sentence]) for sentence in sentence_embeddings
]
for i, vector in enumerate(vectors):
    new_data[i]["emeddings"] = vector

VALUES = ""
for item in new_data:
    VALUES += f"('{item['title']}','{item['link']}','{item['date']}','{item['number']}','{item['text']}','[{item['emeddings']}]'),"
client.command(
    f"""INSERT INTO "{TABLE_NAME}"("Name","Url","Date","Number", "Text", "Embedding") VALUES
    {VALUES}
    """
)
