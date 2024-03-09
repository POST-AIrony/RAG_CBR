import clickhouse_connect
import json
from transformers import AutoTokenizer, AutoModel
import torch
import time

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru")

client = clickhouse_connect.get_client(
    host="0e42-81-5-106-50.ngrok-free.app", username="test", password="test", port="80"
)
TABLE_NAME = "TinyLlamaEmb"
client.command(
    f"""CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    Name String,
    Url String,
    Date String,
    Number String,
    Embedding Array(Float32)
) ENGINE = MergeTree PRIMARY KEY tuple();"""
)

with open("server/RAG/data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

for item in data:
    encoded_input = tokenizer([item["title"]], padding=True, truncation=True, max_length=24, return_tensors='pt')

    #Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    #Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    vectors = ",".join([str(float(i)) for i in sentence_embeddings[0]])
    client.command(
        f"""INSERT INTO "{TABLE_NAME}"("Name","Url","Date","Number","Embedding") VALUES('{item["title"]}','{item['link']}','{item['date']}','{item['number']}',[{vectors}]);"""
    )
    time.sleep