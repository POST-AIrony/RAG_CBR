import json
import time
import torch
from utilities import mean_pooling, load_models
from config import MODEL_EMB_NAME
import random
with open("continue.json", "r", encoding="utf-8") as file:
    data = json.load(file)
random.shuffle(data)

text_data = [item["page_content"] for item in data]
tokenizer, model = load_models(MODEL_EMB_NAME, device="cuda")
vectors = []
N = 50
batch_size = 20
len_data = len(text_data)
start = time.time()
batch_times = []
try:
    for i in range(0, len_data, batch_size):
        batch_start = time.time()
        
        batch = text_data[i : i + batch_size]
        encoded_input = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
        )
        encoded_input = {k: v.to('cuda') for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        vectors.extend([",".join([str(float(i)) for i in sentence]) for sentence in sentence_embeddings])
        torch.cuda.empty_cache()
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        if len(batch_times) > N:
            batch_times = batch_times[-N:]
        avg_time = sum(batch_times) / len(batch_times)  
        time_left = avg_time * (len_data - i) / batch_size
        processed = i + batch_size
        left = len_data - processed
        
        print(f"Осталось данных: {left}/{len_data}, Этот батч обработался за {batch_time:.2f} секунд, Примерно осталось времени: {time_left:.2f} секунд", end="\r")
except Exception as e:
    print(e)
    for i, vector in enumerate(vectors):
        data[i]["emeddings"] = vector


    with open("data_text_chunked_embedded.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=1)
    
    exit()


for i, vector in enumerate(vectors):
    data[i]["emeddings"] = vector


with open("data_text_chunked_embedded.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=1)
