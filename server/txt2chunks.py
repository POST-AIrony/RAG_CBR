import json
import time
from config import MODEL_EMB_NAME
from subchunks import SentenceChunker, RecursiveChunker


with open("server/RAG/data_text_true_full.json", "r", encoding="utf-8") as file:
    data = json.load(file)


recursive_splitter = RecursiveChunker(chunk_overlap=75, chunk_size=1024)
sentence_splitter = SentenceChunker(model_name=MODEL_EMB_NAME)
text = []
metadata_recursive = []
metadata_sentence = []
for item in data:
    text.append(item.pop("text"))
    del item["id"]
    item2 = item.copy()
    item["chunk_type"] = "Recursive"
    item2["chunk_type"] = "Sentence"
    metadata_recursive.append(item)
    metadata_sentence.append(item2)

start = time.time()
print("Начинаем разбиение текста")
docs_recursive = recursive_splitter.create_documents(text, metadata_recursive)
chunking_recursive = recursive_splitter.split_documents(docs_recursive)
print(f"Разбиение {len(docs_recursive)} документов при помощи RecursiveChunker на {len(chunking_recursive)} частей заняло {time.time() - start} секунд")
start_sentence = time.time()
docs_sentence = sentence_splitter.create_documents(text, metadata_sentence)
chunking_sentence = sentence_splitter.split_documents(docs_sentence)
print(f"Разбиение {len(docs_sentence)} документов при помощи SentenceChunker на {len(chunking_sentence)} частей заняло {time.time() - start_sentence} секунд")
all_documents = docs_recursive = docs_sentence
all_chunks = chunking_recursive
print(f"Общее количество документов: {len(all_documents)}\nОбщее количество частей: {len(all_chunks)}\nОбщее время: {time.time() - start}")



with open("data_text_chunked_rec.json", "w", encoding="utf-8") as file:
        json.dump([item.to_dict() for item in all_chunks], file, ensure_ascii=False, indent=1)

