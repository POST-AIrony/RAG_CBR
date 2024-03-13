import json

with open("server/RAG/data_text_true_recovered.json", "r", encoding="utf-8") as file:
        data1 = json.load(file)

with open("server/RAG/data_text_true2.json", "r", encoding="utf-8") as file:
        data2 = json.load(file)

with open("server/RAG/data_text_true.json", "r", encoding="utf-8") as file:
        data3 = json.load(file)
        
new_data = []

for item in data1:
    if "file_type" not in item:
        item["file_type"] = "pdf"
    else:
        item["file_type"] = "text_from_site"
    _ = item.pop("file_name", "")
    item['url'] = item.pop('link')
    new_data.append(item)

for item in data2:
    item["file_type"] = "pdf"
    item['url'] = item.pop('utl')
    new_data.append(item)

for item in data3:
    item["file_type"] = "pdf"
    item['url'] = item.pop('link')
    new_data.append(item)

with open("server/RAG/data_text_true_full.json", "w", encoding="utf-8") as file:
        json.dump(new_data, file, ensure_ascii=False, indent=1)