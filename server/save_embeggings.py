import json

# Открываем исходный JSON файл для чтения
with open("data_text_chunked_embedded.json", "r") as source_file:
    source_data = json.load(source_file)


target_data = []
# Создаем новый JSON файл для записи
with open("test.json", "w") as target_file:
    for i, item in enumerate(source_data):
        
        item["emeddings"] = tuple(map(float,item["emeddings"].split(",")))
        target_data.append(item)
        if i == 3:
            break
    json.dump(target_data, target_file, indent=1, ensure_ascii=False)

