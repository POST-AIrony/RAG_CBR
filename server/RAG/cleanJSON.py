import json
def is_broken_text(text, threshold=0.2, min_length=300):
    if len(text) < min_length:
        return True
    space_count = sum(1 for c in text if c.isspace())
    total_count = len(text)
    space_ratio = space_count / total_count if total_count > 0 else 0
    russian_chars, english_chars = count_chars(text)
    if space_ratio > threshold or russian_chars <= english_chars:
        return True
    return False

def count_chars(text):
    russian_chars = sum(1 for c in text if 'а' <= c <= 'я' or 'А' <= c <= 'Я')
    english_chars = sum(1 for c in text if 'a' <= c <= 'z' or 'A' <= c <= 'Z')
    return russian_chars, english_chars

def clean_json_file(input_file_path, output_file_path):
    clean_data = []
    with open(input_file_path, 'r', encoding='utf-8') as input_file, \
         open(output_file_path, 'w', encoding='utf-8') as output_file:
            lines = json.load(input_file)
            for data in lines:
                if not is_broken_text(data['text']):
                    clean_data.append(data)
            json.dump(clean_data, output_file, ensure_ascii=False, indent=4)

# Пример использования
if __name__ == '__main__':
    input_file_path = 'server/RAG/data_text_clean_true.json'
    output_file_path = 'server/RAG/data_text_clean_true12345678.json'
    clean_json_file(input_file_path, output_file_path)