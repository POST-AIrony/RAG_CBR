import json
import re
import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Извлекает текст из pdf файла.

    Args:
        pdf_path (str): Путь к pdf файлу.

    Returns:
        str: Текст из pdf файла.
    """
    print(f"Извлекаем текст из pdf файла: {pdf_path}")
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    text = re.sub("\s+", " ", text)
    text = re.sub(r"\.+", " ", text)
    text = re.sub(r"\(cid\:\d+\)", "", text)
    start_index = text.find("(800) 300-30-00")
    if start_index != -1:
        text = text[start_index + len("(800) 300-30-00") :]
    return text

if __name__ == "__main__":
    # Получаем список файлов для обработки
    with open("server/RAG/data.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    new_data = []
    # Проходимся по списку файлов и извлекаем текст из pdf файлов
    for item in data:
        try:

            item["text"] = extract_text_from_pdf(f'server/RAG/files/{item["id"]}.pdf')
            new_data.append(item)
        except Exception as e:
            print(f"Ошибка при обработке файла {item['id']}.pdf: {e}")
    
    with open("server/RAG/data_text2.json", "w", encoding="utf-8") as file:
        json.dump(new_data, file, ensure_ascii=False, indent=4)
