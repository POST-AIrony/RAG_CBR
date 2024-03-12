import json
import re
import os
import pdfplumber


def count_chars(text: str) -> tuple:
    """
    Подсчитывает количество русских и английских символов в тексте.

    Parameters:
    - text (str): Текст для анализа.

    Returns:
    - tuple: Кортеж с количеством русских и английских символов.

    Examples:
    >>> russian_chars, english_chars = count_chars('Пример text')
    """
    # Подсчитываем количество русских символов в тексте
    russian_chars = sum(1 for c in text if "а" <= c <= "я" or "А" <= c <= "Я")
    
    # Подсчитываем количество английских символов в тексте
    english_chars = sum(1 for c in text if "a" <= c <= "z" or "A" <= c <= "Z")
    
    return russian_chars, english_chars

def is_broken_text(text: str, threshold: float = 0.2, min_length: int = 300) -> bool:
    """
    Проверяет, является ли текст "сломанным" (недостаточно длинным или слишком много пробелов).

    Parameters:
    - text (str): Текст для проверки.
    - threshold (float): Пороговое значение отношения пробелов к общему количеству символов.
    - min_length (int): Минимальная длина текста для считывания как корректного.

    Returns:
    - bool: True, если текст "сломанный", иначе False.

    Examples:
    >>> is_broken_text('Пример текста', threshold=0.2, min_length=10)
    """
    # Проверяем, достаточно ли длинный текст
    if len(text) < min_length:
        return True

    # Подсчитываем количество пробелов в тексте
    space_count = sum(1 for c in text if c.isspace())
    total_count = len(text)
    
    # Вычисляем отношение пробелов к общему количеству символов
    space_ratio = space_count / total_count if total_count > 0 else 0

    # Проверяем, превышает ли отношение пробелов пороговое значение
    if space_ratio > threshold:
        return True
    return False


def clean_text(text: str) -> str:
    """
    Очищает текст от нежелательных символов и лишних пробелов.

    Parameters:
    - text (str): Текст для очистки.

    Returns:
    - str: Очищенный текст.

    Examples:
    >>> clean_text('Пример    текста, с   лишними пробелами.')
    """
    # Заменяем нежелательные символы на пробелы
    text = re.sub(r"[^А-Яа-яЁё \s.,!?:;-]", " ", text)
    
    # Заменяем последовательности пробелов на одиночные пробелы
    text = re.sub(r"\s+", " ", text)
    
    # Удаляем пробелы перед или после дефиса
    text = re.sub(r"-\s|\s-", "", text)
    
    return text


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Извлекает текст из pdf файла.

    Args:
        pdf_path (str): Путь к pdf файлу.

    Returns:
        str: Текст из pdf файла.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    text = clean_text(text)
    print(f"Успешно обработан файл {os.path.splitext(os.path.basename(pdf_path))[0]}")
    if not is_broken_text(text):
        return text
    else:
        return ""

if __name__ == "__main__":
    # Чтение данных из файла JSON
    with open("server/RAG/data.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    new_data = []  # Список для хранения успешно обработанных данных
    bad_data = []  # Список для хранения данных с ошибками

    # Итерация по элементам данных
    for item in data:
        try:
            # Извлечение текста из PDF файла
            item["text"] = extract_text_from_pdf(
                f'server/RAG/files/{item.get("id")}.pdf'
            )
            # Если текст успешно извлечен
            if item["text"] != "":
                new_data.append(item)  # Добавляем элемент в список успешных данных
        except Exception as e:
            item["error"] = str(e)  # Добавляем информацию об ошибке в элемент данных
            bad_data.append(item)  # Добавляем элемент в список данных с ошибками
            print(f"Ошибка при обработке файла {item['id']}\n{e}")  # Выводим информацию об ошибке

    # Запись успешно обработанных данных в файл JSON
    with open("server/RAG/data_text_true.json", "w", encoding="utf-8") as file:
        json.dump(new_data, file, ensure_ascii=False, indent=4)

    # Запись данных с ошибками в файл JSON
    with open("server/RAG/data_text_false.json", "w", encoding="utf-8") as file:
        json.dump(bad_data, file, ensure_ascii=False, indent=4)