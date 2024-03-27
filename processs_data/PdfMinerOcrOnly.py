import json
import re
from pdf2image import convert_from_path
from tesserocr import PyTessBaseAPI
from concurrent.futures import ThreadPoolExecutor


def process_image(image):
    api = PyTessBaseAPI(lang='rus+eng')
    api.SetImage(image)
    text = api.GetUTF8Text()
    api.End()
    return text


def extract_text_by_tesseract(pdf_path: str) -> str:
    """
    OCR текст из pdf файла при помощи pytesseract.
    Parameters:
    - pdf_path: (str)

    Returns:
    - str: Извлеченный текст.

    Examples:
    >>> element = LTComponent(...)
    >>> page_obj = pdf.getPage(0)
    >>> text = crop_convert_and_extract_text(element, page_obj)
    """
    images = convert_from_path(pdf_path)
    with ThreadPoolExecutor() as executor:
        # Многопоточно обрабатываем изображения из PDF
        results = list(executor.map(process_image, images))
    # Объединяем результаты
    return " ".join(results)



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
    Очищает текст от нежелательных символов, слов, и лишних пробелов.

    Parameters:
    - text (str): Текст для очистки.

    Returns:
    - str: Очищенный текст.

    Examples:
    >>> clean_text('Пример    текста, с   лишними пробелами.')
    """
    # Заменяем нежелательные символы на пробелы
    text = re.sub(r"[^А-Яа-яЁё0-9 \s.,!?:;-]", " ", text)

    # Удаляем пробелы перед или после дефиса
    text = re.sub(r"-\s|\s-", "", text)

    # Удаляем пробелы перед знаками препинания
    text = re.sub(r"\s+([.,!?:;-])", r"\1", text)

    # Удаляем повторяющиеся знаки препинания
    text = re.sub(r"([.,!?:;-])([.,!?:;-])+", r"\1", text)
    words_to_remove = [
        "ЦЕНТРАЛЬНЫЙ БАНК",
        "РОССИЙСКОЙ ФЕДЕРАЦИИ",
        "БАНК РОССИИ",
        "www.cbr.ru",
        "495 771-91-00",
        "8 800 300-30-00",
        "Банк России",
        "107016",
        "Москва, ул. Неглинная, 12",
        "Центральный банк",
        "Российской Федерации",
    ]
    for word in words_to_remove:
        text = text.replace(word, "")

    # Заменяем последовательности пробелов на одиночные пробелы
    text = re.sub(r"\s+", " ", text)
    return text


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Извлекает текст из pdf файла.

    Args:
        pdf_path (str): Путь к pdf файлу.

    Returns:
        str: Текст из pdf файла.
    """
    text = extract_text_by_tesseract(pdf_path)
    text = clean_text(text)
    if not is_broken_text(text):
        return text
    else:
        return ""



if __name__ == "__main__":
    import time

    # Чтение данных из файла JSON
    with open("server/RAG/data_text_so_so.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    new_data = []  # Список для хранения успешно обработанных данных
    bad_data = []  # Список для хранения данных с ошибками
    so_so_data = []

    # Итерация по элементам данных
    start_time = time.time()
    for item in data:
        iteration_start_time = time.time()
        try:
            # Извлечение текста из PDF файла
            item["text"] = extract_text_from_pdf(
                f'server/RAG/files/{item.get("id")}.pdf'
            )

            # Если текст успешно извлечен
            if item["text"] != "":
                print(
                    f"Успешно обработан файл {item.get('id')} Время итерации: {(time.time() - iteration_start_time):.2f} сек, Общее время: {time.time()-start_time:.2f} сек",
                    end="\r",
                )
                new_data.append(item)  # Добавляем элемент в список успешных данных
            else:
                print(
                    f"Плохой файл {item.get('id')} Время итерации: {(time.time() - iteration_start_time):.2f} сек, Общее время: {time.time()-start_time:.2f} сек",
                    end="\r",
                )
                so_so_data.append(item)
        except Exception as e:
            item["error"] = str(e)  # Добавляем информацию об ошибке в элемент данных
            bad_data.append(item)  # Добавляем элемент в список данных с ошибками
            print(
                f"Ошибка при обработке файла {item['id']}:{e} Время итерации: {iteration_start_time - start_time:.2f} сек, Общее время: {time.time()-start_time:.2f} сек'"
            )  # Выводим информацию об ошибке
    api.End()
    # Запись успешно обработанных данных в файл JSON
    with open("server/RAG/data_text_so_so_true.json", "w", encoding="utf-8") as file:
        json.dump(new_data, file, ensure_ascii=False, indent=1)

    # Запись данных с ошибками в файл JSON
    with open("server/RAG/data_text_so_so_false.json", "w", encoding="utf-8") as file:
        json.dump(bad_data, file, ensure_ascii=False, indent=1)

    with open("server/RAG/data_text_bad.json", "w", encoding="utf-8") as file:
        json.dump(so_so_data, file, ensure_ascii=False, indent=1)
