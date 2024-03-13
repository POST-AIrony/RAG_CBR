import io
import json
import re
from typing import Any, List, Tuple, Union

import pdfplumber
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTComponent, LTFigure, LTTextContainer


def text_extraction(element: LTTextContainer) -> Tuple[str, list]:
    """
    Извлекает текст из элемента на странице PDF.

    Parameters:
    - element (LTTextContainer): Элемент страницы PDF, содержащий текст.

    Returns:
    - tuple: текст элемента
    Examples:
    >>> element = LTTextContainer(...)
    >>> text = text_extraction(element)
    """
    # Извлекаем текст элемента
    text = element.get_text()
    return text


def extract_table(pdf, page_num: int, table_num: int) -> List[List[str]]:
    """
    Извлекает таблицу из указанной страницы PDF.

    Parameters:
    - pdf_path (str): Путь к PDF файлу.
    - page_num (int): Номер страницы, с которой нужно извлечь таблицу (начиная с 0).
    - table_num (int): Номер таблицы на странице (начиная с 0).

    Returns:
    - list: Список списков, представляющих извлеченную таблицу.

    Examples:
    >>> table = extract_table('example.pdf', 0, 0)
    """
    table_page = pdf.pages[page_num]
    tables = table_page.extract_tables()

    # Проверяем, что указанный номер таблицы существует на странице
    if table_num < len(tables):
        table = tables[table_num]
        return table
    else:
        raise IndexError(
            f"Таблицы с номером {table_num} не существует на странице {page_num}."
        )


def convert_table_to_string(table: List[List[str]]) -> str:
    """
    Преобразует таблицу (список списков строк) в строку с разделителями.

    Parameters:
    - table (List[List[str]]): Таблица для преобразования.

    Returns:
    - str: Строка с разделителями для каждой ячейки таблицы.

    Examples:
    >>> table = [["A", "B", "C"], ["1", "2", "3"]]
    >>> table_string = convert_table_to_string(table)
    >>> print(table_string)
    """
    text_rows = []

    for row in table:

        text_rows.append(" ".join(str(cell) for cell in row))

    return "\n".join(text_rows)


def is_element_inside_any_table(
    element: LTComponent, page: LTComponent, tables: List[LTComponent]
) -> bool:
    """
    Проверяет, находится ли элемент внутри какой-либо из таблиц на странице.

    Parameters:
    - element (LTComponent): Элемент, который необходимо проверить.
    - page (LTComponent): Страница PDF, на которой находится элемент.
    - tables (List[LTComponent]): Список таблиц на странице.

    Returns:
    - bool: True, если элемент находится внутри хотя бы одной таблицы, иначе False.

    Examples:
    >>> element = LTComponent(bbox=(10, 10, 20, 20))
    >>> page = LTComponent(bbox=(0, 0, 100, 100))
    >>> tables = [LTComponent(bbox=(5, 5, 15, 15)), LTComponent(bbox=(25, 25, 35, 35))]
    >>> is_inside_table = is_element_inside_any_table(element, page, tables)
    >>> is_inside_table  # Вывод: True или False
    """
    x0, y0up, x1, y1up = element.bbox
    y0 = page.bbox[3] - y1up
    y1 = page.bbox[3] - y0up
    for table in tables:
        tx0, ty0, tx1, ty1 = table.bbox
        # Проверяем, находится ли элемент внутри границ таблицы
        if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
            return True
    return False


def find_table_for_element(
    element: LTComponent, page: LTComponent, tables: List[LTComponent]
) -> Union[int, None]:
    """
    Находит индекс таблицы, которая содержит указанный элемент на странице.

    Parameters:
    - element (LTComponent): Элемент, для которого нужно найти таблицу.
    - page (LTComponent): Страница, на которой находится элемент.
    - tables (List[LTComponent]): Список таблиц на странице.

    Returns:
    - int or None: Индекс таблицы, если элемент находится в таблице, иначе None.

    Examples:
    >>> element = LTComponent(...)
    >>> page = LTComponent(...)
    >>> tables = [LTComponent(...), LTComponent(...)]
    >>> table_index = find_table_for_element(element, page, tables)
    """
    # Получаем координаты элемента
    x0, y0up, x1, y1up = element.bbox
    y0 = page.bbox[3] - y1up
    y1 = page.bbox[3] - y0up

    # Перебираем все таблицы на странице
    for i, table in enumerate(tables):
        tx0, ty0, tx1, ty1 = table.bbox
        # Проверяем, содержится ли элемент в текущей таблице
        if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
            return i

    # Если элемент не найден в таблице, возвращаем None
    return None


def crop_convert_and_extract_text(
    element: LTComponent, pdf_path: str, page_num: int
) -> str:
    """
    Обрезает изображение в PDF файле до указанного элемента, конвертирует его в изображение PNG и извлекает текст.

    Parameters:
    - element (LTComponent): Элемент, который определяет область обрезки.
    - page_obj (Any): Объект страницы PDF, содержащий изображение.

    Returns:
    - str: Извлеченный текст.

    Examples:
    >>> element = LTComponent(...)
    >>> page_obj = pdf.getPage(0)
    >>> text = crop_convert_and_extract_text(element, page_obj)
    """
    # Получаем координаты области для обрезки изображения
    image_left, image_top, image_right, image_bottom = (
        element.x0,
        element.y0,
        element.x1,
        element.y1,
    )

    images = convert_from_path(
        pdf_path, first_page=page_num + 1, last_page=page_num + 1
    )
    image = images[0]
    cropped_image = image.crop((image_left, image_top, image_right, image_bottom))

    # Извлекаем текст с изображения
    text = pytesseract.image_to_string(cropped_image, lang="rus+eng")

    return text


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
    Извлекает текст из PDF файла, включая текст изображений и таблиц.

    Parameters:
    - pdf_path (str): Путь к PDF файлу.

    Returns:
    - str: Извлеченный текст.

    Examples:
    >>> extract_text_from_pdf('example.pdf')
    """
    pdf = pdfplumber.open(pdf_path)
    pages_content = []
    for pagenum, page in enumerate(
        extract_pages(pdf_path)
    ):  # Итерация по страницам PDF
        text_from_tables = []  # Текст таблиц на странице
        table_in_page = -1  # Переменная для отслеживания таблиц на странице
        page_tables = pdf.pages[pagenum]  # Таблицы на странице
        tables = page_tables.find_tables()  # Находим таблицы на странице
        if len(tables) != 0:
            table_in_page = 0  # Если есть таблицы, начинаем с первой

        # Итерация по таблицам на странице
        for table_num in range(len(tables)):
            table = extract_table(pdf, pagenum, table_num)  # Извлекаем таблицу
            table_string = convert_table_to_string(
                table
            )  # Конвертируем таблицу в строку
            text_from_tables.append(table_string)  # Добавляем строку таблицы

        # Получаем список элементов страницы и сортируем их по y1
        page_elements = [(element.y1, element) for element in page._objs]
        page_elements.sort(key=lambda a: a[0], reverse=True)

        # Итерация по элементам страницы
        for _, component in enumerate(page_elements):
            element = component[1]

            # Проверка, находится ли элемент внутри таблицы
            if table_in_page == -1:
                pass
            else:
                if is_element_inside_any_table(element, page, tables):
                    table_found = find_table_for_element(element, page, tables)
                    if table_found == table_in_page and table_found is not None:
                        pages_content.append(text_from_tables[table_in_page])
                        table_in_page += 1
                    continue

            # Если элемент не внутри таблицы
            if not is_element_inside_any_table(element, page, tables):
                if isinstance(element, LTTextContainer):
                    # Извлекаем текст из контейнера текста
                    line_text = text_extraction(element)
                    pages_content.append(line_text)

                # if isinstance(element, LTFigure):
                #     # Обрезаем изображение и извлекаем текст
                #     image_text = crop_convert_and_extract_text(element, pdf_path, pagenum)
                #     page_content.append(image_text)

    # Объединяем текст со всех страниц и очищаем его
    text = "".join("".join(page_content) for page_content in pages_content)
    text = clean_text(text)
    if not is_broken_text(text):  # Проверяем текст на целостность
        return text
    else:
        return ""


if __name__ == "__main__":
    import time

    # Чтение данных из файла JSON
    with open("server/RAG/data_recovered.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    new_data = []  # Список для хранения успешно обработанных данных
    bad_data = []  # Список для хранения данных с ошибками
    so_so_data = []

    # Итерация по элементам данных
    start_time = time.time()
    for item in data:
        iteration_start_time = time.time()
        try:
            if "text" in item:
                item["file_type"] = "text_from_site"
                item["text"] = clean_text(item["text"])
                if is_broken_text(item["text"]):
                    item["text"] = ""

            else:
                # Извлечение текста из PDF файла
                item["text"] = extract_text_from_pdf(
                    f'server/RAG/recovered_data/{item.get("file_name")}'
                )
            del item["error"]  # Удаляем поле error

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

    # Запись успешно обработанных данных в файл JSON
    with open(
        "server/RAG/data_text_true_recovered.json", "w", encoding="utf-8"
    ) as file:
        json.dump(new_data, file, ensure_ascii=False, indent=4)

    # Запись данных с ошибками в файл JSON
    with open(
        "server/RAG/data_text_false_recovered.json", "w", encoding="utf-8"
    ) as file:
        json.dump(bad_data, file, ensure_ascii=False, indent=4)

    with open(
        "server/RAG/data_text_so_so_recovered.json", "w", encoding="utf-8"
    ) as file:
        json.dump(so_so_data, file, ensure_ascii=False, indent=4)
