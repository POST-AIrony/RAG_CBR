import json
import re
import PyPDF2
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure, LTComponent
import pdfplumber
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
import os
from typing import Tuple, List, Dict, Union, Any


def text_extraction(element: LTTextContainer) -> Tuple[str, list]:
    """
    Извлекает текст и форматирование шрифта из элемента на странице PDF.

    Parameters:
    - element (LTTextContainer): Элемент страницы PDF, содержащий текст.

    Returns:
    - tuple: Кортеж с текстом элемента и уникальными форматами шрифта в нем.

    Examples:
    >>> element = LTTextContainer(...)
    >>> text, font_formats = text_extraction(element)
    """
    # Извлекаем текст элемента
    text = element.get_text()

    # Инициализируем список для хранения форматов шрифта на каждой строке
    font_formats = []

    # Перебираем каждую строку в элементе
    for text_line in element:
        # Проверяем, что строка текстовый контейнер
        if isinstance(text_line, LTTextContainer):
            # Перебираем каждый символ в строке
            for character in text_line:
                # Проверяем, что символ - это текстовый символ
                if isinstance(character, LTChar):
                    # Добавляем формат шрифта символа в список
                    font_formats.append(character.fontname)
                    font_formats.append(character.size)

    # Получаем уникальные форматы шрифта на каждой строке
    unique_font_formats = list(set(font_formats))

    return (text, unique_font_formats)


def extract_table(pdf_path: str, page_num: int, table_num: int) -> List[List[str]]:
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
    pdf = pdfplumber.open(pdf_path)
    table_page = pdf.pages[page_num]
    tables = table_page.extract_tables()

    # Проверяем, что указанный номер таблицы существует на странице
    if table_num < len(tables):
        table = tables[table_num]
        pdf.close()
        return table
    else:
        pdf.close()
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
    |A|B|C|
    |1|2|3|
    """
    table_string = ""
    for row_num in range(len(table)):
        row = table[row_num]
        cleaned_row = [
            (
                item.replace("\n", " ")
                if item is not None and "\n" in item
                else "None" if item is None else item
            )
            for item in row
        ]
        # Объединяем элементы строки таблицы с разделителями
        table_string += "|" + "|".join(cleaned_row) + "|" + "\n"
    # Удаляем последний символ переноса строки
    table_string = table_string[:-1]
    return table_string


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


def crop_image(element: LTComponent, page_obj: Any) -> None:
    """
    Обрезает изображение в PDF файле до указанного элемента.

    Parameters:
    - element (LTComponent): Элемент, который определяет область обрезки.
    - page_obj (Any): Объект страницы PDF, содержащий изображение.

    Returns:
    - None

    Examples:
    >>> element = LTComponent(...)
    >>> page_obj = pdf.getPage(0)
    >>> crop_image(element, page_obj)
    """
    # Получаем координаты области для обрезки изображения
    image_left, image_top, image_right, image_bottom = (
        element.x0,
        element.y0,
        element.x1,
        element.y1,
    )

    # Устанавливаем новую область медиа-коробки (media box) для обрезки изображения
    page_obj.mediabox.lower_left = (image_left, image_bottom)
    page_obj.mediabox.upper_right = (image_right, image_top)

    # Создаем объект для записи обрезанного изображения в новый PDF файл
    cropped_pdf_writer = PyPDF2.PdfWriter()
    cropped_pdf_writer.add_page(page_obj)

    # Записываем обрезанное изображение в файл "cropped_image.pdf"
    with open("cropped_image.pdf", "wb") as cropped_pdf_file:
        cropped_pdf_writer.write(cropped_pdf_file)


def convert_pdf_to_image(input_file: str) -> None:
    """
    Конвертирует первую страницу PDF файла в изображение PNG.

    Parameters:
    - input_file (str): Путь к входному PDF файлу.

    Returns:
    - None

    Examples:
    >>> convert_pdf_to_image('example.pdf')
    """
    # Конвертируем PDF файл в список изображений
    images = convert_from_path(input_file)

    # Берем первое изображение (первую страницу PDF)
    image = images[0]

    # Сохраняем изображение в формате PNG
    output_file = "PDF_image.png"
    image.save(output_file, "PNG")


def extract_text_from_image(image_path: str) -> str:
    """
    Извлекает текст с изображения, используя pytesseract.

    Parameters:
    - image_path (str): Путь к изображению.

    Returns:
    - str: Извлеченный текст.

    Examples:
    >>> text = extract_text_from_image('image.png')
    """
    # Открываем изображение
    img = Image.open(image_path)

    # Извлекаем текст с изображения
    text = pytesseract.image_to_string(img, lang="rus+eng")

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
    text = re.sub(r"[^А-Яа-яЁё \s.,!?:;-]", " ", text)

    # Заменяем последовательности пробелов на одиночные пробелы
    text = re.sub(r"\s+", " ", text)

    # Удаляем пробелы перед или после дефиса
    text = re.sub(r"-\s|\s-", "", text)

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
    pdf_file = open(pdf_path, "rb")  # Открываем PDF файл
    pdf_reader = PyPDF2.PdfReader(pdf_file)  # Читаем PDF файл
    text_per_page = {}  # Словарь для хранения текста по страницам
    image_flag = False  # Флаг для обнаружения изображений
    for pagenum, page in enumerate(
        extract_pages(pdf_path)
    ):  # Итерация по страницам PDF
        page_obj = pdf_reader.pages[pagenum]  # Объект страницы PDF
        page_text = []  # Текст на странице
        line_format = []  # Форматирование строк
        text_from_images = []  # Текст изображений на странице
        text_from_tables = []  # Текст таблиц на странице
        page_content = []  # Содержимое страницы
        table_in_page = -1  # Переменная для отслеживания таблиц на странице
        pdf = pdfplumber.open(pdf_path)  # Открываем PDF файл с помощью pdfplumber
        page_tables = pdf.pages[pagenum]  # Таблицы на странице
        tables = page_tables.find_tables()  # Находим таблицы на странице
        if len(tables) != 0:
            table_in_page = 0  # Если есть таблицы, начинаем с первой

        # Итерация по таблицам на странице
        for table_num in range(len(tables)):
            table = extract_table(pdf_path, pagenum, table_num)  # Извлекаем таблицу
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
                        page_content.append(text_from_tables[table_in_page])
                        page_text.append("table")
                        line_format.append("table")
                        table_in_page += 1
                    continue

            # Если элемент не внутри таблицы
            if not is_element_inside_any_table(element, page, tables):
                if isinstance(element, LTTextContainer):
                    # Извлекаем текст из контейнера текста
                    (line_text, format_per_line) = text_extraction(element)
                    page_text.append(line_text)
                    line_format.append(format_per_line)
                    page_content.append(line_text)

                if isinstance(element, LTFigure):
                    # Обрезаем изображение и извлекаем текст
                    crop_image(element, page_obj)
                    convert_pdf_to_image("cropped_image.pdf")
                    image_text = extract_text_from_image("PDF_image.png")
                    text_from_images.append(image_text)
                    page_content.append(image_text)
                    page_text.append("image")
                    line_format.append("image")
                    image_flag = True

        # Сохраняем содержимое страницы в словаре
        dct_key = "Page_" + str(pagenum)
        text_per_page[dct_key] = [
            page_text,
            line_format,
            text_from_images,
            text_from_tables,
            page_content,
        ]

    pdf_file.close()  # Закрываем PDF файл
    if image_flag:
        os.remove("cropped_image.pdf")
        os.remove("PDF_image.png")
    # Объединяем текст со всех страниц и очищаем его
    text = "".join("".join(texts[4]) for texts in text_per_page.values())
    text = clean_text(text)
    print(f"Успешно обработан файл {os.path.splitext(os.path.basename(pdf_path))[0]}")
    if not is_broken_text(text):  # Проверяем текст на целостность
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
            print(
                f"Ошибка при обработке файла {item['id']}\n{e}"
            )  # Выводим информацию об ошибке

    # Запись успешно обработанных данных в файл JSON
    with open("server/RAG/data_text_true.json", "w", encoding="utf-8") as file:
        json.dump(new_data, file, ensure_ascii=False, indent=4)

    # Запись данных с ошибками в файл JSON
    with open("server/RAG/data_text_false.json", "w", encoding="utf-8") as file:
        json.dump(bad_data, file, ensure_ascii=False, indent=4)
