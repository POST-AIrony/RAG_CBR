from __future__ import annotations
from typing import Any, Sequence
import copy
import logging
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    cast,
)
from TextTypes import Document
import re
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tokenizer:
    """
    Класс для токенизации текста.

    Attributes:
        chunk_overlap (int): Количество символов перекрытия между частями текста.
        tokens_per_chunk (int): Максимальное количество токенов в части.
        decode (Callable[[List[int]], str]): Функция декодирования списка токенов в строку.
        encode (Callable[[str], List[int]]): Функция кодирования строки в список токенов.
    """

    chunk_overlap: int
    tokens_per_chunk: int
    decode: Callable[[List[int]], str]
    encode: Callable[[str], List[int]]


class SentenceChunker:
    """
    Класс для разбиения текста на части по предложениям с использованием токенизатора.

    Attributes:
        chunk_overlap (int): Количество символов перекрытия между частями текста.
        model_name (str): Название модели для разбиения текста.
        tokens_per_chunk (Optional[int]): Максимальное количество токенов в части.
        add_start_index (bool): Флаг добавления индекса начала части.
        strip_whitespace (bool): Флаг удаления пробелов в начале и конце части.

    Methods:
        create_documents(texts: List[str], metadatas: Optional[List[dict]] = None) -> List[Document]:
            Создает документы на основе списка текстов.

        split_documents(documents: Iterable[Document]) -> List[Any]:
            Разбивает список документов на тексты и метаданные и вызывает метод create_documents.

        split_text(text: str) -> List[str]:
            Разбивает текст на части с помощью токенизатора.

        count_tokens(text: str) -> int:
            Считает количество токенов в тексте.
    """

    def __init__(
        self,
        chunk_overlap: int = 50,
        model_name: str = "ai-forever/sbert_large_nlu_ru",
        tokens_per_chunk: Optional[int] = None,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
        **kwargs: Any,
    ) -> None:

        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "Пожалуйста, установите transformers с помощью `pip install transformers`."
            )
        self._chunk_overlap = chunk_overlap
        self.model_name = model_name
        self._model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._initialize_chunk_configuration(tokens_per_chunk=tokens_per_chunk)
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    def _initialize_chunk_configuration(
        self, *, tokens_per_chunk: Optional[int]
    ) -> None:
        """
        Инициализация конфигурации разбиения текста на части.

        Args:
            tokens_per_chunk (Optional[int]): Максимальное количество токенов в части.
        """

        # Получение максимального количества токенов модели.
        self.maximum_tokens_per_chunk = cast(
            int, self._model.config.max_position_embeddings
        )

        # Установка значения tokens_per_chunk в соответствии с переданным или максимальным значением.
        if tokens_per_chunk is None:
            self.tokens_per_chunk = self.maximum_tokens_per_chunk
        else:
            self.tokens_per_chunk = tokens_per_chunk

        # Проверка, что tokens_per_chunk не превышает максимальное значение токенов модели.
        if self.tokens_per_chunk > self.maximum_tokens_per_chunk:
            raise ValueError(
                f"Лимит токенов модели '{self.model_name}'"
                f" составляет: {self.maximum_tokens_per_chunk}."
                f" Аргумент tokens_per_chunk={self.tokens_per_chunk}"
                f" > максимального лимита токенов."
            )

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """
        Создание документов на основе текстов.

        Args:
            texts (List[str]): Список текстов для обработки.
            metadatas (Optional[List[dict]]): Список метаданных для каждого текста.

        Returns:
            List[Document]: Список документов.
        """

        # Создание пустых списков для текстов и метаданных.
        _metadatas = metadatas or [{}] * len(texts)
        documents = []

        # Итерация по текстам для создания документов.
        for i, text in enumerate(texts):
            index = 0
            previous_chunk_len = 0

            # Итерация по частям текста.
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])

                # Добавление индекса начала части, если флаг установлен.
                if self._add_start_index:
                    offset = index + previous_chunk_len - self._chunk_overlap
                    index = text.find(chunk, max(0, offset))
                    metadata["start_index"] = index
                    previous_chunk_len = len(chunk)

                # Создание нового документа и добавление его в список документов.
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: Iterable[Document]) -> List[Any]:
        """
        Разбивает список документов на отдельные тексты и метаданные,
        и вызывает метод create_documents для создания документов на основе текстов.

        Args:
            documents (Iterable[Document]): Итерируемый список документов.

        Returns:
            List[Any]: Список созданных документов.
        """
        # Извлечение текстов и метаданных из документов
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        # Создание документов на основе текстов и метаданных
        return self.create_documents(texts, metadatas=metadatas)

    def split_text(self, text: str) -> List[str]:
        """
        Разбивает текст на части с помощью токенизатора.

        Args:
            text (str): Исходный текст.

        Returns:
            List[str]: Список частей текста.
        """

        def encode_strip_start_and_stop_token_ids(text: str) -> List[int]:
            # Функция для кодирования текста без токенов начала и конца
            return self._encode(text)[1:-1]

        # Создание экземпляра Tokenizer с необходимыми параметрами
        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self.tokens_per_chunk,
            decode=self.tokenizer.decode,
            encode=encode_strip_start_and_stop_token_ids,
        )
        # Разбиение текста на части на основе токенов
        return self.split_text_on_tokens(text=text, tokenizer=tokenizer)

    @staticmethod
    def split_text_on_tokens(*, text: str, tokenizer: Tokenizer) -> List[str]:
        """
        Разбивает текст на части с учетом токенизатора.

        Args:
            text (str): Исходный текст.
            tokenizer (Tokenizer): Токенизатор для разбиения текста.

        Returns:
            List[str]: Список частей текста.
        """
        # Разбиение текста на части на основе токенов
        splits: List[str] = []
        input_ids = tokenizer.encode(text)
        start_idx = 0
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
        while start_idx < len(input_ids):
            splits.append(tokenizer.decode(chunk_ids))
            if cur_idx == len(input_ids):
                break
            start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
            cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
            chunk_ids = input_ids[start_idx:cur_idx]
        return splits

    def count_tokens(self, *, text: str) -> int:
        """
        Считает количество токенов в тексте.

        Args:
            text (str): Исходный текст.

        Returns:
            int: Количество токенов.
        """
        return len(self._encode(text))

    _max_length_equal_32_bit_integer: int = 2**32

    def _encode(self, text: str) -> List[int]:
        """
        Кодирует текст в токены с учетом максимальной длины.

        Args:
            text (str): Исходный текст.

        Returns:
            List[int]: Список токенов.
        """
        token_ids_with_start_and_end_token_ids = self.tokenizer.encode(
            text,
            max_length=self._max_length_equal_32_bit_integer,
            truncation="do_not_truncate",
        )
        return token_ids_with_start_and_end_token_ids


class RecursiveChunker:
    """
    Класс для разделения текста на части с учётом различных разделителей и ограничений на размер части.

    Args:
        separators (Optional[List[str]]): Список разделителей, по которым будет производиться разделение текста.
            По умолчанию содержит стандартные разделители: ["\n\n", "\n", " ", ""].
        keep_separator (bool): Флаг, определяющий, нужно ли сохранять разделители в итоговых частях текста. По умолчанию True.
        is_separator_regex (bool): Флаг, указывающий, являются ли разделители регулярными выражениями. По умолчанию False.
        length_function (Callable[[str], int]): Функция для определения длины текста. По умолчанию используется функция len.
        chunk_size (int): Максимальный размер части текста в символах. По умолчанию 256.
        chunk_overlap (int): Количество символов перекрытия между частями текста. По умолчанию 50.
        strip_whitespace (bool): Флаг, определяющий, нужно ли удалять пробелы в начале и конце частей текста. По умолчанию True.
        **kwargs: Дополнительные аргументы.

    Methods:
        _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
            Объединяет части текста с разделителем separator, учитывая ограничения на размер части.

        _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
            Объединяет список частей текста в один текст с разделителем separator.

        create_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> List[Document]:
            Создает список документов на основе текстов.

        split_documents(self, documents: Iterable[Document]) -> List[Any]:
            Разбивает документы на части и создает новые документы на основе частей текста.

        transform_documents(self, documents: Sequence[Document]) -> Sequence[Document]:
            Трансформирует документы, разбивая их на части и создавая новые документы на основе частей текста.

        _split_text_with_regex(text: str, separator: str, keep_separator: bool) -> List[str]:
            Разделяет текст на части с помощью регулярного выражения separator.

        _split_text(self, text: str, separators: List[str]) -> List[str]:
            Рекурсивно разбивает текст на части с учетом списка разделителей separators.

        split_text(self, text: str) -> List[str]:
            Разделяет текст на части с учетом установленных разделителей.

    """

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
        length_function: Callable[[str], int] = len,
        chunk_size: int = 256,
        chunk_overlap: int = 50,
        strip_whitespace: bool = True,
        add_start_index: bool = False,
        **kwargs: Any,
    ) -> None:
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        """
        Объединяет части текста с разделителем separator, учитывая ограничения на размер части.

        Args:
            splits (Iterable[str]): Список частей текста.
            separator (str): Разделитель для объединения частей.

        Returns:
            List[str]: Список объединенных частей текста.
        """
        separator_len = self._length_function(separator)

        docs = []
        current_doc: List[str] = []
        total = 0
        for split in splits:
            _len = self._length_function(split)
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    logger.warning(
                        f"Создан фрагмент размером {total}, "
                        f"который длиннее указанного {self._chunk_size}"
                    )

                if len(current_doc) > 0:
                    # Объединяем текущий документ и добавляем его в список объединенных документов.
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)

                    # Удаляем из текущего документа части, которые не поместились в текущий чанк.
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(split)
            total += _len + (separator_len if len(current_doc) > 1 else 0)

        # Объединяем оставшиеся части текста в последний документ.
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        """
        Объединяет список частей текста в один текст с разделителем separator.

        Args:
            docs (List[str]): Список частей текста.
            separator (str): Разделитель для объединения частей.

        Returns:
            Optional[str]: Объединенный текст или None, если текст пуст.
        """
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        if text == "":
            return None
        else:
            return text

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """
        Создает список документов на основе текстов.

        Args:
            texts (List[str]): Список текстов для обработки.
            metadatas (Optional[List[dict]]): Список метаданных для каждого текста.

        Returns:
            List[Document]: Список созданных документов.
        """
        # Инициализация метаданных для текстов, если они не заданы.
        _metadatas = metadatas or [{}] * len(texts)
        documents = []

        for i, text in enumerate(texts):
            index = 0
            previous_chunk_len = 0

            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])

                if self._add_start_index:
                    offset = index + previous_chunk_len - self._chunk_overlap
                    # Находим индекс начала текущей части в исходном тексте.
                    index = text.find(chunk, max(0, offset))
                    metadata["start_index"] = index
                    previous_chunk_len = len(chunk)

                # Создаем новый документ на основе части текста и метаданных.
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: Iterable[Document]) -> List[Any]:
        """
        Разбивает документы на части и создает новые документы на основе частей текста.

        Args:
            documents (Iterable[Document]): Список документов для разбиения.

        Returns:
            List[Any]: Список созданных документов или их частей.
        """
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    @staticmethod
    def _split_text_with_regex(
        text: str, separator: str, keep_separator: bool
    ) -> List[str]:
        """
        Разделяет текст на части с использованием регулярного выражения separator.

        Args:
            text (str): Исходный текст.
            separator (str): Регулярное выражение для разделения текста.
            keep_separator (bool): Флаг, указывающий, нужно ли сохранять разделители.

        Returns:
            List[str]: Список частей текста.
        """
        if separator:
            if keep_separator:
                splits = re.split(f"({separator})", text)
                # Объединяем разделители с соответствующими частями текста.
                splits = [splits[i] + splits[i + 1] for i in range(1, len(splits), 2)]
                # Если количество разделителей нечетное, добавляем последний элемент.
                if len(splits) % 2 == 0:
                    splits += splits[-1:]
                # Добавляем первоначальный элемент текста в начало списка.
                splits = [splits[0]] + splits
            else:
                splits = re.split(separator, text)
        else:
            # Если разделитель не указан, разбиваем текст на символы.
            splits = list(text)
        # Удаляем пустые строки из списка.
        return [s for s in splits if s != ""]

    def transform_documents(self, documents: Sequence[Document]) -> Sequence[Document]:
        return self.split_documents(list(documents))

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Рекурсивно разбивает текст на части с учетом списка разделителей separators.

        Args:
            text (str): Исходный текст.
            separators (List[str]): Список разделителей для разбиения текста.

        Returns:
            List[str]: Список частей текста.
        """
        # Инициализация пустого списка для хранения окончательных частей текста.
        final_chunks = []

        # Инициализация разделителя как последнего в списке.
        current_separator = separators[-1]

        # Инициализация списка разделителей.
        remaining_separators = []

        # Проверка каждого разделителя в списке.
        for i, separator in enumerate(separators):
            regex_separator = (
                separator if self._is_separator_regex else re.escape(separator)
            )

            # Если разделитель пустой, устанавливаем его и прерываем цикл.
            if separator == "":
                current_separator = separator
                break

            # Если найден разделитель в тексте, устанавливаем его и запоминаем оставшиеся разделители.
            if re.search(regex_separator, text):
                current_separator = separator
                remaining_separators = separators[i + 1 :]
                break

        # Применяем регулярное выражение к разделителю.
        regex_separator = (
            current_separator
            if self._is_separator_regex
            else re.escape(current_separator)
        )

        # Разделяем текст на части с помощью регулярного выражения.
        splits = self._split_text_with_regex(
            text, regex_separator, self._keep_separator
        )

        # Создаем список для хранения "хороших" частей текста.
        good_splits = []

        # Инициализация разделителя как пустой строки или исходного разделителя.
        current_separator = "" if self._keep_separator else current_separator

        # Проходим по каждой части текста после разделения.
        for split in splits:

            # Если длина части меньше максимального размера части, добавляем её в "хорошие" части.
            if self._length_function(split) < self._chunk_size:
                good_splits.append(split)
            else:

                # Если часть больше максимального размера, объединяем "хорошие" части и добавляем в итоговый список.
                if good_splits:
                    merged_text = self._merge_splits(good_splits, current_separator)
                    final_chunks.extend(merged_text)
                    good_splits = []

                # Если есть новые разделители, разбиваем текущую часть на еще более мелкие части.
                if not remaining_separators:
                    final_chunks.append(split)
                else:
                    other_chunks = self._split_text(split, remaining_separators)
                    final_chunks.extend(other_chunks)

        # Добавляем "хорошие" части, если они остались после обработки.
        if good_splits:
            merged_text = self._merge_splits(good_splits, current_separator)
            final_chunks.extend(merged_text)

        # Возвращаем итоговый список частей текста.
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        """
        Разделяет текст на части с учетом установленных разделителей.

        Args:
            text (str): Исходный текст.

        Returns:
            List[str]: Список частей текста.
        """
        return self._split_text(text, self._separators)
