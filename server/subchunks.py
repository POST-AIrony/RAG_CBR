from __future__ import annotations
from typing import Any, Sequence
from typing_extensions import ParamSpec
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
    TypeVar,
    cast,
)
from TextTypes import Document
import re
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


@dataclass(frozen=True)
class Tokenizer:
    chunk_overlap: int
    tokens_per_chunk: int
    decode: Callable[[List[int]], str]
    encode: Callable[[str], List[int]]


class SentenceChunker:
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
        self.maximum_tokens_per_chunk = cast(
            int, self._model.config.max_position_embeddings
        )

        if tokens_per_chunk is None:
            self.tokens_per_chunk = self.maximum_tokens_per_chunk
        else:
            self.tokens_per_chunk = tokens_per_chunk

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
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            index = 0
            previous_chunk_len = 0
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    offset = index + previous_chunk_len - self._chunk_overlap
                    index = text.find(chunk, max(0, offset))
                    metadata["start_index"] = index
                    previous_chunk_len = len(chunk)
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: Iterable[Document]) -> List[Any]:
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def split_text(self, text: str) -> List[str]:
        def encode_strip_start_and_stop_token_ids(text: str) -> List[int]:
            return self._encode(text)[1:-1]

        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self.tokens_per_chunk,
            decode=self.tokenizer.decode,
            encode=encode_strip_start_and_stop_token_ids,
        )
        return self.split_text_on_tokens(text=text, tokenizer=tokenizer)

    @staticmethod
    def split_text_on_tokens(*, text: str, tokenizer: Tokenizer) -> List[str]:
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
        return len(self._encode(text))

    _max_length_equal_32_bit_integer: int = 2**32

    def _encode(self, text: str) -> List[int]:
        token_ids_with_start_and_end_token_ids = self.tokenizer.encode(
            text,
            max_length=self._max_length_equal_32_bit_integer,
            truncation="do_not_truncate",
        )
        return token_ids_with_start_and_end_token_ids


class RecursiveChunker:
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
        separator_len = self._length_function(separator)

        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
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
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            index = 0
            previous_chunk_len = 0
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    offset = index + previous_chunk_len - self._chunk_overlap
                    index = text.find(chunk, max(0, offset))
                    metadata["start_index"] = index
                    previous_chunk_len = len(chunk)
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: Iterable[Document]) -> List[Any]:
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    @staticmethod
    def _split_text_with_regex(
        text: str, separator: str, keep_separator: bool
    ) -> List[str]:
        if separator:
            if keep_separator:
                _splits = re.split(f"({separator})", text)
                splits = [
                    _splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)
                ]
                if len(_splits) % 2 == 0:
                    splits += _splits[-1:]
                splits = [_splits[0]] + splits
            else:
                splits = re.split(separator, text)
        else:
            splits = list(text)
        return [s for s in splits if s != ""]

    def transform_documents(self, documents: Sequence[Document]) -> Sequence[Document]:
        return self.split_documents(list(documents))

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = self._split_text_with_regex(text, _separator, self._keep_separator)
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, self._separators)


if __name__ == "__main__":
    import json

    text_splitter = SentenceChunker()
    with open("server/RAG/data_text_true.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    text = []
    metadata = []
    for i, item in enumerate(data):
        text.append(item.pop("text"))
        metadata.append(item)
        if i == 1000:
            break
    docs = text_splitter.create_documents(text, metadata)
    chunking = text_splitter.split_documents(docs)
    print(len(chunking))
