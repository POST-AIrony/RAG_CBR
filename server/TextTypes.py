from __future__ import annotations
from pydantic import Field, BaseModel
from typing import (
    Any,
    Dict,
    Literal,
)


class Document(BaseModel):
    """
    Модель для представления документа с контентом страницы и метаданными.

    Attributes:
        page_content (str): Контент страницы документа.
        metadata (dict): Метаданные документа. По умолчанию пустой словарь.
        type (Literal["Document"]): Тип документа. По умолчанию "Document".

    Methods:
        to_dict: Возвращает словарь с контентом страницы и метаданными.
        lc_secrets: Свойство, возвращает пустой словарь для хранения секретных данных.
        lc_attributes: Свойство, возвращает пустой словарь для хранения атрибутов.
        try_neq_default: Статический метод, проверяет, отличается ли значение от значения по умолчанию для поля модели.

    Special Methods:
        __repr_args__: Метод для представления аргументов объекта в строке repr.

    """
    page_content: str
    metadata: dict = Field(default_factory=dict)
    type: Literal["Document"] = "Document"

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразует документ в словарь.

        Returns:
            dict: Словарь с контентом страницы и метаданными.
        """
        return {"page_content": self.page_content, **self.metadata}

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """
        Возвращает пустой словарь для хранения секретных данных.

        Returns:
            dict: Пустой словарь.
        """
        return dict()

    @property
    def lc_attributes(self) -> Dict:
        """
        Возвращает пустой словарь для хранения атрибутов.

        Returns:
            dict: Пустой словарь.
        """
        return {}

    @classmethod
    def try_neq_default(cls, value: Any, key: str, model: BaseModel) -> bool:
        """
        Проверяет, отличается ли значение от значения по умолчанию для поля модели.

        Args:
            value (Any): Значение поля.
            key (str): Название поля.
            model (BaseModel): Модель.

        Returns:
            bool: Результат сравнения.
        """
        try:
            return model.model_fields[key].default != value
        except Exception:
            return True

    def __repr_args__(self) -> Any:
        """
        Представляет аргументы объекта в строке repr.

        Returns:
            Any: Аргументы объекта.
        """
        return [
            (k, v)
            for k, v in super().__repr_args__()
            if (k not in self.model_fields or self.try_neq_default(v, k, self))
        ]
