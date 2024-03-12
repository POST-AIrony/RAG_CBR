from __future__ import annotations
from typing import Any, List, Literal
from pydantic import Field, BaseModel
from typing import (
    Any,
    Dict,
    List,
    Literal,
)


class Document(BaseModel):
    page_content: str
    metadata: dict = Field(default_factory=dict)
    type: Literal["Document"] = "Document"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        return ["langchain", "schema", "document"]

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return dict()

    @property
    def lc_attributes(self) -> Dict:
        return {}

    @classmethod
    def lc_id(cls) -> List[str]:
        return [*cls.get_lc_namespace(), cls.__name__]

    class Config:
        extra = "ignore"

    @classmethod
    def try_neq_default(cls, value: Any, key: str, model: BaseModel) -> bool:
        try:
            return model.model_fields[key].default != value
        except Exception:
            return True

    def __repr_args__(self) -> Any:
        return [
            (k, v)
            for k, v in super().__repr_args__()
            if (k not in self.model_fields or self.try_neq_default(v, k, self))
        ]
