from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatInfo(BaseModel):
    messages: list[ChatMessage]