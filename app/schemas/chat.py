from typing import Any

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    context: dict[str, Any] = {}


class ChatAction(BaseModel):
    type: str
    page: str


class ChatResponse(BaseModel):
    reply: str
    actions: list[ChatAction] = []
