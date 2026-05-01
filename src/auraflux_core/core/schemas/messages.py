from typing import Optional

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str
    name: str
    token_usage: Optional[int] = 0
