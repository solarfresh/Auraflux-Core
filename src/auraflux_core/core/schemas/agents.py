from enum import Enum
from typing import Literal

from pydantic import BaseModel


class AgentConfig(BaseModel):
    name: str
    model: str
    lang: str = 'en'
    system_message: str | None = None
    tool_use: Literal['TOOL_USE_DIRECT', 'TOOL_USE_AND_PROCESS', 'NO_TOOL_USE'] = 'NO_TOOL_USE'
    cot_message: str | None = None
    turn_limit: int = 100
