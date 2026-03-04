from enum import Enum
from typing import Any, Dict, Literal

from pydantic import BaseModel, NonNegativeInt


class AgentConfig(BaseModel):
    name: str
    model: str
    lang: str = 'en'
    system_message: str | None = None

    tool_use: Literal['TOOL_USE_DIRECT', 'TOOL_USE_AND_PROCESS', 'NO_TOOL_USE'] = 'NO_TOOL_USE'
    tool_configs: Dict[str, Any] = {}

    cot_message: str | None = None
    turn_limit: int = 100
    max_tokens: NonNegativeInt = 4096
    temperature: float = 0.7
