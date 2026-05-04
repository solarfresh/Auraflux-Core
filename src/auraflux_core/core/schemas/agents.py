from enum import Enum
from typing import Any, Dict, Literal

from pydantic import BaseModel, NonNegativeInt


class AgentConfig(BaseModel):
    name: str
    provider: str
    model: str
    lang: str = 'en'
    system_message: str | None = None

    tool_call_protocol: Literal['NATIVE', 'PROMPT'] = 'NATIVE'
    tool_execution_strategy: Literal['NONE', 'DIRECT', 'REFLECTIVE'] = 'NONE'
    tool_configs: Dict[str, Any] = {}

    cot_message: str | None = None
    turn_limit: int = 100
    max_tokens: NonNegativeInt = 4096
    temperature: float = 0.7
