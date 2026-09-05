from enum import Enum

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, NonNegativeInt

from auraflux_core.core.tools.base_tool import BaseTool


class ThinkingLevel(str, Enum):
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MEDIUM = 'MEDIUM'
    HIGH = 'HIGH'


class AgentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    provider: str
    model: str
    lang: str = 'en'
    system_message: str | None = None
    output_format: Literal['TEXT', 'JSON'] = 'TEXT'

    thinking_level: Optional[Literal['minimal', 'low', 'medium', 'high']] = None

    tools: List["BaseTool"] = []
    tool_configs: Dict[str, Any] = {}
    tool_call_protocol: Literal['NATIVE', 'PROMPT', 'FORCE'] = 'NATIVE'
    tool_execution_strategy: Literal['NONE', 'DIRECT', 'REFLECTIVE'] = 'NONE'

    turn_limit: int = 100
    max_tokens: NonNegativeInt = 4096
    temperature: float = 0.7
