from typing import Any, Dict, List, Protocol

from auraflux_core.core.protocols.clients import ClientManagerProtocol
from auraflux_core.core.schemas.agents import AgentConfig
from auraflux_core.core.schemas.messages import Message


class AgentProtocol(Protocol):
    @property
    def name(self) -> str:
        ...

    @property
    def provider(self) -> str:
        ...

    @property
    def model(self) -> str:
        ...

    def __init__(self, config: AgentConfig, client_manager: ClientManagerProtocol) -> None:
        ...

    async def generate(self, messages: List[Any]) -> Message:
        ...

    async def execute(self, payload: Dict[str, Any]) -> Any:
        ...

    def register_tools(self, tools: Any) -> "AgentProtocol":
        ...
