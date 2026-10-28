from typing import Any, Dict, List, Protocol


class AgentProtocol(Protocol):
    name: str

    async def generate(self, messages: List[Any]) -> Any:
        ...

    async def execute(self, payload: Dict[str, Any]) -> Any:
        ...

    def register_tools(self, tools: Any) -> "AgentProtocol":
        ...
