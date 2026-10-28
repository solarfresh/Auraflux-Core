from typing import Any, Dict, Protocol


class ToolProtocol(Protocol):
    async def run(self, **kwargs: Any) -> Any:
        ...

    def get_name(self) -> str:
        ...

    def get_description(self) -> str:
        ...

    def get_parameters(self) -> Dict[str, Any]:
        ...
