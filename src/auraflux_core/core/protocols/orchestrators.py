from typing import Any, Dict, Iterable, Optional, Protocol


class OrchestrationStrategyProtocol(Protocol):
    async def execute(
        self,
        payload: Any,
        tools: Dict[str, Any],
        agents: Dict[str, Any],
        state: Any
    ) -> Any:
        ...

    async def dispatch(
        self,
        state: Any,
        actor_name: str,
        actors: Dict[str, Any],
        method_name: str = "generate",
        **kwargs
    ) -> Any:
        ...

    def register_tools_to_agents(
        self,
        tools: Dict[str, Any],
        agents: Dict[str, Any],
        target_agent_keys: Optional[Iterable[str]] = None
    ) -> None:
        ...
