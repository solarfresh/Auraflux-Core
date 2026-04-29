from abc import ABC, abstractmethod
from typing import Any, Dict
from auraflux_core.core.orchestrators.state import OrchestratorState


class OrchestrationStrategy(ABC):
    """
    Abstract Interface for execution logic.
    Subclasses implement specific patterns like Sequential, Reflective, or Debate.
    """
    @abstractmethod
    async def execute(
        self,
        input_data: Any,
        tools: Dict[str, Any],
        agents: Dict[str, Any],
        state: OrchestratorState
    ) -> OrchestratorState:
        pass
