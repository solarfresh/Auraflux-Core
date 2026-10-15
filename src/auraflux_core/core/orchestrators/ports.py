from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from auraflux_core.core.orchestrators.state import (ExecutionStep,
                                                    OrchestratorState,
                                                    OrchestratorStatus)


class CheckpointPort(ABC):
    """
    Port defining the contract for persisting and loading orchestration checkpoints.
    Enables session resilience and recovery.
    """

    @abstractmethod
    async def save(self, session_id: str, state_json: str) -> None:
        """Persists the orchestrator state JSON for a given session."""
        pass

    @abstractmethod
    async def load(self, session_id: str) -> Optional[str]:
        """Retrieves the persisted state JSON for a given session, if it exists."""
        pass


class OrchestrationCallbackPort(ABC):
    """
    Port defining the telemetry and lifecycle event contract for orchestration.
    Decouples core dual-layer strategy from external transports (like SSE, WebSockets).
    """

    @abstractmethod
    async def on_status_changed(self, state: OrchestratorState, new_status: OrchestratorStatus) -> None:
        pass

    @abstractmethod
    async def on_step_start(self, state: OrchestratorState, agent_name: str, method_name: str, step_kwargs: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def on_step_complete(self, state: OrchestratorState, step: ExecutionStep) -> None:
        pass

    @abstractmethod
    async def on_orchestration_complete(self, state: OrchestratorState) -> None:
        pass

    @abstractmethod
    async def on_orchestration_failed(self, state: OrchestratorState, error_message: str) -> None:
        pass
