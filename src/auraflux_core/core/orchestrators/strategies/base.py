import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

from auraflux_core.core.orchestrators.state import (ExecutionStep,
                                                    OrchestratorState)


class OrchestrationStrategy(ABC):
    """
    Abstract Interface for execution logic.
    Subclasses implement specific patterns like Sequential or Agentic.
    """
    def __init__(self):
        self.logger = logging.getLogger(f"[{self.__class__.__name__}]")

    @abstractmethod
    async def execute(
        self,
        input_data: Any,
        tools: Dict[str, Any],
        agents: Dict[str, Any],
        state: OrchestratorState
    ) -> OrchestratorState:
        pass

    async def dispatch(
        self,
        state: OrchestratorState,
        actor_name: str,
        actors: Dict[str, Any],
        method_name: str = "generate", # Default to Agent's generate or Tool's run
        **kwargs
    ) -> Any:
        """
        Unified dispatching mechanism to ensure consistent telemetry and error handling.
        """
        actor = actors.get(actor_name)
        if not actor:
            raise ValueError(f"Actor '{actor_name}' not found in provided context.")

        # Uniformly identify the callable method (Agent.process/audit vs Tool.run)
        # If method_name is 'process' but actor is a Tool, fall back to 'run'
        actual_method = getattr(actor, method_name, None)
        if not actual_method and hasattr(actor, "run"):
            actual_method = actor.run

        if not actual_method:
            raise AttributeError(f"Actor '{actor_name}' has no method '{method_name}' or 'run'.")

        start_time = time.perf_counter()
        try:
            # Execute the call
            result = await actual_method(**kwargs)
            duration = (time.perf_counter() - start_time) * 1000

            # Consistent Telemetry Recording
            step = ExecutionStep(
                tool_name=actor_name,
                input_params=kwargs,
                output_data=result,
                token_usage=result.get("usage", 0) if isinstance(result, dict) else 0,
                duration_ms=duration
            )
            state.add_step(step)
            return result

        except Exception as e:
            self.logger.error(f"Dispatch failure for {actor_name}: {str(e)}")
            state.add_step(ExecutionStep(tool_name=actor_name, input_params=kwargs, error=str(e)))
            raise e