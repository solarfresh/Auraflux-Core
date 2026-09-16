import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional

import structlog

from auraflux_core.core.configs.logging_config import get_logger
from auraflux_core.core.orchestrators.state import (ExecutionStep,
                                                    OrchestratorState)

logger = get_logger(__name__)


class OrchestrationStrategy(ABC):
    """
    Abstract Interface for execution logic.
    Subclasses implement specific patterns like Sequential or Agentic.
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
            logger.error(
                "actor_not_found",
                actor_name=actor_name,
                available_actors=list(actors.keys()),
            )
            raise ValueError(f"Actor '{actor_name}' not found in provided context.")

        # Uniformly identify the callable method (Agent.process/audit vs Tool.run)
        # If method_name is 'process' but actor is a Tool, fall back to 'run'
        actual_method = getattr(actor, method_name, None)
        if not actual_method and hasattr(actor, "run"):
            actual_method = actor.run

        if not actual_method:
            logger.error(
                "actor_method_missing",
                actor_name=actor_name,
                requested_method=method_name,
            )
            raise AttributeError(f"Actor '{actor_name}' has no method '{method_name}' or 'run'.")

        resolved_method_name = actual_method.__name__

        # 綁定當前 Dispatch 的 Actor 上下文
        structlog.contextvars.bind_contextvars(
            actor_name=actor_name,
            dispatch_method=resolved_method_name,
        )

        start_time = time.perf_counter()
        logger.info("dispatch_started")

        try:
            # Execute the call
            result = await actual_method(**kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000
            token_usage = getattr(result, "token_usage", 0)

            # Consistent Telemetry Recording
            step = ExecutionStep(
                actor_name=actor_name,
                input_params=kwargs,
                output_data=result,
                token_usage=token_usage,
                duration_ms=duration_ms
            )
            state.add_step(step)

            logger.info(
                "dispatch_completed",
                duration_ms=round(duration_ms, 2),
                token_usage=token_usage,
                status="success",
            )
            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "dispatch_failed",
                error_type=type(e).__name__,
                error_msg=str(e),
                duration_ms=round(duration_ms, 2),
                exc_info=True,
            )
            state.add_step(ExecutionStep(actor_name=actor_name, input_params=kwargs, error=str(e)))
            raise e
        finally:
            # 清理 ContextVars
            structlog.contextvars.unbind_contextvars("actor_name", "dispatch_method")

    def register_tools_to_agents(
        self, tools: Dict[str, Any],
        agents: Dict[str, Any],
        target_agent_keys: Optional[Iterable[str]] = None
    ) -> None:
        """
        Broadcasts and registers tools across target agents.
        If target_agent_keys is None, attempts to register across all agents in the dictionary.
        """
        if not tools or not agents:
            logger.debug(
                "tools_registration_skipped",
                has_tools=bool(tools),
                has_agents=bool(agents),
            )
            return

        keys_to_process = target_agent_keys if target_agent_keys is not None else agents.keys()
        registered_count = 0

        for agent_key in keys_to_process:
            agent = agents.get(agent_key)
            if agent and hasattr(agent, "register_tools"):
                agent.register_tools(tools)
                registered_count += 1

        logger.info(
            "tools_registered_to_agents",
            tool_names=list(tools.keys()),
            target_agent_count=registered_count,
        )
