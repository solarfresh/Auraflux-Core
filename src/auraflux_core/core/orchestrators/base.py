import time
from typing import Any, Dict, List, Optional

import structlog

from auraflux_core.core.configs.logging_config import get_logger
from auraflux_core.core.orchestrators.state import (OrchestratorState,
                                                    OrchestratorStatus)
from auraflux_core.core.orchestrators.strategies.base import \
    OrchestrationStrategy
from auraflux_core.core.tools.base_tool import BaseTool

logger = get_logger(__name__)


class BaseOrchestrator:
    """
    Universal Context for AI Orchestration.
    Uses the Strategy Pattern to switch between Sequential and Agentic modes.
    """
    def __init__(
        self,
        strategy: OrchestrationStrategy,
        tools: List[BaseTool],
        agents: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ):
        self.strategy = strategy
        self.tools = {tool.get_name(): tool for tool in tools}
        self.agents = agents
        self.config = config or {}
        self.state = OrchestratorState()

        logger.info(
            "orchestrator_initialized",
            orchestrator_class=self.__class__.__name__,
            strategy=self.strategy.__class__.__name__,
            tool_count=len(self.tools),
            agent_names=list(self.agents.keys()),
        )

    async def run(self, input_data: Any) -> OrchestratorState:
        """
        Delegates the execution to the injected strategy.
        Automatically binds orchestrator context and logs execution state transitions.
        """
        orchestrator_name = self.__class__.__name__
        strategy_name = self.strategy.__class__.__name__

        # 綁定 Orchestrator 層級的 Context
        structlog.contextvars.bind_contextvars(
            orchestrator=orchestrator_name,
            orchestration_strategy=strategy_name,
        )

        start_time = time.perf_counter()
        logger.info(
            "orchestration_started",
            strategy=strategy_name,
        )

        self._update_status_and_log(OrchestratorStatus.INGESTING)

        try:
            # Delegate execution logic to the strategy
            final_state = await self.strategy.execute(
                input_data=input_data,
                tools=self.tools,
                agents=self.agents,
                state=self.state
            )

            latency_sec = time.perf_counter() - start_time
            self._update_status_and_log(OrchestratorStatus.COMPLETED)

            logger.info(
                "orchestration_completed",
                latency_sec=round(latency_sec, 4),
                status="success",
            )
            return final_state

        except Exception as e:
            latency_sec = time.perf_counter() - start_time
            self._update_status_and_log(OrchestratorStatus.FAILED)
            self.state.metadata["error"] = str(e)

            logger.error(
                "orchestration_failed",
                error_type=type(e).__name__,
                error_msg=str(e),
                latency_sec=round(latency_sec, 4),
                exc_info=True,
            )
            return self.state
        finally:
            # 清理 Orchestrator 層級 Context
            structlog.contextvars.unbind_contextvars("orchestrator", "orchestration_strategy")

    def set_strategy(self, strategy: OrchestrationStrategy):
        """Allows dynamic switching of the orchestration logic at runtime."""
        previous_strategy = self.strategy.__class__.__name__
        new_strategy = strategy.__class__.__name__

        logger.info(
            "strategy_switched",
            previous_strategy=previous_strategy,
            new_strategy=new_strategy,
        )
        self.strategy = strategy

    def _update_status_and_log(self, new_status: OrchestratorStatus):
        """Helper method to update state status and emit structured log."""
        old_status = getattr(self.state, "status", None)
        self.state.update_status(new_status)
        logger.info(
            "orchestrator_status_changed",
            previous_status=str(old_status) if old_status else None,
            current_status=str(new_status),
        )
