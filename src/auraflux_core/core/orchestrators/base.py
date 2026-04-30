from typing import Any, Dict, List, Optional

from auraflux_core.core.configs.logging_config import setup_logging
from auraflux_core.core.orchestrators.state import (OrchestratorState,
                                                    OrchestratorStatus)
from auraflux_core.core.orchestrators.strategies.base import \
    OrchestrationStrategy
from auraflux_core.core.tools.base_tool import BaseTool


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
        self.logger = setup_logging(name=f"[{self.__class__.__name__}]")

    async def run(self, input_data: Any) -> OrchestratorState:
        """
        Delegates the execution to the injected strategy.
        """
        self.logger.info(f"Starting orchestration with {self.strategy.__class__.__name__}")
        self.state.update_status(OrchestratorStatus.INGESTING)

        try:
            # Delegate execution logic to the strategy
            final_state = await self.strategy.execute(
                input_data=input_data,
                tools=self.tools,
                agents=self.agents,
                state=self.state
            )
            self.state.update_status(OrchestratorStatus.COMPLETED)
            return final_state

        except Exception as e:
            self.logger.error(f"Orchestration failed: {str(e)}")
            self.state.update_status(OrchestratorStatus.FAILED)
            self.state.metadata["error"] = str(e)
            return self.state

    def set_strategy(self, strategy: OrchestrationStrategy):
        """Allows dynamic switching of the orchestration logic at runtime."""
        self.logger.info(f"Switching strategy to {strategy.__class__.__name__}")
        self.strategy = strategy
