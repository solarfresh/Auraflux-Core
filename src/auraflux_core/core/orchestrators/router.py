from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from auraflux_core.core.configs.logging_config import get_logger
from auraflux_core.core.orchestrators.state import OrchestratorState

logger = get_logger(__name__)


class BaseAgentRouter(ABC):
    """
    Abstract Base Class for multi-agent routing strategies.
    Decouples routing logic from the execution strategy.
    """

    def __init__(self, max_steps: int = 10) -> None:
        self.max_steps = max_steps

    @abstractmethod
    def get_next_agent(
        self, state: OrchestratorState, agents: Dict[str, Any], last_result: Any = None
    ) -> Optional[str]:
        """
        Determines the next agent to take control based on the current state and last execution result.
        Returns the agent key/name, or None to terminate orchestration.
        """
        pass

    def evaluate_completion(self, state: OrchestratorState, last_result: Any) -> bool:
        """
        Evaluates whether the overall orchestration task is completed.
        """
        if len(state.history) >= self.max_steps:
            logger.info("orchestration_terminated_max_steps_reached", total_steps=len(state.history))
            return True

        if last_result and getattr(last_result, "is_terminal", False):
            logger.info("orchestration_marked_as_complete")
            return True

        return False


class DynamicHandoffRouter(BaseAgentRouter):
    """
    Dynamic router supporting explicit agent handoff signals (e.g., result.next_agent)
    and fallback sequence rotation.
    """

    def get_next_agent(
        self, state: OrchestratorState, agents: Dict[str, Any], last_result: Any = None
    ) -> Optional[str]:
        if not agents:
            logger.warning("routing_failed_no_agents_available")
            return None

        # 1. Check explicit handoff signal from previous step output
        if last_result and hasattr(last_result, "next_agent") and last_result.next_agent in agents:
            next_agent = last_result.next_agent
            logger.info("agent_routed_via_handoff_signal", next_agent=next_agent)
            return next_agent

        step_count = len(state.history)
        if step_count >= self.max_steps:
            return None

        # 2. Initial routing fallback
        if step_count == 0:
            default_agent = list(agents.keys())[0]
            logger.info("agent_routed_initial", next_agent=default_agent)
            return default_agent

        # 3. Dynamic alternation fallback
        available_agent_names = list(agents.keys())
        last_actor = state.history[-1].actor_name if state.history else None
        candidates = [name for name in available_agent_names if name != last_actor]
        selected_agent = candidates[0] if candidates else available_agent_names[0]

        logger.info(
            "agent_routed_dynamically",
            next_agent=selected_agent,
            previous_actor=last_actor,
        )
        return selected_agent


class SequentialAgentRouter(BaseAgentRouter):
    """
    Classic sequential router that cycles through available agents
    in a deterministic, round-robin dictionary order.
    """

    def get_next_agent(
        self, state: OrchestratorState, agents: Dict[str, Any], last_result: Any = None
    ) -> Optional[str]:
        if not agents:
            logger.warning("routing_failed_no_agents_available")
            return None

        step_count = len(state.history)
        if step_count >= self.max_steps:
            logger.info("orchestration_terminated_max_steps_reached", total_steps=step_count)
            return None

        available_agent_names = list(agents.keys())
        next_agent = available_agent_names[step_count % len(available_agent_names)]

        logger.info(
            "agent_routed_sequentially",
            next_agent=next_agent,
            step_count=step_count,
        )
        return next_agent
