import pytest
from typing import Any, Dict, Optional
from auraflux_core.core.orchestrators.state import OrchestratorState, ExecutionStep
from auraflux_core.core.orchestrators.router import (
    SequentialAgentRouter,
    DynamicHandoffRouter,
)


class DummyResult:
    """Helper class to simulate execution result objects."""
    def __init__(self, next_agent: Optional[str] = None, is_terminal: bool = False):
        self.next_agent = next_agent
        self.is_terminal = is_terminal


def test_sequential_router_cycling(mock_orchestrator_state: OrchestratorState, mock_agents_registry: Dict[str, Any]) -> None:
    """
    Validates that SequentialAgentRouter cycles through available agents in a round-robin order.
    """
    router = SequentialAgentRouter(max_steps=5)

    # Step 0: Should pick the first agent
    agent_1 = router.get_next_agent(mock_orchestrator_state, mock_agents_registry)
    assert agent_1 == "agent_a"

    # Simulate history recording for step 1 using ExecutionStep definition
    mock_orchestrator_state.history.append(
        ExecutionStep(actor_name="agent_a", input_params={"payload": {}})
    )

    # Step 1: Should pick the second agent
    agent_2 = router.get_next_agent(mock_orchestrator_state, mock_agents_registry)
    assert agent_2 == "agent_b"

    mock_orchestrator_state.history.append(
        ExecutionStep(actor_name="agent_b", input_params={"payload": {}})
    )

    # Step 2: Should pick the third agent
    agent_3 = router.get_next_agent(mock_orchestrator_state, mock_agents_registry)
    assert agent_3 == "agent_c"

    mock_orchestrator_state.history.append(
        ExecutionStep(actor_name="agent_c", input_params={"payload": {}})
    )

    # Step 3: Should loop back to the first agent
    agent_4 = router.get_next_agent(mock_orchestrator_state, mock_agents_registry)
    assert agent_4 == "agent_a"


def test_dynamic_handoff_router_with_signal(mock_orchestrator_state: OrchestratorState, mock_agents_registry: Dict[str, Any]) -> None:
    """
    Validates that DynamicHandoffRouter correctly obeys explicit handoff signals (result.next_agent).
    """
    router = DynamicHandoffRouter(max_steps=5)

    # Initial routing
    first_agent = router.get_next_agent(mock_orchestrator_state, mock_agents_registry)
    assert first_agent == "agent_a"

    mock_orchestrator_state.history.append(
        ExecutionStep(actor_name="agent_a", input_params={"payload": {}})
    )

    # Provide a result explicitly handing off control to 'agent_c'
    handoff_result = DummyResult(next_agent="agent_c")
    next_agent = router.get_next_agent(mock_orchestrator_state, mock_agents_registry, last_result=handoff_result)

    assert next_agent == "agent_c"


def test_dynamic_handoff_router_fallback(mock_orchestrator_state: OrchestratorState, mock_agents_registry: Dict[str, Any]) -> None:
    """
    Validates that DynamicHandoffRouter alternates dynamically when no explicit handoff signal is present.
    """
    router = DynamicHandoffRouter(max_steps=5)

    # Initial routing
    assert router.get_next_agent(mock_orchestrator_state, mock_agents_registry) == "agent_a"

    # Simulate execution by agent_a without handoff signal
    mock_orchestrator_state.history.append(
        ExecutionStep(actor_name="agent_a", input_params={"payload": {}})
    )

    # Should fallback to an agent other than 'agent_a'
    fallback_agent = router.get_next_agent(mock_orchestrator_state, mock_agents_registry, last_result=DummyResult())
    assert fallback_agent != "agent_a"
    assert fallback_agent in ["agent_b", "agent_c"]


def test_router_max_steps_termination(mock_orchestrator_state: OrchestratorState, mock_agents_registry: Dict[str, Any]) -> None:
    """
    Validates that the router correctly triggers completion and returns None when max_steps is reached.
    """
    router = SequentialAgentRouter(max_steps=2)

    # Execute up to max_steps
    mock_orchestrator_state.history.append(
        ExecutionStep(actor_name="agent_a", input_params={"payload": {}})
    )
    mock_orchestrator_state.history.append(
        ExecutionStep(actor_name="agent_b", input_params={"payload": {}})
    )

    # Router should return None when limit is hit
    next_agent = router.get_next_agent(mock_orchestrator_state, mock_agents_registry)
    assert next_agent is None

    # Completion evaluation should evaluate to True
    is_complete = router.evaluate_completion(mock_orchestrator_state, last_result=None)
    assert is_complete is True
