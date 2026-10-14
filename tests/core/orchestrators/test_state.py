from datetime import datetime
from auraflux_core.core.orchestrators.state import OrchestratorState, OrchestratorStatus, ExecutionStep


def test_orchestrator_state_initialization() -> None:
    """
    Validate that OrchestratorState initializes with correct default values
    and the IDLE status.
    """
    state = OrchestratorState()

    assert state.session_id.startswith("session_")
    assert state.status == OrchestratorStatus.IDLE
    assert state.history == []
    assert state.retry_count == 0
    assert state.total_token_count == 0
    assert state.output == {}
    assert state.metadata == {}


def test_orchestrator_state_status_update() -> None:
    """
    Validate that update_status correctly modifies the status of the orchestrator state.
    """
    state = OrchestratorState()

    state.update_status(OrchestratorStatus.PROCESSING)
    assert state.status == OrchestratorStatus.PROCESSING

    state.update_status(OrchestratorStatus.VALIDATING)
    assert state.status == OrchestratorStatus.VALIDATING

    state.update_status(OrchestratorStatus.COMPLETED)
    assert state.status == OrchestratorStatus.COMPLETED


def test_execution_step_creation() -> None:
    """
    Validate that ExecutionStep initializes correctly with default values
    and provided parameters.
    """
    step = ExecutionStep(
        actor_name="agent_alpha",
        method_name="process_query",
        input_params={"prompt": "hello world"},
        output_data={"response": "hi"},
        token_usage=25,
        duration_ms=45.67
    )

    assert step.step_id is not None
    assert isinstance(step.timestamp, datetime)
    assert step.actor_name == "agent_alpha"
    assert step.method_name == "process_query"
    assert step.input_params == {"prompt": "hello world"}
    assert step.output_data == {"response": "hi"}
    assert step.token_usage == 25
    assert step.duration_ms == 45.67
    assert step.error is None


def test_orchestrator_state_add_step_and_token_accumulation() -> None:
    """
    Validate that adding execution steps correctly updates the history list
    and accumulates total token usage.
    """
    state = OrchestratorState()

    step_1 = ExecutionStep(
        actor_name="agent_a",
        method_name="step_one",
        input_params={"x": 1},
        token_usage=10
    )
    step_2 = ExecutionStep(
        actor_name="agent_b",
        method_name="step_two",
        input_params={"y": 2},
        token_usage=35
    )

    state.add_step(step_1)
    state.add_step(step_2)

    assert len(state.history) == 2
    assert state.history[0] == step_1
    assert state.history[1] == step_2
    assert state.total_token_count == 45  # 10 + 35
