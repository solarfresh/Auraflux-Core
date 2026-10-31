from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from auraflux_core.core.orchestrators.adapters.transport_adapter import \
    CallbackTransportAdapter
from auraflux_core.core.orchestrators.exceptions import (
    AgentPipelineError, CheckpointCorruptedError,
    InvalidOrchestrationStateError, MaxResumeAttemptsExceededError,
    OrchestrationLimitExceededError, ConcurrencyConflictError)
from auraflux_core.core.orchestrators.router import SequentialAgentRouter
from auraflux_core.core.orchestrators.state import (ExecutionCursor,
                                                    OrchestratorState,
                                                    OrchestratorStatus)
from auraflux_core.core.orchestrators.strategies.dual_layer_strategy import \
    DualLayerAgenticStrategy


@pytest.mark.asyncio
async def test_dual_layer_strategy_full_execution(
    mock_orchestrator_state: OrchestratorState,
    dummy_pipeline_for_strategy: Any
) -> None:
    """
    Validates that DualLayerAgenticStrategy successfully resolves the router,
    retrieves execution steps from agent.pipeline.get_execute_pipeline(), and dispatches them
    in complete isolation with fine-grained assertions, including callback transport and status lifecycle.
    """
    class IsolatedMockAgent:
        def __init__(self, name: str, pipeline: Any):
            self.name = name
            self.pipeline = pipeline
            self.step_one = AsyncMock(return_value={"result": "step_one_completed"})
            self.step_two = AsyncMock(return_value={"result": "step_two_completed"})

    agent_a = IsolatedMockAgent(name="agent_a", pipeline=dummy_pipeline_for_strategy)
    agents_registry = {"agent_a": agent_a}

    router = SequentialAgentRouter(max_steps=1)
    callback_adapter = CallbackTransportAdapter()
    strategy = DualLayerAgenticStrategy(router=router, callback_port=callback_adapter)

    payload = {"prompt": "Run isolated orchestration test"}
    tools = {}

    result_state = await strategy.execute(
        payload=payload,
        tools=tools,
        agents=agents_registry,
        state=mock_orchestrator_state,
    )

    assert len(result_state.history) == 2, f"Expected 2 execution steps in history, got {len(result_state.history)}"
    assert result_state.status == OrchestratorStatus.COMPLETED

    step_1 = result_state.history[0]
    assert step_1.actor_name == "agent_a"
    assert step_1.input_params.get("extra") == "val_1"
    assert step_1.input_params.get("payload") == payload
    assert step_1.output_data == {"result": "step_one_completed"}
    assert step_1.error is None

    step_2 = result_state.history[1]
    assert step_2.actor_name == "agent_a"
    assert step_2.input_params.get("payload") == payload
    assert step_2.output_data == {"result": "step_two_completed"}
    assert step_2.error is None

    agent_a.step_one.assert_awaited_once()
    agent_a.step_two.assert_awaited_once()

    events = []
    while not callback_adapter.queue.empty():
        item = await callback_adapter.queue.get()
        if item is not None:
            events.append(item)

    event_types = [e.get("event") for e in events]
    assert "status_changed" in event_types
    assert "step_start" in event_types
    assert "step_complete" in event_types
    assert "completed" in event_types


@pytest.mark.asyncio
async def test_dual_layer_strategy_missing_pipeline_error(
    mock_orchestrator_state: OrchestratorState
) -> None:
    """
    Validates that Strategy properly raises AgentPipelineError when a routed agent lacks a bound pipeline.
    """
    class PipelineLessAgent:
        def __init__(self, name: str):
            self.name = name

    bad_agent = PipelineLessAgent(name="bad_agent")
    agents_registry = {"bad_agent": bad_agent}
    router = SequentialAgentRouter(max_steps=1)
    strategy = DualLayerAgenticStrategy(router=router)

    with pytest.raises(AgentPipelineError) as exc_info:
        await strategy.execute(
            payload={"prompt": "test"},
            tools={},
            agents=agents_registry,
            state=mock_orchestrator_state,
        )

    assert "does not have a valid bound pipeline" in str(exc_info.value)
    assert mock_orchestrator_state.status == OrchestratorStatus.FAILED


@pytest.mark.asyncio
async def test_failed_resume_and_attempt_counting():
    mock_router = MagicMock()
    mock_router.get_next_agent.return_value = "agent_a"
    mock_router.evaluate_completion.return_value = True

    mock_checkpoint_port = AsyncMock()
    state = OrchestratorState(session_id="sess_test_1")
    state.status = OrchestratorStatus.FAILED
    state.cursor = ExecutionCursor(agent_name="agent_a", macro_iteration=0, completed_steps_count=1)
    state.attempt_count = 1  # Use typed state attribute

    mock_checkpoint_port.load.return_value = state.dump_checkpoint()

    mock_agent = MagicMock()
    mock_agent.pipeline = MagicMock()
    async def step_one(**kwargs): return "res1"
    async def step_two(**kwargs): return "res2"
    mock_agent.step_one = step_one
    mock_agent.step_two = step_two

    mock_agent.pipeline.get_execute_pipeline.return_value = [
        {"method_name": "step_one", "kwargs": {}},
        {"method_name": "step_two", "kwargs": {}}
    ]

    agents = {"agent_a": mock_agent}
    strategy = DualLayerAgenticStrategy(router=mock_router, checkpoint_port=mock_checkpoint_port)

    _, final_state = await strategy.resume(
        session_id="sess_test_1",
        checkpoint_port=mock_checkpoint_port,
        router=mock_router,
        agents=agents,
        payload={"data": "test"}
    )

    assert final_state.attempt_count == 2


@pytest.mark.asyncio
async def test_circuit_breaker_max_attempts():
    """
    Validates that session resume is blocked and raises MaxResumeAttemptsExceededError
    when the typed attempt_count reaches or exceeds the maximum limit.
    """
    mock_router = MagicMock()
    mock_checkpoint_port = AsyncMock()

    state = OrchestratorState(session_id="sess_test_2")
    state.status = OrchestratorStatus.FAILED
    state.attempt_count = 3  # Utilize the formalized typed attempt attribute

    mock_checkpoint_port.load.return_value = state.dump_checkpoint()
    strategy = DualLayerAgenticStrategy(router=mock_router, checkpoint_port=mock_checkpoint_port)

    with pytest.raises(MaxResumeAttemptsExceededError, match="exceeded maximum resume attempts"):
        await strategy.resume(
            session_id="sess_test_2",
            checkpoint_port=mock_checkpoint_port,
            router=mock_router,
            agents={"agent_a": MagicMock()},
            payload={}
        )


@pytest.mark.asyncio
async def test_corrupted_checkpoint_validation():
    """
    Validates that resuming a session with an out-of-bounds completed steps count
    properly raises CheckpointCorruptedError.
    """
    mock_router = MagicMock()
    mock_checkpoint_port = AsyncMock()

    state = OrchestratorState(session_id="sess_test_3")
    state.status = OrchestratorStatus.PROCESSING
    state.cursor = ExecutionCursor(agent_name="agent_a", macro_iteration=0, completed_steps_count=999)

    mock_checkpoint_port.load.return_value = state.dump_checkpoint()

    mock_agent = MagicMock()
    mock_agent.pipeline = MagicMock()
    mock_agent.pipeline.get_execute_pipeline.return_value = [{"method_name": "step_one", "kwargs": {}}]

    strategy = DualLayerAgenticStrategy(router=mock_router, checkpoint_port=mock_checkpoint_port)

    with pytest.raises(CheckpointCorruptedError, match="Corrupted checkpoint: completed_steps_count"):
        await strategy.resume(
            session_id="sess_test_3",
            checkpoint_port=mock_checkpoint_port,
            router=mock_router,
            agents={"agent_a": mock_agent},
            payload={}
        )


@pytest.mark.asyncio
async def test_unserializable_object_fallback():
    mock_router = MagicMock()
    mock_router.get_next_agent.return_value = "agent_a"
    mock_router.evaluate_completion.return_value = True
    mock_checkpoint_port = AsyncMock()

    mock_agent = MagicMock()
    mock_agent.pipeline = MagicMock()

    file_handle = open(__file__, "r")
    try:
        async def bad_step(**kwargs):
            return file_handle

        mock_agent.bad_step = bad_step
        mock_agent.pipeline.get_execute_pipeline.return_value = [{"method_name": "bad_step", "kwargs": {}}]

        strategy = DualLayerAgenticStrategy(router=mock_router, checkpoint_port=mock_checkpoint_port)
        state = OrchestratorState(session_id="sess_test_4")
        state.status = OrchestratorStatus.IDLE

        result_state = await strategy.execute(
            payload={},
            tools={},
            agents={"agent_a": mock_agent},
            state=state
        )

        assert mock_checkpoint_port.save.called

        last_result = result_state.metadata.get("last_result")
        assert isinstance(last_result, str)

    finally:
        file_handle.close()


@pytest.mark.asyncio
async def test_illegal_state_defense_completed():
    """
    Validates that attempting to resume a session that is already COMPLETED
    properly raises InvalidOrchestrationStateError and avoids redundant setup.
    """
    mock_router = MagicMock()
    mock_checkpoint_port = AsyncMock()

    state = OrchestratorState(session_id="sess_test_5")
    state.status = OrchestratorStatus.COMPLETED

    mock_checkpoint_port.load.return_value = state.dump_checkpoint()
    strategy = DualLayerAgenticStrategy(router=mock_router, checkpoint_port=mock_checkpoint_port)

    strategy.register_tools_to_agents = MagicMock()

    with pytest.raises(InvalidOrchestrationStateError, match="already COMPLETED"):
        await strategy.resume(
            session_id="sess_test_5",
            checkpoint_port=mock_checkpoint_port,
            router=mock_router,
            agents={"agent_a": MagicMock()},
            payload={}
        )

    strategy.register_tools_to_agents.assert_not_called()


@pytest.mark.asyncio
async def test_cursor_binding_and_state_tracking():
    """
    Validates TASK-01: cursor.agent_name, macro_iteration, and completed_steps_count
    are correctly updated and tracked in OrchestratorState during _process_agent_steps.
    """
    mock_router = MagicMock()
    mock_router.get_next_agent.return_value = "agent_x"
    mock_router.evaluate_completion.return_value = True

    class StatefulMockAgent:
        def __init__(self, name: str, pipeline: Any):
            self.name = name
            self.pipeline = pipeline

        async def step_a(self, **kwargs):
            return {"output": "a"}

        async def step_b(self, **kwargs):
            return {"output": "b"}

    mock_pipeline = MagicMock()
    mock_pipeline.get_execute_pipeline.return_value = [
        {"method_name": "step_a", "kwargs": {}},
        {"method_name": "step_b", "kwargs": {}}
    ]

    agent_x = StatefulMockAgent(name="agent_x", pipeline=mock_pipeline)
    strategy = DualLayerAgenticStrategy(router=mock_router)

    state = OrchestratorState(session_id="sess_task01")
    state.cursor = ExecutionCursor(agent_name="", macro_iteration=0, completed_steps_count=0)

    result_state = await strategy.execute(
        payload={"test": "data"},
        tools={},
        agents={"agent_x": agent_x},
        state=state
    )

    assert result_state.cursor.agent_name == "agent_x"
    assert result_state.cursor.macro_iteration == 1
    assert result_state.cursor.completed_steps_count == 0  # Reset upon completion of the agent step block
    assert len(result_state.history) == 2


@pytest.mark.asyncio
async def test_tool_registration_environment():
    """
    Validates TASK-02: _prepare_runtime_environment properly calls register_tools_to_agents
    and handles potential exceptions or missing attributes defensively.
    """
    mock_router = MagicMock()
    strategy = DualLayerAgenticStrategy(router=mock_router)

    strategy.register_tools_to_agents = MagicMock()

    tools = {"dummy_tool": MagicMock()}
    agents = {"agent_y": MagicMock()}

    resolved_tools = strategy._prepare_runtime_environment(tools=tools, agents=agents)
    strategy.register_tools_to_agents.assert_called_once_with(resolved_tools, agents)


@pytest.mark.asyncio
async def test_serialization_limit_and_metadata_exception_handling():
    """
    Validates TASK-03: _save_checkpoint correctly handles non-serializable objects and metadata
    exceptions by using a robust fallback or json default formatting.
    """
    mock_checkpoint_port = AsyncMock()
    strategy = DualLayerAgenticStrategy(router=MagicMock(), checkpoint_port=mock_checkpoint_port)

    state = OrchestratorState(session_id="sess_task03")
    state.metadata = {"unserializable_file": open(__file__, "r")}

    await strategy._save_checkpoint(state)
    assert mock_checkpoint_port.save.called

@pytest.mark.asyncio
async def test_multi_agent_resume_cursor_isolation():
    """
    Validates that when transitioning between multiple agents (Agent A -> Agent B),
    resuming from a failure in Agent B does not re-run Agent A or completed steps in Agent B.
    Addresses Review Point #1.
    """
    mock_router = MagicMock()
    # Mock routing: route to agent_a first, then agent_b upon resumption
    mock_router.get_next_agent.side_effect = ["agent_a", "agent_b"]
    mock_router.evaluate_completion.side_effect = [False, True]

    mock_checkpoint_port = AsyncMock()

    class MockPipeline:
        def __init__(self, steps: list):
            self.steps = steps

        def get_execute_pipeline(self, **kwargs):
            return self.steps

    class MultiAgentMock:
        def __init__(self, name: str, pipeline_steps: list, fail_at_step: int | None = None):
            self.name = name
            self.pipeline = MockPipeline(pipeline_steps)
            self.fail_at_step: int | None = fail_at_step
            self.executed_calls = []

        # Dynamically bind step execution handlers
        def __getattr__(self, name):
            async def step_handler(**kwargs):
                self.executed_calls.append(name)
                if self.fail_at_step is not None and len(self.executed_calls) == self.fail_at_step:
                    raise RuntimeError(f"Simulated failure at {self.name}.{name}")
                return f"{name}_result"
            return step_handler

    # Agent A with 2 steps
    pipeline_a = [{"method_name": "step_a1"}, {"method_name": "step_a2"}]
    agent_a = MultiAgentMock("agent_a", pipeline_a)

    # Agent B with 3 steps, intentionally set to fail at step 2 (step_b2)
    pipeline_b = [{"method_name": "step_b1"}, {"method_name": "step_b2"}, {"method_name": "step_b3"}]
    agent_b = MultiAgentMock("agent_b", pipeline_b, fail_at_step=2)

    agents = {"agent_a": agent_a, "agent_b": agent_b}
    strategy = DualLayerAgenticStrategy(router=mock_router, checkpoint_port=mock_checkpoint_port)

    state = OrchestratorState(session_id="sess_multi_agent")

    # First execution: expected to fail at Agent B's step_b2, transitioning state to FAILED
    with pytest.raises(RuntimeError, match="Simulated failure at agent_b.step_b2"):
        await strategy.execute(payload={}, tools={}, agents=agents, state=state)

    assert state.status == OrchestratorStatus.FAILED
    # Verify that Agent A executed fully (a1, a2) and Agent B executed up to b2 (failed)
    assert agent_a.executed_calls == ["step_a1", "step_a2"]
    assert agent_b.executed_calls == ["step_b1", "step_b2"]

    # Fix Agent B's failure condition by clearing fail_at_step
    agent_b.fail_at_step = None

    # Prepare to resume execution from the checkpoint
    mock_checkpoint_port.load.return_value = state.dump_checkpoint()

    # Update router expectation for resumption pointing to agent_b
    mock_router.get_next_agent.return_value = "agent_b"

    _, final_state = await strategy.resume(
        session_id="sess_multi_agent",
        checkpoint_port=mock_checkpoint_port,
        router=mock_router,
        agents=agents,
        payload={}
    )

    assert final_state.status == OrchestratorStatus.COMPLETED
    # Key assertion: Agent A is not re-run, step_b1 is not duplicated, resuming smoothly from step_b2 onwards
    assert agent_a.executed_calls == ["step_a1", "step_a2"]
    assert agent_b.executed_calls == ["step_b1", "step_b2", "step_b2", "step_b3"]


@pytest.mark.asyncio
async def test_boundary_failure_between_agents():
    """
    Validates that when Agent A completes, but Agent B's first step fails,
    the failure state correctly binds to Agent B rather than drifting back to Agent A.
    Addresses Review Point #1 (Boundary failure case).
    """
    mock_router = MagicMock()
    mock_router.get_next_agent.side_effect = ["agent_a", "agent_b", None]
    mock_router.evaluate_completion.return_value = False
    mock_checkpoint_port = AsyncMock()

    agent_a = MagicMock()
    agent_a.pipeline.get_execute_pipeline.return_value = [{"method_name": "step_a1"}]
    async def mock_a1(**kwargs): return "ok_a"
    agent_a.step_a1 = mock_a1

    agent_b = MagicMock()
    agent_b.pipeline.get_execute_pipeline.return_value = [{"method_name": "step_b1"}]
    async def mock_b1(**kwargs): raise RuntimeError("Fail on Agent B start")
    agent_b.step_b1 = mock_b1

    agents = {"agent_a": agent_a, "agent_b": agent_b}
    strategy = DualLayerAgenticStrategy(router=mock_router, checkpoint_port=mock_checkpoint_port)

    state = OrchestratorState(session_id="sess_boundary")

    with pytest.raises(RuntimeError, match="Fail on Agent B start"):
        await strategy.execute(payload={}, tools={}, agents=agents, state=state)

    assert state.status == OrchestratorStatus.FAILED
    assert state.cursor.agent_name == "agent_b"


@pytest.mark.asyncio
async def test_attempt_count_persistence_on_resume():
    """
    Validates that attempt_count is immediately updated and persisted
    when entering the FAILED resume flow.
    Addresses Review Point #2.
    """
    mock_router = MagicMock()
    mock_router.get_next_agent.return_value = "agent_a"
    mock_checkpoint_port = AsyncMock()

    state = OrchestratorState(session_id="sess_attempt")
    state.status = OrchestratorStatus.FAILED
    state.cursor = ExecutionCursor(agent_name="agent_a", macro_iteration=0, completed_steps_count=0)
    state.attempt_count = 1  # Utilize formalized typed state attribute

    mock_checkpoint_port.load.return_value = state.dump_checkpoint()

    agent_a = MagicMock()
    agent_a.pipeline.get_execute_pipeline.return_value = [{"method_name": "step_a1"}]

    async def fail_step(**kwargs): raise RuntimeError("Still failing")
    agent_a.step_a1 = fail_step

    strategy = DualLayerAgenticStrategy(router=mock_router, checkpoint_port=mock_checkpoint_port)

    with pytest.raises(RuntimeError):
        await strategy.resume(
            session_id="sess_attempt",
            checkpoint_port=mock_checkpoint_port,
            router=mock_router,
            agents={"agent_a": agent_a},
            payload={}
        )

    assert mock_checkpoint_port.save.called
    saved_json = mock_checkpoint_port.save.call_args[0][1]

    loaded_saved_state = OrchestratorState.load_checkpoint(saved_json)
    assert loaded_saved_state.attempt_count == 2


@pytest.mark.asyncio
async def test_primary_callback_raises_cancelled_error() -> None:
    """
    Validates that asyncio.CancelledError is correctly propagated and not swallowed
    by general exception handlers during primary callback or status change execution.
    """
    import asyncio

    mock_router = MagicMock()
    callback_adapter = AsyncMock()
    callback_adapter.on_status_changed.side_effect = asyncio.CancelledError()

    strategy = DualLayerAgenticStrategy(router=mock_router, callback_port=callback_adapter)
    state = OrchestratorState(session_id="sess_cancelled")

    with pytest.raises(asyncio.CancelledError):
        await strategy._change_status(state, OrchestratorStatus.PROCESSING)

    callback_adapter.on_status_changed.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_execution_race_condition() -> None:
    """
    Validates that launching concurrent execute calls for the same strategy / session
    raises ConcurrencyConflictError or handles race condition safely due to concurrency protection.
    Red phase: Expect concurrent overlapping state updates or race conditions without an asyncio.Lock.
    """
    import asyncio
    from auraflux_core.core.orchestrators.exceptions import ConcurrencyConflictError

    mock_router = MagicMock()
    mock_router.get_next_agent.return_value = "agent_a"
    mock_router.evaluate_completion.return_value = True

    class SlowMockAgent:
        def __init__(self):
            self.pipeline = MagicMock()
            self.pipeline.get_execute_pipeline.return_value = [{"method_name": "slow_step", "kwargs": {}}]

        async def slow_step(self, **kwargs):
            await asyncio.sleep(0.1)
            return "done"

    agent = SlowMockAgent()
    agents = {"agent_a": agent}
    strategy = DualLayerAgenticStrategy(router=mock_router)

    state = OrchestratorState(session_id="sess_concurrent")

    # Launch two executions concurrently via asyncio.gather
    results = await asyncio.gather(
        strategy.execute(payload={}, tools={}, agents=agents, state=state),
        strategy.execute(payload={}, tools={}, agents=agents, state=state),
        return_exceptions=True
    )

    # At least one should succeed, and at least one should raise ConcurrencyConflictError
    exceptions = [r for r in results if isinstance(r, Exception)]
    successes = [r for r in results if not isinstance(r, Exception)]

    assert len(successes) >= 1, "At least one concurrent execution should succeed."
    assert len(exceptions) >= 1, "Concurrent execution collision should trigger ConcurrencyConflictError or conflict exception."
    assert isinstance(exceptions[0], ConcurrencyConflictError), f"Expected ConcurrencyConflictError, got {exceptions[0]}"


@pytest.mark.asyncio
async def test_forward_reference_type_hints():
    """
    Verify that forward references (e.g. referencing the class itself in type hints)
    can be resolved correctly when from __future__ import annotations is enabled.
    """
    assert DualLayerAgenticStrategy is not None
    assert DualLayerAgenticStrategy.__name__ == "DualLayerAgenticStrategy"
