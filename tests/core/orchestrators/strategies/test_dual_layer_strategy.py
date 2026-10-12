import pytest
from typing import Any
from unittest.mock import AsyncMock

from auraflux_core.core.orchestrators.state import OrchestratorState
from auraflux_core.core.orchestrators.strategies.dual_layer_strategy import DualLayerAgenticStrategy
from auraflux_core.core.orchestrators.router import SequentialAgentRouter


@pytest.mark.asyncio
async def test_dual_layer_strategy_full_execution(
    mock_orchestrator_state: OrchestratorState,
    dummy_pipeline_for_strategy: Any
) -> None:
    """
    Validates that DualLayerAgenticStrategy successfully resolves the router,
    retrieves execution steps from agent.pipeline.get_run_pipeline(), and dispatches them
    in complete isolation with fine-grained assertions.
    """
    class IsolatedMockAgent:
        def __init__(self, name: str, pipeline: Any):
            self.name = name
            self.pipeline = pipeline
            self.step_one = AsyncMock(return_value={"result": "step_one_completed"})
            self.step_two = AsyncMock(return_value={"result": "step_two_completed"})

    agent_a = IsolatedMockAgent(name="agent_a", pipeline=dummy_pipeline_for_strategy)

    agents_registry = {"agent_a": agent_a}

    # Limit max_steps=1 to ensure the routing loop executes exactly one iteration (yielding 2 steps)
    router = SequentialAgentRouter(max_steps=1)
    strategy = DualLayerAgenticStrategy(router=router)

    input_data = {"prompt": "Run isolated orchestration test"}
    tools = {}

    result_state = await strategy.execute(
        input_data=input_data,
        tools=tools,
        agents=agents_registry,
        state=mock_orchestrator_state,
    )

    # 1. Validate the overall history count
    assert len(result_state.history) == 2, f"Expected 2 execution steps in history, got {len(result_state.history)}"

    # 2. Fine-grained assertion: Validate telemetry attributes and outputs of step_one
    step_1 = result_state.history[0]
    assert step_1.actor_name == "agent_a"
    assert step_1.input_params.get("extra") == "val_1"
    assert step_1.input_params.get("payload") == input_data
    assert step_1.output_data == {"result": "step_one_completed"}
    assert step_1.error is None

    # 3. Fine-grained assertion: Validate telemetry attributes and outputs of step_two
    step_2 = result_state.history[1]
    assert step_2.actor_name == "agent_a"
    assert step_2.input_params.get("payload") == input_data
    assert step_2.output_data == {"result": "step_two_completed"}
    assert step_2.error is None

    # 4. Verify underlying agent methods were correctly and precisely awaited
    agent_a.step_one.assert_awaited_once()
    agent_a.step_two.assert_awaited_once()


@pytest.mark.asyncio
async def test_dual_layer_strategy_missing_pipeline_error(
    mock_orchestrator_state: OrchestratorState
) -> None:
    """
    Validates that Strategy properly raises AttributeError when a routed agent lacks a bound pipeline.
    """
    class PipelineLessAgent:
        def __init__(self, name: str):
            self.name = name

    bad_agent = PipelineLessAgent(name="bad_agent")
    agents_registry = {"bad_agent": bad_agent}
    router = SequentialAgentRouter(max_steps=1)
    strategy = DualLayerAgenticStrategy(router=router)

    with pytest.raises(AttributeError) as exc_info:
        await strategy.execute(
            input_data={"prompt": "test"},
            tools={},
            agents=agents_registry,
            state=mock_orchestrator_state,
        )

    # Fine-grained assertion: Ensure the error message precisely matches expectations
    assert "does not have a valid bound pipeline" in str(exc_info.value)
