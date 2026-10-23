from unittest.mock import MagicMock

import pytest

from auraflux_core.core.orchestrators.base import BaseOrchestrator
from auraflux_core.core.orchestrators.state import (OrchestratorState,
                                                    OrchestratorStatus)
from auraflux_core.core.orchestrators.strategies.base import \
    OrchestrationStrategy


class MockStrategy(OrchestrationStrategy):
    async def execute(self, input_data, tools, agents, state):
        return state

@pytest.mark.asyncio
async def test_base_orchestrator_initialization():
    strategy = MockStrategy()
    tools = []
    agents = {"agent1": MagicMock()}

    orchestrator = BaseOrchestrator(strategy=strategy, tools=tools, agents=agents)
    assert orchestrator.strategy == strategy
    assert orchestrator.agents == agents

@pytest.mark.asyncio
async def test_base_orchestrator_invalid_agents():
    strategy = MockStrategy()
    tools = []
    agents = ["invalid_type"]  # type: ignore

    with pytest.raises(TypeError, match="Agents must be a dictionary"):
        BaseOrchestrator(strategy=strategy, tools=tools, agents=agents)  # type: ignore

@pytest.mark.asyncio
async def test_base_orchestrator_run_success():
    strategy = MockStrategy()
    tools = []
    agents = {"agent1": MagicMock()}

    orchestrator = BaseOrchestrator(strategy=strategy, tools=tools, agents=agents)
    state = await orchestrator.run(input_data={"task": "do something"})

    assert orchestrator.state.status == OrchestratorStatus.COMPLETED
