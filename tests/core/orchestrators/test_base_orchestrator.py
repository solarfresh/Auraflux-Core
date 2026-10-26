from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from auraflux_core.core.orchestrators.base import BaseOrchestrator
from auraflux_core.core.orchestrators.state import (OrchestratorState,
                                                    OrchestratorStatus)
from auraflux_core.core.orchestrators.strategies.base import \
    OrchestrationStrategy
from auraflux_core.core.schemas.orchestrators import OrchestratorConfig


class MockStrategy(OrchestrationStrategy):
    async def execute(self, input_data, tools, agents, state):
        return state

@pytest.mark.asyncio
async def test_base_orchestrator_initialization():
    strategy = MockStrategy()
    tools = []
    agents = {"agent1": MagicMock()}
    config = OrchestratorConfig(timeout=15.0, max_retries=2)

    orchestrator = BaseOrchestrator(strategy=strategy, tools=tools, agents=agents, config=config)
    assert orchestrator.strategy == strategy
    assert orchestrator.agents == agents
    assert orchestrator.config.timeout == 15.0
    assert orchestrator.config.max_retries == 2

@pytest.mark.asyncio
async def test_orchestrator_config_validation():
    # Test valid config
    config = OrchestratorConfig(timeout=10.0, max_retries=1)
    assert config.timeout == 10.0
    assert config.max_retries == 1

    # Test invalid timeout (<= 0)
    with pytest.raises(ValidationError):
        OrchestratorConfig(timeout=0)

    with pytest.raises(ValidationError):
        OrchestratorConfig(timeout=-5.0)

    # Test invalid max_retries (< 0)
    with pytest.raises(ValidationError):
        OrchestratorConfig(max_retries=-1)

    # Test extra forbidden fields
    with pytest.raises(ValidationError):
        OrchestratorConfig(unknown_field="not_allowed")  # type: ignore

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
    state = await orchestrator.execute(input_data={"task": "do something"})

    assert orchestrator.state.status == OrchestratorStatus.COMPLETED

