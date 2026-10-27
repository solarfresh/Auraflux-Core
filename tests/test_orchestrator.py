import pytest

from auraflux_core.core.orchestrators.base import BaseOrchestrator
from auraflux_core.core.orchestrators.router import SequentialAgentRouter
from auraflux_core.core.orchestrators.strategies.dual_layer_strategy import \
    DualLayerAgenticStrategy
from auraflux_core.core.schemas.orchestrators import OrchestratorConfig
from auraflux_core.orchestrators import OrchestratorImplementation


@pytest.fixture
def orchestrator_impl():
    return OrchestratorImplementation(
        orchestrator_class=BaseOrchestrator,
        strategy_class=DualLayerAgenticStrategy,
        router_class=SequentialAgentRouter
    )

@pytest.fixture
def custom_config():
    strategy = DualLayerAgenticStrategy(router=SequentialAgentRouter())
    return OrchestratorConfig(strategy=strategy, timeout=20.0, max_retries=3)

def test_build_orchestrator_default(orchestrator_impl):
    # Execute test
    orchestrator = orchestrator_impl.build_orchestrator()
    # Assertions
    assert orchestrator is not None
    assert orchestrator.config is not None
    # Assuming default values
    assert orchestrator.config.timeout > 0

def test_build_orchestrator_custom(orchestrator_impl, custom_config):
    # Execute test
    orchestrator = orchestrator_impl.build_orchestrator(config=custom_config)
    # Assertions
    assert orchestrator.config.timeout == 20.0
    assert orchestrator.config.max_retries == 3

def test_build_orchestrator_invalid_config_type(orchestrator_impl):
    # Execute test and verify exception
    with pytest.raises(ValueError):
        orchestrator_impl.build_orchestrator(config="not_a_config_instance")  # type: ignore
