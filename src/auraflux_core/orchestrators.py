from typing import Any, Dict, NamedTuple, Type

from auraflux_core.core.orchestrators.base import BaseOrchestrator
from auraflux_core.core.orchestrators.router import SequentialAgentRouter
from auraflux_core.core.orchestrators.strategies.dual_layer_strategy import \
    DualLayerAgenticStrategy


class OrchestratorImplementation(NamedTuple):
    orchestrator_class: Type[BaseOrchestrator]
    strategy_class: Type[Any]
    router_class: Type[Any]

ORCHESTRATOR_REGISTRY: Dict[str, OrchestratorImplementation] = {
    "default": OrchestratorImplementation(
        orchestrator_class=BaseOrchestrator,
        strategy_class=DualLayerAgenticStrategy,
        router_class=SequentialAgentRouter
    )
}
