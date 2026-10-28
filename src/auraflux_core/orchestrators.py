from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from auraflux_core.core.orchestrators.base import BaseOrchestrator
from auraflux_core.core.orchestrators.router import SequentialAgentRouter
from auraflux_core.core.orchestrators.strategies.dual_layer_strategy import \
    DualLayerAgenticStrategy
from auraflux_core.core.schemas.orchestrators import OrchestratorConfig


@dataclass
class OrchestratorImplementation:
    orchestrator_class: Type[BaseOrchestrator]
    strategy_class: Type[Any]
    router_class: Type[Any]
    config_class: Type[OrchestratorConfig] = OrchestratorConfig

    def build_orchestrator(self, config: Optional[OrchestratorConfig] = None) -> BaseOrchestrator:
        if config is not None and not isinstance(config, OrchestratorConfig):
            raise ValueError("config must be an instance of OrchestratorConfig")

        if config is None:
            strategy = self.strategy_class(router=self.router_class())
            config = self.config_class(strategy=strategy)

        return self.orchestrator_class(config=config)

ORCHESTRATOR_REGISTRY: Dict[str, OrchestratorImplementation] = {
    "default": OrchestratorImplementation(
        orchestrator_class=BaseOrchestrator,
        strategy_class=DualLayerAgenticStrategy,
        router_class=SequentialAgentRouter,
        config_class=OrchestratorConfig
    )
}
