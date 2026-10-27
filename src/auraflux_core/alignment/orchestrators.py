from typing import Any, Dict, List, Optional

from auraflux_core.alignment.strategies import \
    AlignmentOrchestrationStrategy
from auraflux_core.core.orchestrators.base import BaseOrchestrator
from auraflux_core.core.schemas.orchestrators import OrchestratorConfig
from auraflux_core.core.tools.base_tool import BaseTool


class AlignmentOrchestrator(BaseOrchestrator):
    """
    Dedicated Orchestrator for managing multi-agent alignment workflows.
    """
    def __init__(
        self,
        config: OrchestratorConfig
    ):
        if config.strategy is None:
            config.strategy = AlignmentOrchestrationStrategy()
        super().__init__(config=config)
