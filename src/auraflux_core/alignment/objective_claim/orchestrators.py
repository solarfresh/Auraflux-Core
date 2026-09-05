from typing import Any, Dict, List, Optional

from auraflux_core.alignment.objective_claim.strategies import \
    AlignmentOrchestrationStrategy
from auraflux_core.core.orchestrators.base import BaseOrchestrator
from auraflux_core.core.tools.base_tool import BaseTool


class AlignmentOrchestrator(BaseOrchestrator):
    """
    Dedicated Orchestrator for managing multi-agent alignment workflows.
    """
    def __init__(
        self,
        agents: Dict[str, Any],
        tools: Optional[List[BaseTool]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        strategy = AlignmentOrchestrationStrategy()
        super().__init__(
            strategy=strategy,
            tools=tools or [],
            agents=agents,
            config=config
        )
