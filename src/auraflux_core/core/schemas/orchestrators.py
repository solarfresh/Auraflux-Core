from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class OrchestratorConfig(BaseModel):
    """Configuration for an orchestrator."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    strategy: Optional[Any] = None
    tools: Optional[List[Any]] = None
    agents: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = Field(default=30.0, description="Workflow execution timeout in seconds.")
