from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OrchestratorConfig(BaseModel):
    """Configuration for an orchestrator."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    strategy: Optional[Any] = None
    tools: Optional[List[Any]] = None
    agents: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = Field(default=30.0, description="Workflow execution timeout in seconds.")
    max_retries: Optional[int] = Field(default=3, description="Maximum number of retries.")

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("timeout must be greater than 0")
        return v

    @field_validator("max_retries")
    @classmethod
    def validate_max_retries(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("max_retries must be greater than or equal to 0")
        return v
