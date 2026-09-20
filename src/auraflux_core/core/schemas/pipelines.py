from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """
    Represents the output of the inspection/validation step (Stage 1.5).
    Used by PlanAndExecutePipeline to determine if execution should proceed
    or if replanning/error handling is required.
    """
    is_valid: bool = Field(
        default=True,
        description="Indicates whether the plan output passed validation checks."
    )
    reasons: List[str] = Field(
        default_factory=list,
        description="List of error messages, missing fields, or validation failure details."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context, structured tool raw outputs, or diagnostic metadata."
    )
