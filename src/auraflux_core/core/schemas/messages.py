from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str
    name: str
    token_usage: Optional[int] = 0
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Bucket for scenario-specific data (e.g., timestamps, confidence scores)."
    )
