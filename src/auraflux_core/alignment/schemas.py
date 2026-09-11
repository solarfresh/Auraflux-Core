from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class TripleItem(BaseModel):
    """Expresses a bound semantic triple: Subject -> Predicate -> Object."""

    subject: str = Field(..., description="Subject entity.")
    predicate: str = Field(..., description="Relation, predicate, or operator.")
    object: str = Field(..., description="Metric, constraint, or object entity.")

