from typing import Optional
from pydantic import BaseModel, Field


class TripleItem(BaseModel):
    """
    Expresses a bound semantic triple: Subject -> Predicate -> Object,
    optionally enriched with normalized quantitative metric metadata.
    Provides a closed-world statement ensuring entities, logical conditions, and quantities remain coupled.
    """

    subject: str = Field(..., description="Subject entity")
    predicate: str = Field(..., description="Relation / Predicate / Operator")
    object: str = Field(..., description="Metric / Constraint / Object entity")

    # Embedded Quantitative Metric Normalization Properties
    metric_name: Optional[str] = Field(
        None,
        description="Standardized metric identifier (e.g., 'budget', 'duration', 'sla_time', 'penalty')"
    )
    normalized_value: Optional[float] = Field(
        None,
        description="Pure numeric value converted strictly to the standard base unit"
    )
    unit: Optional[str] = Field(
        None,
        description="Standard unit symbol (e.g., 'TWD', 'USD', 'day', 'hour', 'm2', 'kg', 'GB', '%')"
    )
    operator: Optional[str] = Field(
        None,
        description="Boundary condition operator strictly chosen from ['<=', '>=', '==', '<', '>']"
    )
