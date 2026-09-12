from typing import Optional

from pydantic import BaseModel, Field, field_validator


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

    @field_validator("normalized_value", mode="before")
    @classmethod
    def parse_empty_float(cls, v):
        if v == "" or v is None or v == "null":
            return None
        return float(v)

    @field_validator("metric_name", "unit", "operator", mode="before")
    @classmethod
    def parse_empty_string(cls, v):
        if v == "" or v == "null":
            return None
        return v
