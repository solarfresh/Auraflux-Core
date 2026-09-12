from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, Field, field_validator

PremiseT = TypeVar("PremiseT")
QuantT = TypeVar("QuantT")

ClaimStatus = Literal[
    "VERIFIED",
    "PARTIALLY_VERIFIED",
    "UNSUPPORTED",
    "VIOLATED",
    "FAIL",
    "BORDERLINE"
]


class ClaimVerdict(BaseModel):
    """Shared verdict envelope produced by every diagnostic agent
    (mental_model / objective / threshold). Downstream cross-checking
    logic (Step 3) consumes this common shape uniformly, regardless
    of which agent produced it.

    Subclasses MUST add their own `diagnostics` field (structure differs
    per agent's main-battlefield gate) and MAY add agent-specific fields
    (e.g. threshold's `preset_options`).
    """

    proposition_id: str = Field(..., description="Unique identifier for the atomic claim.")
    claim_text: str = Field(..., description="The original atomic claim statement.")
    triples: List[TripleItem] = Field(
        default_factory=list,
        description="Bound semantic triples representing the structured facts of the claim."
    )
    status: ClaimStatus = Field(..., description="Final verification state shared across all agents.")
    verification_proofs: List[str] = Field(
        default_factory=list,
        description="Extracted proof points, citations, or references supporting the verdict."
    )
    compliance_gap: Optional[str] = Field(
        default=None,
        description="Gap narrative when status indicates a violation, block, or unsupported claim."
    )


class DiagnosticAnalysisBase(BaseModel, Generic[PremiseT, QuantT]):
    """Shared 3-gate diagnostic contract. Concrete agents only parametrize
    the two fields that may be upgraded into a strongly-typed 'main
    battlefield' structure; boundary_conflicts stays free-form across
    all agents since none of them have formalized its shape yet.
    """

    implicit_premises: PremiseT
    quantification_requirements: QuantT
    boundary_conflicts: Dict[str, Any] = Field(default_factory=dict)


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
