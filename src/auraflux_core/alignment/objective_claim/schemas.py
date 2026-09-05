from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class TripleItem(BaseModel):
    """Expresses a bound semantic triple: Subject -> Predicate -> Object."""

    subject: str = Field(..., description="Subject entity.")
    predicate: str = Field(..., description="Relation, predicate, or operator.")
    object: str = Field(..., description="Metric, constraint, or object entity.")


class DiagnosticAnalysis(BaseModel):
    """Orthogonal diagnostic dimensions for objective claims."""

    implicit_premises: List[str] = Field(
        default_factory=list,
        description="Implicit operational or contextual premises required for the claim to hold."
    )
    quantification_requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Specifications for audit criteria and necessary verification artifacts."
    )
    boundary_conflicts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Conflict detection results against baseline policies or historical records."
    )


class ObjectiveClaimVerdict(BaseModel):
    """Generic verification payload for an individual objective claim."""

    proposition_id: str = Field(..., description="Unique identifier for the atomic claim.")
    claim_text: str = Field(..., description="The original atomic claim statement.")
    triples: List[TripleItem] = Field(
        default_factory=list,
        description="Bound semantic triples representing the structured facts of the claim."
    )
    diagnostics: DiagnosticAnalysis = Field(..., description="Orthogonal diagnostic analysis.")
    status: Literal["VERIFIED", "PARTIALLY_VERIFIED", "UNSUPPORTED"] = Field(
        ...,
        description="Final verification state. UNSUPPORTED triggers a block condition."
    )
    verification_proofs: List[str] = Field(
        default_factory=list,
        description="Extracted proof points, citations, or references supporting the verdict."
    )
    compliance_gap: Optional[str] = Field(
        default=None,
        description="Detailed explanation if the claim is UNSUPPORTED or PARTIALLY_VERIFIED."
    )
