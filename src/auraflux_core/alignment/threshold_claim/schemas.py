from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from auraflux_core.alignment.schemas import TripleItem


class NormalizedMetric(BaseModel):
    """Gate 2: Specific for quantitative threshold normalization."""
    raw_text: str = Field(..., description="Raw quantitative expression, e.g., '500萬TWD'")
    metric_name: str = Field(..., description="Name of the variable, e.g., 'budget'")
    normalized_value: float = Field(..., description="Parsed numeric value, e.g., 5000000.0")
    unit: str = Field(..., description="Standardized unit, e.g., 'TWD', 'days'")
    operator: Literal["<=", ">=", "==", "<", ">"] = Field(..., description="Comparison operator")


class ThresholdDiagnosticAnalysis(BaseModel):
    """Orthogonal diagnostic dimensions for threshold claims."""
    implicit_premises: List[str] = Field(
        default_factory=list,
        description="Gate 1: Premises like 'external environment remains static'."
    )
    quantification_requirements: List[NormalizedMetric] = Field(
        default_factory=list,
        description="Gate 2 (Main Battlefield): Formatted & normalized quantitative metrics."
    )
    boundary_conflicts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Gate 3 (Main Battlefield): Boundary violations, dual-metric paradoxes."
    )


class ThresholdClaimVerdict(BaseModel):
    """Verification payload for quantitative threshold claims."""
    proposition_id: str = Field(..., description="Unique identifier for the atomic claim.")
    claim_text: str = Field(..., description="The original atomic claim statement.")
    triples: List[TripleItem] = Field(
        default_factory=list,
        description="Bound semantic triples (e.g. Budget -> <= -> 5000000 TWD)."
    )
    diagnostics: ThresholdDiagnosticAnalysis = Field(..., description="Orthogonal diagnostic analysis.")
    status: Literal["PASS", "FAIL", "PARADOX", "BORDERLINE"] = Field(
        ...,
        description="PASS: Met boundary; FAIL/PARADOX: Conflict/Out-of-bounds (BLOCK); BORDERLINE: Vague value."
    )
    preset_options: List[str] = Field(
        default_factory=list,
        description="Suggested human trade-off choices if BORDERLINE or PARADOX triggers a BLOCK."
    )
    compliance_gap: Optional[str] = Field(
        default=None,
        description="Detailed explanation of the numeric paradox or boundary overflow."
    )
