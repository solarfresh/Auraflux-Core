from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from auraflux_core.alignment.schemas import (ClaimStatus, ClaimVerdict,
                                             DiagnosticAnalysisBase)


class NormalizedMetric(BaseModel):
    """Gate 2: Specific for quantitative threshold normalization."""
    raw_text: str = Field(..., description="Raw quantitative expression, e.g., '500萬TWD'")
    metric_name: str = Field(..., description="Name of the variable, e.g., 'budget'")
    normalized_value: float = Field(..., description="Parsed numeric value, e.g., 5000000.0")
    unit: str = Field(..., description="Standardized unit, e.g., 'TWD', 'days'")
    operator: Literal["<=", ">=", "==", "<", ">"] = Field(..., description="Comparison operator")


class ThresholdDiagnosticAnalysis(DiagnosticAnalysisBase[List[str], List[NormalizedMetric]]):
    pass


class ThresholdClaimVerdict(ClaimVerdict):
    """Verification payload for quantitative threshold claims."""

    diagnostics: ThresholdDiagnosticAnalysis = Field(..., description="Orthogonal diagnostic analysis.")
    preset_options: List[str] = Field(
        default_factory=list,
        description="Suggested human trade-off choices when status is BORDERLINE or a paradox triggers BLOCK."
    )


class ThresholdCalculatorInput(BaseModel):
    """Input payload for threshold_calculator tool."""
    claim_metrics: List[NormalizedMetric] = Field(
        ...,
        description="List of target metrics extracted from the user claim."
    )
    baseline_metrics: List[NormalizedMetric] = Field(
        ...,
        description="List of baseline metrics parsed from database/policy context."
    )


class MetricComparisonResult(BaseModel):
    """Evaluation result for an individual metric comparison."""
    metric_name: str = Field(..., description="Name of the evaluated metric.")
    claimed_value: float = Field(..., description="Numeric value presented in the claim.")
    baseline_value: Optional[float] = Field(None, description="Extracted baseline limit/threshold from policy.")
    unit: str = Field(..., description="Unit of the metric.")
    operator: str = Field(..., description="Comparison operator used (<=, >=, ==, <, >).")
    is_satisfied: Optional[bool] = Field(None, description="True if within limits, False if violated, None if uncomparable.")
    overflow_value: float = Field(0.0, description="Absolute variance or overflow amount if boundary violated.")
    note: str = Field("", description="Detailed explanation of the calculation outcome.")


class ThresholdCalculatorOutput(BaseModel):
    """Output results produced by threshold_calculator tool."""
    status: ClaimStatus = Field(
        ...,
        description="Overall deterministic calculation status."
    )
    comparisons: List[MetricComparisonResult] = Field(
        default_factory=list,
        description="Detailed comparison breakdown for each metric."
    )
    unprocessed_warnings: List[str] = Field(
        default_factory=list,
        description="Warnings regarding mismatched units or unparseable context values that require human/trade-off handling."
    )
