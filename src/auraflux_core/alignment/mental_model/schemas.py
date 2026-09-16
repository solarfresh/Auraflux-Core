from typing import Any, Dict, List

from pydantic import BaseModel, Field

from auraflux_core.alignment.schemas import (ClaimVerdict,
                                             DiagnosticAnalysisBase)


class FourDimensionAssumptionMatrix(BaseModel):
    """Gate 1 (main battlefield): implicit assumption 4D matrix."""

    capability_assumptions: List[str] = Field(
        default_factory=list,
        description="Unverified assumptions about the subject/counterparty's technical or execution capability."
    )
    resource_assumptions: List[str] = Field(
        default_factory=list,
        description="Unverified assumptions about resource allocation, budget, or authorization scope."
    )
    causality_assumptions: List[str] = Field(
        default_factory=list,
        description="Unexamined links in the causal chain taken for granted."
    )
    environment_assumptions: List[str] = Field(
        default_factory=list,
        description="Unverified assumptions that the external environment (market, regulation, counterparty org) stays static."
    )


class MentalModelDiagnosticAnalysis(DiagnosticAnalysisBase[FourDimensionAssumptionMatrix, Dict[str, Any]]):
    pass


class MentalModelClaimVerdict(ClaimVerdict):
    """Verification payload for subjective mental-model (expectation/belief) claims."""

    diagnostics: MentalModelDiagnosticAnalysis = Field(..., description="Orthogonal diagnostic analysis.")
