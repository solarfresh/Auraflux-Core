from typing import Any, Dict, List

from pydantic import Field

from auraflux_core.alignment.schemas import (ClaimVerdict,
                                             DiagnosticAnalysisBase)


class ObjectiveDiagnosticAnalysis(DiagnosticAnalysisBase[List[str], Dict[str, Any]]):
    pass


class ObjectiveClaimVerdict(ClaimVerdict):
    """Verification payload for objective, testable claims."""

    diagnostics: ObjectiveDiagnosticAnalysis = Field(..., description="Orthogonal diagnostic analysis.")
