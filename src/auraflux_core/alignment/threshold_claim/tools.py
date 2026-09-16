from typing import Any, Dict, List, Optional, Tuple

from auraflux_core.alignment.threshold_claim.schemas import (
    MetricComparisonResult, NormalizedMetric, ThresholdCalculatorInput,
    ThresholdCalculatorOutput)
from auraflux_core.core.schemas.tools import ToolConfig
from auraflux_core.core.tools.base_tool import BaseTool


class ThresholdCalculatorTool(BaseTool):
    """
    Pure deterministic calculation tool for threshold verification.
    Accepts explicit claim_metrics and baseline_metrics lists, decoupled from DB/storage schemas.
    """

    def __init__(self, config: ToolConfig = ToolConfig()) -> None:
        super().__init__(config=config)

    async def run(
        self,
        claim_metrics: List[Dict[str, Any]],
        baseline_metrics: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes zero-hallucination python boundary checks comparing claim metrics against baseline metrics.
        Implements BaseTool.run() abstract method.
        """
        self.logger.info(
            f"Executing calculation: {len(claim_metrics)} claim metrics against {len(baseline_metrics)} baseline metrics."
        )

        claims: List[NormalizedMetric] = [
            m if isinstance(m, NormalizedMetric) else NormalizedMetric(**m)
            for m in claim_metrics
        ]

        baselines: List[NormalizedMetric] = [
            m if isinstance(m, NormalizedMetric) else NormalizedMetric(**m)
            for m in baseline_metrics
        ]

        # Convert baselines to map for rapid O(1) matching by metric_name
        baseline_map: Dict[str, NormalizedMetric] = {
            b.metric_name.lower(): b for b in baselines
        }

        comparisons: List[MetricComparisonResult] = []
        unprocessed_warnings: List[str] = []
        has_failure = False
        has_paradox = False
        has_unprocessed = False

        for claim in claims:
            baseline = baseline_map.get(claim.metric_name.lower())

            if not baseline:
                warning_msg = (
                    f"Metric '{claim.metric_name}': Baseline reference metric missing in input baseline_metrics."
                )
                unprocessed_warnings.append(warning_msg)
                has_unprocessed = True
                comparisons.append(
                    MetricComparisonResult(
                        metric_name=claim.metric_name,
                        claimed_value=claim.normalized_value,
                        baseline_value=None,
                        unit=claim.unit,
                        operator=claim.operator,
                        is_satisfied=None,
                        overflow_value=0.0,
                        note="Baseline metric not provided for comparison."
                    )
                )
                continue

            if baseline.unit.upper() != claim.unit.upper():
                warning_msg = (
                    f"Metric '{claim.metric_name}': Unit mismatch between claim ('{claim.unit}') "
                    f"and baseline ('{baseline.unit}'). Converter unavailable; comparison skipped."
                )
                unprocessed_warnings.append(warning_msg)
                has_unprocessed = True
                comparisons.append(
                    MetricComparisonResult(
                        metric_name=claim.metric_name,
                        claimed_value=claim.normalized_value,
                        baseline_value=baseline.normalized_value,
                        unit=f"{claim.unit} vs {baseline.unit}",
                        operator=claim.operator,
                        is_satisfied=None,
                        overflow_value=0.0,
                        note="Unit mismatch cannot be converted automatically."
                    )
                )
                continue

            # Pure Python deterministic boundary check
            satisfied, overflow, err_note = self._evaluate_boundary(
                claimed_val=claim.normalized_value,
                baseline_val=baseline.normalized_value,
                operator=claim.operator
            )

            if err_note:
                has_paradox = True
            if satisfied is False:
                has_failure = True

            comparisons.append(
                MetricComparisonResult(
                    metric_name=claim.metric_name,
                    claimed_value=claim.normalized_value,
                    baseline_value=baseline.normalized_value,
                    unit=claim.unit,
                    operator=claim.operator,
                    is_satisfied=satisfied,
                    overflow_value=overflow,
                    note=err_note or ("Condition satisfied." if satisfied else f"Boundary violated. Overflow: {overflow}")
                )
            )

        # 全新的狀態對齊邏輯 (VERIFIED / VIOLATED / PARTIALLY_VERIFIED / UNSUPPORTED)
        if len(claims) > 0 and len(baselines) == 0:
            status = "UNSUPPORTED"
        elif has_failure:
            status = "VIOLATED"
        elif has_paradox or has_unprocessed:
            status = "PARTIALLY_VERIFIED"
        else:
            status = "VERIFIED"

        output = ThresholdCalculatorOutput(
            status=status,
            comparisons=comparisons,
            unprocessed_warnings=unprocessed_warnings
        )

        return output.model_dump()

    def _evaluate_boundary(
        self, claimed_val: float, baseline_val: float, operator: str
    ) -> Tuple[Optional[bool], float, str]:
        """Evaluates boundary mathematical rules using pure python execution."""
        try:
            if operator == "<=":
                is_satisfied = claimed_val <= baseline_val
                overflow = max(0.0, claimed_val - baseline_val)
            elif operator == "<":
                is_satisfied = claimed_val < baseline_val
                overflow = max(0.0, claimed_val - baseline_val)
            elif operator == ">=":
                is_satisfied = claimed_val >= baseline_val
                overflow = max(0.0, baseline_val - claimed_val)
            elif operator == ">":
                is_satisfied = claimed_val > baseline_val
                overflow = max(0.0, baseline_val - claimed_val)
            elif operator == "==":
                is_satisfied = claimed_val == baseline_val
                overflow = abs(claimed_val - baseline_val)
            else:
                return None, 0.0, f"Unsupported comparison operator '{operator}'."

            return is_satisfied, overflow, ""
        except Exception as e:
            return None, 0.0, f"Mathematical evaluation error: {str(e)}"

    def get_name(self) -> str:
        return "threshold_calculator"

    def get_description(self) -> str:
        return (
            "Performs zero-hallucination deterministic boundary checks given claim_metrics and baseline_metrics lists."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return ThresholdCalculatorInput.model_json_schema()
