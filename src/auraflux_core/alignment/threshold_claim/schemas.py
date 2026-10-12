from typing import List, Literal, Optional

from pydantic import BaseModel, Field, NonNegativeInt

from auraflux_core.alignment.schemas import (ClaimStatus, ClaimVerdict,
                                             DiagnosticAnalysisBase)
from auraflux_core.alignment.threshold_claim.constants import (
    DEFAULT_THRESHOLD_CLAIM_AGENT_PLAN_STAGE_OUTPUT_SCHEMA,
    DEFAULT_THRESHOLD_CLAIM_AGENT_PLAN_TEMPLATE,
    DEFAULT_THRESHOLD_CLAIM_AGENT_SYNTHESIS_STAGE_OUTPUT_SCHEMA,
    DEFAULT_THRESHOLD_CLAIM_AGENT_SYNTHESIS_TEMPLATE,
    DEFAULT_THRESHOLD_CLAIM_AGENT_SYSTEM_PROMPT_EN,
    DEFAULT_THRESHOLD_CLAIM_AGENT_SYSTEM_PROMPT_ZH)
from auraflux_core.core.schemas.agents import AgentConfig
from auraflux_core.core.schemas.messages import (OutputFormat, PromptConfig,
                                                 StageSpec, SystemPromptSpec)


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


class ThresholdClaimAgentConfig(AgentConfig):
    """
    Dedicated AgentConfig for ExtractKeywordsAgent with specialized domain defaults.
    """
    output_format: Literal['TEXT', 'JSON'] = 'JSON'
    pipeline_name: Literal['direct', 'plan_and_execute'] = 'plan_and_execute'
    prompt_config: PromptConfig = Field(
        default_factory=lambda: PromptConfig(
            system=SystemPromptSpec(
                templates={
                    'zh': DEFAULT_THRESHOLD_CLAIM_AGENT_SYSTEM_PROMPT_ZH,
                    'en': DEFAULT_THRESHOLD_CLAIM_AGENT_SYSTEM_PROMPT_EN,
                    'default': DEFAULT_THRESHOLD_CLAIM_AGENT_SYSTEM_PROMPT_EN
                }),
            stages={
                'plan': StageSpec(
                    templates={'default': DEFAULT_THRESHOLD_CLAIM_AGENT_PLAN_TEMPLATE},
                    template_variables={
                        "proposition_id": "The unique identifier for the target proposition or claim being analyzed.",
                        "claim_text": "The raw input text or statement describing the subjective expectation or proposition.",
                    },
                    output_format=OutputFormat.JSON,
                    output_schema=DEFAULT_THRESHOLD_CLAIM_AGENT_PLAN_STAGE_OUTPUT_SCHEMA,
                ),
                'synthesis': StageSpec(
                    templates={'default': DEFAULT_THRESHOLD_CLAIM_AGENT_SYNTHESIS_TEMPLATE},
                    template_variables={
                        "proposition_id": "The unique identifier for the proposition or claim being analyzed.",
                        "claim_text": "The raw input text or statement describing the subjective expectation.",
                        "quantification_requirements": "The JSON or string representation of Stage 1 quantification requirements.",
                        "calc_summary": "The summary result of the deterministic calculator computation.",
                        "formatted_evidence_text": "The formatted text containing retrieved structured core context evidence.",
                    },
                    output_format=OutputFormat.JSON,
                    output_schema=DEFAULT_THRESHOLD_CLAIM_AGENT_SYNTHESIS_STAGE_OUTPUT_SCHEMA,
                )
            }
        )
    )

    max_tokens: NonNegativeInt= 32768
    temperature: float = 0.0
