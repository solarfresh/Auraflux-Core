from typing import Any, Dict, List, Literal

from pydantic import Field, NonNegativeInt

from auraflux_core.alignment.objective_claim.constants import (
    DEFAULT_OBJECTIVE_CLAIM_AGENT_PLAN_STAGE_OUTPUT_SCHEMA,
    DEFAULT_OBJECTIVE_CLAIM_AGENT_PLAN_TEMPLATE,
    DEFAULT_OBJECTIVE_CLAIM_AGENT_SYNTHESIS_STAGE_OUTPUT_SCHEMA,
    DEFAULT_OBJECTIVE_CLAIM_AGENT_SYNTHESIS_TEMPLATE,
    DEFAULT_OBJECTIVE_CLAIM_AGENT_SYSTEM_PROMPT_EN,
    DEFAULT_OBJECTIVE_CLAIM_AGENT_SYSTEM_PROMPT_ZH)
from auraflux_core.alignment.schemas import (ClaimVerdict,
                                             DiagnosticAnalysisBase)
from auraflux_core.core.schemas.agents import AgentConfig
from auraflux_core.core.schemas.messages import (OutputFormat, PromptConfig,
                                                 StageSpec, SystemPromptSpec)


class ObjectiveDiagnosticAnalysis(DiagnosticAnalysisBase[List[str], Dict[str, Any]]):
    pass


class ObjectiveClaimVerdict(ClaimVerdict):
    """Verification payload for objective, testable claims."""

    diagnostics: ObjectiveDiagnosticAnalysis = Field(..., description="Orthogonal diagnostic analysis.")


class ObjectiveClaimAgentConfig(AgentConfig):
    """
    Dedicated AgentConfig for ExtractKeywordsAgent with specialized domain defaults.
    """
    output_format: Literal['TEXT', 'JSON'] = 'JSON'
    pipeline_name: Literal['direct', 'plan_and_execute'] = 'plan_and_execute'
    prompt_config: PromptConfig = Field(
        default_factory=lambda: PromptConfig(
            system=SystemPromptSpec(
                templates={
                    'zh': DEFAULT_OBJECTIVE_CLAIM_AGENT_SYSTEM_PROMPT_ZH,
                    'en': DEFAULT_OBJECTIVE_CLAIM_AGENT_SYSTEM_PROMPT_EN,
                    'default': DEFAULT_OBJECTIVE_CLAIM_AGENT_SYSTEM_PROMPT_EN
                }),
            stages={
                'plan': StageSpec(
                    templates={'default': DEFAULT_OBJECTIVE_CLAIM_AGENT_PLAN_TEMPLATE},
                    template_variables={
                        "proposition_id": "The unique identifier for the target proposition or claim being analyzed.",
                        "claim_text": "The raw input text or statement describing the subjective expectation or proposition.",
                    },
                    output_format=OutputFormat.JSON,
                    output_schema=DEFAULT_OBJECTIVE_CLAIM_AGENT_PLAN_STAGE_OUTPUT_SCHEMA,
                ),
                'synthesis': StageSpec(
                    templates={'default': DEFAULT_OBJECTIVE_CLAIM_AGENT_SYNTHESIS_TEMPLATE},
                    template_variables={
                        "proposition_id": "The unique identifier for the proposition or claim being analyzed.",
                        "claim_text": "The raw input text or statement describing the subjective expectation.",
                        "triples": "The JSON or string representation of extracted semantic triples from Stage 1.",
                        "diagnostics": "The JSON or string representation of Stage 1 diagnostics (implicit premises, quantification, conflicts).",
                        "formatted_evidence_text": "The formatted text containing retrieved structured core context evidence.",
                    },
                    output_format=OutputFormat.JSON,
                    output_schema=DEFAULT_OBJECTIVE_CLAIM_AGENT_SYNTHESIS_STAGE_OUTPUT_SCHEMA,
                )
            }
        )
    )

    max_tokens: NonNegativeInt= 32768
    temperature: float = 0.0
