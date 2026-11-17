from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, NonNegativeInt

from auraflux_core.canvases.decomposition.constants import (
    DEFAULT_DECOMPOSITION_AGENT_PLAN_REFINE_TEMPLATE,
    DEFAULT_DECOMPOSITION_AGENT_PLAN_STAGE_OUTPUT_SCHEMA,
    DEFAULT_DECOMPOSITION_AGENT_PLAN_TEMPLATE,
    DEFAULT_DECOMPOSITION_AGENT_SYNTHESIS_STAGE_OUTPUT_SCHEMA,
    DEFAULT_DECOMPOSITION_AGENT_SYNTHESIS_TEMPLATE,
    DEFAULT_DECOMPOSITION_AGENT_SYSTEM_PROMPT_EN,
    DEFAULT_DECOMPOSITION_AGENT_SYSTEM_PROMPT_ZH)
from auraflux_core.canvases.schemas import ConceptualEdge, ConceptualNode
from auraflux_core.core.schemas.agents import AgentConfig
from auraflux_core.core.schemas.messages import (OutputFormat, PromptConfig,
                                                 StageSpec, SystemPromptSpec)


class DecompositionAgentConfig(AgentConfig):
    """Configuration for DecompositionAgent."""
    name: str = 'DecompositionAgent'
    output_format: Literal['TEXT', 'JSON'] = 'JSON'
    pipeline_name: Literal['direct', 'plan_and_execute'] = 'plan_and_execute'
    prompt_config: PromptConfig = Field(
        default_factory=lambda: PromptConfig(
            system=SystemPromptSpec(
                templates={
                    'zh': DEFAULT_DECOMPOSITION_AGENT_SYSTEM_PROMPT_ZH,
                    'en': DEFAULT_DECOMPOSITION_AGENT_SYSTEM_PROMPT_EN,
                    'default': DEFAULT_DECOMPOSITION_AGENT_SYSTEM_PROMPT_EN,
                }
            ),
            stages={
                'plan': StageSpec(
                    templates={'default': DEFAULT_DECOMPOSITION_AGENT_PLAN_TEMPLATE},
                    template_variables={
                        "source_node": "The source conceptual node to be decomposed.",
                    },
                    output_format=OutputFormat.JSON,
                    output_schema=DEFAULT_DECOMPOSITION_AGENT_PLAN_STAGE_OUTPUT_SCHEMA,
                ),
                'plan_refinement': StageSpec(
                    templates={'default': DEFAULT_DECOMPOSITION_AGENT_PLAN_REFINE_TEMPLATE},
                    template_variables={
                        "source_node": "The source conceptual node to be decomposed.",
                        "plan_output": "The output generated from the previous planning stage.",
                        "validation_warnings": "Validation error reasons and warnings.",
                    },
                    output_format=OutputFormat.JSON,
                    output_schema=DEFAULT_DECOMPOSITION_AGENT_PLAN_STAGE_OUTPUT_SCHEMA,
                ),
                'synthesis': StageSpec(
                    templates={'default': DEFAULT_DECOMPOSITION_AGENT_SYNTHESIS_TEMPLATE},
                    template_variables={
                        "source_node": "The source conceptual node to be decomposed.",
                        "plan_output": "The output generated from the planning stage.",
                    },
                    output_format=OutputFormat.JSON,
                    output_schema=DEFAULT_DECOMPOSITION_AGENT_SYNTHESIS_STAGE_OUTPUT_SCHEMA,
                ),
            },
        )
    )
    max_tokens: NonNegativeInt = 32768
    temperature: float = 0.0


class DecompositionResponse(BaseModel):
    """Response from DecompositionAgent."""
    nodes: List[ConceptualNode]
    edges: List[ConceptualEdge]


class DecompositionErrorResponse(BaseModel):
    """Standard error response structure."""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
