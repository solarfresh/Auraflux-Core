from typing import Literal

from pydantic import Field, NonNegativeInt

from auraflux_core.canvases.boundary.constants import (
    DEFAULT_BOUNDARY_AGENT_PLAN_REFINE_TEMPLATE,
    DEFAULT_BOUNDARY_AGENT_PLAN_STAGE_OUTPUT_SCHEMA,
    DEFAULT_BOUNDARY_AGENT_PLAN_TEMPLATE,
    DEFAULT_BOUNDARY_AGENT_SYNTHESIS_STAGE_OUTPUT_SCHEMA,
    DEFAULT_BOUNDARY_AGENT_SYNTHESIS_TEMPLATE,
    DEFAULT_BOUNDARY_AGENT_SYSTEM_PROMPT_EN,
    DEFAULT_BOUNDARY_AGENT_SYSTEM_PROMPT_ZH)
from auraflux_core.core.schemas.agents import AgentConfig
from auraflux_core.core.schemas.messages import (OutputFormat, PromptConfig,
                                                 StageSpec, SystemPromptSpec)


class BoundaryAgentConfig(AgentConfig):
    """Configuration for BoundaryAgent."""
    name: str = 'BoundaryAgent'
    output_format: Literal['TEXT', 'JSON'] = 'JSON'
    pipeline_name: Literal['direct', 'plan_and_execute'] = 'plan_and_execute'
    prompt_config: PromptConfig = Field(
        default_factory=lambda: PromptConfig(
            system=SystemPromptSpec(
                templates={
                    'zh': DEFAULT_BOUNDARY_AGENT_SYSTEM_PROMPT_ZH,
                    'en': DEFAULT_BOUNDARY_AGENT_SYSTEM_PROMPT_EN,
                    'default': DEFAULT_BOUNDARY_AGENT_SYSTEM_PROMPT_EN,
                }
            ),
            stages={
                'plan': StageSpec(
                    templates={'default': DEFAULT_BOUNDARY_AGENT_PLAN_TEMPLATE},
                    template_variables={
                        "source_node": "The source conceptual node to be analyzed.",
                    },
                    output_format=OutputFormat.JSON,
                    output_schema=DEFAULT_BOUNDARY_AGENT_PLAN_STAGE_OUTPUT_SCHEMA,
                ),
                'plan_refinement': StageSpec(
                    templates={'default': DEFAULT_BOUNDARY_AGENT_PLAN_REFINE_TEMPLATE},
                    template_variables={
                        "source_node": "The source conceptual node to be analyzed.",
                        "plan_output": "The output generated from the previous planning stage.",
                        "validation_warnings": "Validation error reasons and warnings.",
                    },
                    output_format=OutputFormat.JSON,
                    output_schema=DEFAULT_BOUNDARY_AGENT_PLAN_STAGE_OUTPUT_SCHEMA,
                ),
                'synthesis': StageSpec(
                    templates={'default': DEFAULT_BOUNDARY_AGENT_SYNTHESIS_TEMPLATE},
                    template_variables={
                        "source_node": "The source conceptual node to be analyzed.",
                        "plan_output": "The output generated from the planning stage.",
                    },
                    output_format=OutputFormat.JSON,
                    output_schema=DEFAULT_BOUNDARY_AGENT_SYNTHESIS_STAGE_OUTPUT_SCHEMA,
                ),
            },
        )
    )
    max_refine_retries: int = Field(default=3, ge=0)
    max_tokens: NonNegativeInt = 32768
    temperature: float = 0.0
