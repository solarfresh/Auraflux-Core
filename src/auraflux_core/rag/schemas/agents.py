from typing import List, Literal

from pydantic import Field, NonNegativeInt

from auraflux_core.core.schemas.agents import AgentConfig
from auraflux_core.core.schemas.messages import (OutputFormat, PromptConfig,
                                                 StageSpec, SystemPromptSpec)
from auraflux_core.core.tools.base_tool import BaseTool
from auraflux_core.rag.constants.agents import (
    DEFAULT_EXTRACT_KEYWORDS_AGENT_PLAN_REFINE_STAGE_OUTPUT_SCHEMA,
    DEFAULT_EXTRACT_KEYWORDS_AGENT_PLAN_REFINEMENT_TEMPLATE,
    DEFAULT_EXTRACT_KEYWORDS_AGENT_PLAN_STAGE_OUTPUT_SCHEMA,
    DEFAULT_EXTRACT_KEYWORDS_AGENT_PLAN_TEMPLATE,
    DEFAULT_EXTRACT_KEYWORDS_AGENT_SYSTEM_PROMPT)
from auraflux_core.rag.tools.triple_processor_tool import (
    TripleProcessorTool, TripleRuleCheckerTool)


class ExtractKeywordsAgentConfig(AgentConfig):
    """
    Dedicated AgentConfig for ExtractKeywordsAgent with specialized domain defaults.
    """
    output_format: Literal['TEXT', 'JSON'] = 'JSON'
    pipeline_name: Literal['direct', 'plan_and_execute'] = 'plan_and_execute'
    prompt_config: PromptConfig = Field(
        default_factory=lambda: PromptConfig(
            system=SystemPromptSpec(
                templates={
                    'en': DEFAULT_EXTRACT_KEYWORDS_AGENT_SYSTEM_PROMPT,
                    'default': DEFAULT_EXTRACT_KEYWORDS_AGENT_SYSTEM_PROMPT
                }),
            stages={
                'plan': StageSpec(
                    templates={'default': DEFAULT_EXTRACT_KEYWORDS_AGENT_PLAN_TEMPLATE},
                    template_variables={
                        'excerpt_text': 'The raw document text chunk to be analyzed for triples and keywords extraction.'
                    },
                    output_format=OutputFormat.JSON,
                    output_schema=DEFAULT_EXTRACT_KEYWORDS_AGENT_PLAN_STAGE_OUTPUT_SCHEMA,
                ),
                'plan_refinement': StageSpec(
                    templates={'default': DEFAULT_EXTRACT_KEYWORDS_AGENT_PLAN_REFINEMENT_TEMPLATE},
                    template_variables={
                        'excerpt_text': 'The localized source text chunk used as ground truth for repairing errors.',
                        'flagged_triples': 'A JSON list or formatted string of triples that failed quality rules, including violation reasons.'
                    },
                    output_format=OutputFormat.JSON,
                    output_schema=DEFAULT_EXTRACT_KEYWORDS_AGENT_PLAN_REFINE_STAGE_OUTPUT_SCHEMA,
                )
            }
        )
    )

    tools: List["BaseTool"] = [
        TripleProcessorTool(),
        TripleRuleCheckerTool()
    ]

    max_refine_retries: int = 3
    max_tokens: NonNegativeInt= 32768
    temperature: float = 0.0
