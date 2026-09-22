from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class OutputFormat(str, Enum):
    TEXT = "TEXT"
    JSON = "JSON"


class StageSpec(BaseModel):
    """
    Defines the specification for a specific execution stage,
    including language templates, variable mappings, and expected output structure.
    """
    templates: Dict[str, str] = Field(
        default_factory=dict,
        description="Language to prompt template mapping, e.g., {'default': '...', 'zh-TW': '...'}"
    )
    template_variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Data source/type mappings for variables in this stage's templates, e.g., {'topic': 'CURRENT_TURN'}"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.TEXT,
        description="Expected format for this stage's LLM output"
    )
    output_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema defining required output structure if format is JSON"
    )

    def get_template(self, lang: str = "default") -> str:
        """Retrieves the corresponding template based on language code."""
        return self.templates.get(lang, self.templates.get("default", ""))


class SystemPromptSpec(BaseModel):
    """
    Global system prompt spec. System prompts usually only need language templates.
    """
    templates: Dict[str, str] = Field(
        default_factory=dict,
        description="Language to system prompt template mapping"
    )

    def get_template(self, lang: str = "default") -> str:
        return self.templates.get(lang, self.templates.get("default", ""))


class PromptConfig(BaseModel):
    """
    Agent Prompt Configuration Package.
    """
    system: Optional[SystemPromptSpec] = Field(
        default=None,
        description="Global system prompt spec for the agent"
    )
    stages: Optional[Dict[str, StageSpec]] = Field(
        default_factory=dict,
        description="Stage name to StageSpec mapping"
    )

    def get_stage_spec(self, stage_name: str) -> Optional[StageSpec]:
        if not self.stages:
            return None

        return self.stages.get(stage_name)


class Message(BaseModel):
    role: str
    content: str
    name: str
    token_usage: Optional[int] = 0
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Bucket for scenario-specific data (e.g., timestamps, confidence scores)."
    )
