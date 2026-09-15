from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PromptSpec(BaseModel):
    """
    Generic Prompt Specification: Handles language-specific template resolution.
    """
    templates: Dict[str, str] = Field(
        default_factory=dict,
        description="Language to prompt template mapping, e.g., {'default': '...', 'zh-TW': '...'}"
    )

    def get_template(self, lang: str = "default") -> str:
        """Retrieves the corresponding template based on language code."""
        return self.templates.get(lang, self.templates.get("default", ""))


class PromptConfig(BaseModel):
    """
    Agent Prompt Configuration Package: Contains global system prompt spec and stage-specific user prompt specs.
    """
    system: Optional[PromptSpec] = Field(
        default=None,
        description="Global system prompt spec for the agent"
    )
    stages: Dict[str, PromptSpec] = Field(
        default_factory=dict,
        description="Stage name to user prompt spec mapping"
    )

    def get_stage_spec(self, stage_name: str) -> Optional[PromptSpec]:
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
