import string
from enum import Enum
from typing import Any, Dict, Optional, Set

from pydantic import BaseModel, Field, model_validator


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
    template_variables: Dict[str, str] = Field(
        default_factory=dict,
        description="Descriptions for variables in this stage's templates, e.g., {'topic': 'The target user interest topic to analyze.'}"
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

    @model_validator(mode="after")
    def validate_spec_consistency(self) -> "StageSpec":
        """
        Validates internal specification consistency upon StageSpec initialization:
        1. Ensures all placeholders in template strings are declared in template_variables.
        2. Ensures output_schema is provided if output_format is JSON.
        """
        declared_vars = set(self.template_variables.keys())

        # 1. Validate template placeholders against template_variables
        for lang, template_str in self.templates.items():
            required_placeholders = self.extract_template_placeholders(template_str)
            missing_declarations = required_placeholders - declared_vars

            if missing_declarations:
                raise ValueError(
                    f"[StageSpec Config Error] Template for language '{lang}' requires variable(s) "
                    f"{list(missing_declarations)}, but they are missing from StageSpec.template_variables. "
                    f"Declared variables: {list(declared_vars)}"
                )

        # 2. Validate JSON output schema contract
        if self.output_format == OutputFormat.JSON and not self.output_schema:
            raise ValueError(
                "[StageSpec Config Error] output_format is set to OutputFormat.JSON, "
                "but output_schema is empty."
            )

        return self

    def extract_template_placeholders(self, template_str: str) -> Set[str]:
        """Helper function to extract all root placeholder variable names from a format string."""
        formatter = string.Formatter()
        placeholders = set()
        for _, field_name, _, _ in formatter.parse(template_str):
            if field_name is not None:
                root_var = field_name.split(".")[0].split("[")[0]
                placeholders.add(root_var)
        return placeholders


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
