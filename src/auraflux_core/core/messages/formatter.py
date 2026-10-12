from typing import Any, Dict, Tuple

from auraflux_core.core.schemas.messages import Message, StageSpec


class PromptFormatter:
    """Formatter responsible for resolving system prompts and formatting stage messages."""

    def __init__(self, config: Any):
        self.config = config

    @property
    def prompt_config(self):
        """Dynamically fetch prompt_config from the bound AgentConfig."""
        return getattr(self.config, "prompt_config", None)

    def _validate_stage_spec(self, agent_name: str, stage_name: str) -> Tuple[Any, str]:
        """Verify if the stage specification and template exist for the current configuration."""
        if (
            not self.prompt_config
            or not getattr(self.prompt_config, "stages", None)
            or stage_name not in self.prompt_config.stages
        ):
            raise ValueError(
                f"[PromptFormatter] Stage '{stage_name}' is not configured in "
                f"prompt_config.stages for agent '{agent_name}'."
            )

        stage_spec = self.prompt_config.stages[stage_name]
        lang_code = getattr(self.config, "lang", "default")
        template_str = stage_spec.get_template(lang=lang_code)

        if not template_str:
            raise ValueError(
                f"[PromptFormatter] Template not found for stage '{stage_name}' "
                f"with language '{lang_code}'."
            )

        return stage_spec, template_str

    def _validate_runtime_kwargs(
        self, stage_name: str, stage_spec: StageSpec, template_str: str, provided_kwargs: Dict[str, Any]
    ) -> None:
        """
        Validate at runtime that all required placeholders in the template are provided in kwargs.
        """
        required_placeholders = stage_spec.extract_template_placeholders(template_str)
        provided_keys = set(provided_kwargs.keys())
        missing_vars = required_placeholders - provided_keys

        if missing_vars:
            raise ValueError(
                f"[PromptFormatter] Missing required variable(s) for stage '{stage_name}': "
                f"{list(missing_vars)}. Expected: {list(required_placeholders)}, "
                f"Provided: {list(provided_keys)}"
            )

    def format_system_message(self, **kwargs) -> str:
        """Resolve and format the system message from prompt_config."""
        if not self.prompt_config or not getattr(self.prompt_config, "system", None):
            return ""

        lang_code = getattr(self.config, "lang", "default")
        template_str = self.prompt_config.system.get_template(lang=lang_code)

        if not template_str:
            return ""

        return template_str.format(**kwargs) if kwargs else template_str

    def format_stage_message(
        self, agent_name: str, stage_name: str, **kwargs
    ) -> Message:
        """
        Format a stage-specific user prompt into a Message object after resolving
        the stage spec and validating runtime arguments.
        """
        stage_spec, template_str = self._validate_stage_spec(agent_name, stage_name)
        self._validate_runtime_kwargs(stage_name, stage_spec, template_str, kwargs)

        content = template_str.format(**kwargs) if kwargs else template_str
        return Message(role="user", content=content, name=agent_name)
