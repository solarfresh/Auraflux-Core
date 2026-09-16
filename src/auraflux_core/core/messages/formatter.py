from typing import Any

from auraflux_core.core.schemas.messages import Message


class PromptFormatter:
    """Formatter responsible for resolving system prompts and formatting stage messages."""

    def __init__(self, config: Any):
        self.config = config

    @property
    def prompt_config(self):
        """Dynamically fetch prompt_config from the bound AgentConfig."""
        return getattr(self.config, "prompt_config", None)

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
        """Format a stage-specific user prompt into a Message object."""
        lang_code = getattr(self.config, "lang", "default")
        content = ""

        if (
            self.prompt_config
            and getattr(self.prompt_config, "stages", None)
            and stage_name in self.prompt_config.stages
        ):
            stage_spec = self.prompt_config.stages[stage_name]
            template_str = stage_spec.get_template(lang=lang_code)
            if template_str:
                content = (
                    template_str.format(**kwargs) if kwargs else template_str
                )

        return Message(role="user", content=content, name=agent_name)
