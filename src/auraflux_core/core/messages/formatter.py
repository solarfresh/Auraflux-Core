from typing import Dict, Optional

from auraflux_core.core.schemas.agents import AgentConfig
from auraflux_core.core.schemas.messages import PromptConfig, Message


class PromptFormatter:
    """
    Handles prompt construction, global system message resolution,
    and stage-specific user message formatting using structured prompt configurations.
    """

    def __init__(
        self,
        config: AgentConfig,
        prompt_config: Optional[PromptConfig] = None,
        system_message_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.config = config
        self.prompt_config = prompt_config or PromptConfig()
        self.system_message_map = system_message_map or {}

    def format_system_message(self, **kwargs) -> str:
        """
        Resolves and formats the global system message based on AgentPromptConfig,
        AgentConfig override, or fallback language mapping.

        Returns:
            str: Resolved system message text (guaranteed non-None).
        """
        lang_code = self._get_language_code()

        # 1. Resolve from AgentPromptConfig.system (Pydantic PromptSpec)
        if self.prompt_config.system:
            template_str = self.prompt_config.system.get_template(lang=lang_code)
            if template_str:
                return template_str.format(**kwargs) if kwargs else template_str

        # 2. Resolve from direct config override
        config_sys_msg = getattr(self.config, "system_message", None)
        if config_sys_msg is not None:
            return config_sys_msg

        # 3. Fallback to legacy system_message_map
        resolved_msg = self.resolve_language_mapped_message(self.system_message_map)
        return resolved_msg if resolved_msg is not None else ""

    def format_stage_message(self, stage_name: str, agent_name: str, **kwargs) -> Message:
        """
        Formats and returns a user Message for a specific execution stage.

        Args:
            stage_name (str): Target stage key defined in AgentPromptConfig.stages.
            **kwargs: Dynamic variables to populate in the prompt template.

        Returns:
            Message: Formatted assistant/user Message object.
        """
        spec = self.prompt_config.get_stage_spec(stage_name)
        if not spec:
            return Message(role="user", content="", name=agent_name)

        lang_code = self._get_language_code()
        template_str = spec.get_template(lang=lang_code)

        content = template_str.format(**kwargs) if kwargs and template_str else template_str
        return Message(role="user", content=content, name=agent_name)

    def resolve_language_mapped_message(
        self, message_map: Optional[Dict[str, str]]
    ) -> Optional[str]:
        """
        Resolves a template from a simple language-to-string dictionary.
        """
        if not message_map:
            return None

        lang_code = self._get_language_code()
        return message_map.get(lang_code, message_map.get("default"))

    def _get_language_code(self) -> str:
        """
        Helper method to extract current language code from config.
        """
        return getattr(self.config, "lang", getattr(self.config, "language", "default"))
