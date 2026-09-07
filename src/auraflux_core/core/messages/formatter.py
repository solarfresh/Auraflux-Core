from typing import Dict, Optional
from auraflux_core.core.schemas.agents import AgentConfig


class PromptFormatter:
    """
    Handles prompt construction, system message resolution,
    and language-specific message mapping.
    """

    def __init__(
        self,
        config: AgentConfig,
        system_message_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.config = config
        self.system_message_map = system_message_map or {}

    def format_system_message(self) -> str:
        """
        Resolve and format the system message based on configuration or language mapping.

        Returns:
            str: Resolved system message text (guaranteed non-None).
        """
        config_sys_msg = getattr(self.config, "system_message", None)
        if config_sys_msg is not None:
            return config_sys_msg

        resolved_msg = self.resolve_language_mapped_message(self.system_message_map)
        return resolved_msg if resolved_msg is not None else ""

    def resolve_language_mapped_message(
        self, message_map: Optional[Dict[str, str]]
    ) -> Optional[str]:
        if not message_map:
            return None

        lang_code = getattr(self.config, "lang", getattr(self.config, "language", "default"))
        return message_map.get(lang_code, message_map.get("default"))
