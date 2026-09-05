from typing import Dict, Optional

from auraflux_core.core.schemas.agents import AgentConfig


class PromptFormatter:
    """
    Handles prompt construction, system message resolution,
    language-specific message mapping, and Chain-of-Thought (CoT) enrichment.
    """

    def __init__(
        self,
        config: AgentConfig,
        system_message_map: Optional[Dict[str, str]] = None,
        cot_message_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize the PromptFormatter.

        Args:
            config (AgentConfig): Agent configuration instance.
            system_message_map (Optional[Dict[str, str]]): Mapping of languages/models to system prompts.
            cot_message_map (Optional[Dict[str, str]]): Mapping of languages/models to CoT prompts.
        """
        self.config = config
        self.system_message_map = system_message_map or {}
        self.cot_message_map = cot_message_map or {}

    def format_system_message(self) -> str:
        """
        Resolve and format the system message based on configuration or language mapping.

        Returns:
            str: Resolved system message text.
        """
        if self.config.system_message is not None:
            return self.config.system_message

        return self.resolve_language_mapped_message(self.system_message_map) or ""

    def append_cot_message(self, content: str) -> str:
        """
        Append Chain-of-Thought (CoT) instructions to the message content if configured.

        Args:
            content (str): The original user or assistant message content.

        Returns:
            str: Updated content with CoT prompt appended.
        """
        cot_text = self.config.cot_message or self.resolve_language_mapped_message(
            self.cot_message_map
        )

        if cot_text:
            return f"{content}\n\n{cot_text}"
        return content

    def resolve_language_mapped_message(
        self, message_map: Optional[Dict[str, str]]
    ) -> Optional[str]:
        """
        Helper method to resolve a message from a dictionary map using the configured language code.

        Args:
            message_map (Optional[Dict[str, str]]): Dictionary containing language-key mapping.

        Returns:
            Optional[str]: Mapped string matching the language code, or default value.
        """
        if not message_map:
            return None

        # Fallback priority: configured language -> 'default' -> None
        return message_map.get(self.config.lang, message_map.get("default"))
