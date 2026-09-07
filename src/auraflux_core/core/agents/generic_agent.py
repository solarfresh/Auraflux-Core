from typing import Dict, Optional

from auraflux_core.core.agents.base_agent import BaseAgent
from auraflux_core.core.clients.client_manager import ClientManager
from auraflux_core.core.schemas.agents import AgentConfig


class GenericAgent(BaseAgent):
    """
    A minimal, but fully capable agent implementation that serves as a
    Configuration-as-a-Service agent, allowing all behavior (persona, schema,
    parameters) to be overridden dynamically at runtime.
    """

    def __init__(
        self,
        config: AgentConfig,
        client_manager: ClientManager,
        system_message_map: Optional[Dict[str, str]] = None,
        cot_message_map: Optional[Dict[str, str]] = None,
    ):
        self._system_message_map = system_message_map or {}
        self._cot_message_map = cot_message_map or {}

        super().__init__(config=config, client_manager=client_manager)

    def get_system_message_map(self) -> Dict[str, str]:
        """Returns the instance-specific system message mapping."""
        return self._system_message_map

    def update_system_message_map(self, new_map: Dict[str, str]) -> None:
        """
        Update the system message map and sync with the underlying PromptFormatter.

        Args:
            new_map (Dict[str, str]): A dictionary mapping model families or languages to system messages.
        """
        self._system_message_map = new_map
        if hasattr(self, "prompt_formatter") and self.prompt_formatter:
            self.prompt_formatter.system_message_map = new_map
