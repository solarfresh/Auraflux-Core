from typing import Dict

from auraflux_core.core.agents.base_agent import BaseAgent


class GenericAgent(BaseAgent):
    """
    A minimal, but fully capable agent implementation that serves as a
    Configuration-as-a-Service agent, allowing all behavior (persona, schema,
    parameters) to be overridden dynamically at runtime.
    """
    system_message_map: Dict[str, str] = {}

    def get_system_message_map(self) -> Dict[str, str]:
        """
        Abstract method to be implemented by subclasses to provide a mapping of model families
        to their respective system messages.
        """
        return self.system_message_map

    def update_system_message_map(self, new_map: Dict[str, str]) -> None:
        """
        Update the system message map with a new mapping.

        Args:
            new_map (Dict[str, str]): A dictionary mapping model families to system messages.
        """
        self.system_message_map = new_map