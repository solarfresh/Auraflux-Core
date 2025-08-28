from abc import ABC, abstractmethod
from typing import Dict, List

from autogen.agentchat import ConversableAgent

from auraflux_core.schemas.agents import AgentConfig


class BaseAgent(ConversableAgent, ABC):
    """
    This class provides a shared logging setup and a consistent initialization pattern.
    The agent's specific behavior should be defined in subclasses by implementing their
    role within an AutoGen GroupChat or other conversational flows.
    """
    def __init__(self, config: AgentConfig, **kwargs):

        if config.system_messages is not None:
            system_message = config.system_messages
        else:
            system_message = self.get_system_message(config)

        super().__init__(
            name=config.name,
            system_message=system_message,
            llm_config=config.llm_config,
            **kwargs
        )

    @abstractmethod
    def get_system_message(self, config: AgentConfig) -> str | List[Dict[str, str]]:
        """
        An abstract method that must be overridden by all subclasses.
        It should return the system message based on the LLM configuration.
        """
        pass
