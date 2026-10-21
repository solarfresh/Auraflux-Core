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
    ):
        super().__init__(config=config, client_manager=client_manager)
