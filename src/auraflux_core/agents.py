from typing import Any, Dict, NamedTuple, Type, Union

from auraflux_core.canvases.agents import GraphSynthesistAgent
from auraflux_core.canvases.schemas import (GraphSynthesistAgentConfig,
                                            SpatialLocateToolConfig)
from auraflux_core.core.agents.generic_agent import GenericAgent
from auraflux_core.core.schemas.agents import AgentConfig

Agent = Union[GraphSynthesistAgent, GenericAgent]


class AgentImplementation(NamedTuple):
    agent_class: Type[Any]
    config_class: Type[Any]
    tool_config_mapping: Dict[str, Type[Any]] = {}

# Central Registry
AGENT_REGISTRY: Dict[str, AgentImplementation] = {
    'GraphSynthesistAgent': AgentImplementation(
        agent_class=GraphSynthesistAgent,
        config_class=GraphSynthesistAgentConfig,
        tool_config_mapping={
            'spatial_locate': SpatialLocateToolConfig
        }
    ),
    # Default fallback or other agents
    'default': AgentImplementation(
        agent_class=GenericAgent,
        config_class=AgentConfig
    )
}