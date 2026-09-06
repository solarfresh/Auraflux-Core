from typing import Any, Dict, NamedTuple, Type, Union

from auraflux_core.canvases.agents import (GraphSynthesistAgent,
                                           KnowledgeArchitect, OntologyAuditor)
from auraflux_core.canvases.schemas import (GraphSynthesistAgentConfig,
                                            SpatialLocateToolConfig)
from auraflux_core.core.agents.generic_agent import GenericAgent
from auraflux_core.core.schemas.agents import AgentConfig
from auraflux_core.rag.agents.keywords_extractor import ExtractKeywordsAgent

Agent = Union[
    ExtractKeywordsAgent,
    # TODO: interfaces of BaseAgent were changed, so these agents are
    # TODO: temporarily disabled until they are updated to match
    # TODO: the new interface.
    # GraphSynthesistAgent,
    # KnowledgeArchitect,
    GenericAgent,
    # OntologyAuditor
]


class AgentImplementation(NamedTuple):
    agent_class: Type[Any]
    config_class: Type[Any]
    tool_config_mapping: Dict[str, Type[Any]] = {}

# Central Registry
AGENT_REGISTRY: Dict[str, AgentImplementation] = {
    'ExtractKeywordsAgent': AgentImplementation(
        agent_class=ExtractKeywordsAgent,
        config_class=AgentConfig
    ),
    # 'GraphSynthesistAgent': AgentImplementation(
    #     agent_class=GraphSynthesistAgent,
    #     config_class=GraphSynthesistAgentConfig,
    #     tool_config_mapping={
    #         'spatial_locate': SpatialLocateToolConfig
    #     }
    # ),
    # 'KnowledgeArchitect': AgentImplementation(
    #     agent_class=KnowledgeArchitect,
    #     config_class=AgentConfig,
    # ),
    # 'OntologyAuditor': AgentImplementation(
    #     agent_class=OntologyAuditor,
    #     config_class=AgentConfig,
    # ),
    # Default fallback or other agents
    'default': AgentImplementation(
        agent_class=GenericAgent,
        config_class=AgentConfig
    )
}