from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Type, Union

from auraflux_core.alignment.mental_model.agents import MentalModelAgent
from auraflux_core.alignment.objective_claim.agents import ObjectiveClaimAgent
from auraflux_core.alignment.threshold_claim.agents import ThresholdClaimAgent
from auraflux_core.canvases.agents import (GraphSynthesistAgent,
                                           KnowledgeArchitect, OntologyAuditor)
from auraflux_core.canvases.schemas import (GraphSynthesistAgentConfig,
                                            SpatialLocateToolConfig)
from auraflux_core.core.agents.base import BaseAgent
from auraflux_core.core.protocols.agents import AgentProtocol
from auraflux_core.core.protocols.clients import ClientManagerProtocol
from auraflux_core.core.protocols.tools import ToolProtocol
from auraflux_core.core.schemas.agents import AgentConfig
from auraflux_core.rag.agents.keywords_extractor import ExtractKeywordsAgent

Agent = Union[
    MentalModelAgent,
    ObjectiveClaimAgent,
    ThresholdClaimAgent,
    ExtractKeywordsAgent,
    # TODO: interfaces of BaseAgent were changed, so these agents are
    # TODO: temporarily disabled until they are updated to match
    # TODO: the new interface.
    # GraphSynthesistAgent,
    # KnowledgeArchitect,
    # OntologyAuditor
]


@dataclass
class AgentImplementation:
    agent_class: Callable[[AgentConfig, ClientManagerProtocol], AgentProtocol]
    config_class: Type[AgentConfig]
    tool_config_mapping: Dict[str, ToolProtocol] = field(default_factory=dict)

    def build_agent(self, config: AgentConfig, client_manager: ClientManagerProtocol) -> AgentProtocol:
        if config is None:
            raise ValueError("config cannot be None; AgentConfig requires mandatory fields (name, provider, model)")

        if not isinstance(config, self.config_class):
            raise ValueError(f"config must be an instance of {self.config_class.__name__}")

        return self.agent_class(config, client_manager)

# Central Registry
AGENT_REGISTRY: Dict[str, AgentImplementation] = {
    # Alignment Agents
    'MentalModelAgent': AgentImplementation(
        agent_class=MentalModelAgent,
        config_class=AgentConfig
    ),
    'ObjectiveClaimAgent': AgentImplementation(
        agent_class=ObjectiveClaimAgent,
        config_class=AgentConfig
    ),
    'ThresholdClaimAgent': AgentImplementation(
        agent_class=ThresholdClaimAgent,
        config_class=AgentConfig
    ),
    # RAG Agents
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
        agent_class=BaseAgent,
        config_class=AgentConfig
    )
}