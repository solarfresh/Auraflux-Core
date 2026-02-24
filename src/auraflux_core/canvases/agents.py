from typing import Dict, List, Any, Optional
from auraflux_core.core.schemas.messages import Message
from auraflux_core.core.agents.base_agent import BaseAgent
from .schemas import ConceptualNode


class GraphSynthesistAgent(BaseAgent):
    """
    Agent specialized in graph expansion and structural recommendation.
    It evaluates existing research nodes to propose new functional nodes
    (RESOURCE, INSIGHT, QUERY, etc.) and determines their optimal placement.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tool_cache: Optional[Dict[str, Any]] = None

    def get_system_message_map(self) -> Dict[str, str]:
        # Fallback system message; primary logic is managed via backend AgentConfig
        return {
            'default': (
                "You are the Graph Synthesist. Your goal is to build a coherent "
                "research map by recommending logical node extensions. "
                "Analyze the FOCUS and CONCEPT nodes to identify gaps."
            )
        }

    def get_tool_map(self) -> Dict[str, Any]:
        """
        Lazily loads and caches the Graphing tools.
        """
        if self._tool_cache is None:
            # Local import to prevent circular dependencies and heavy startup
            from .tools import SpatialLocateTool
            self._tool_cache = {
                # 'locate_node': SpatialLocateTool()
            }

        return self._tool_cache

    def postprocess_llm_output(self, output_string: str) -> str:
        """
        Final cleanup to ensure the WebSocket receives a valid JSON
        batch of proposed nodes.
        """
        return output_string.replace('```json', '').replace('```', '').strip()
