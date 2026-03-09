import json
from typing import Any, Dict

from auraflux_core.core.agents.base_agent import BaseAgent


class GraphSynthesistAgent(BaseAgent):
    """
    Agent specialized in graph expansion and structural recommendation.
    It evaluates existing research nodes to propose new functional nodes
    (RESOURCE, INSIGHT, QUERY, etc.) and determines their optimal placement.
    """
    def get_tool_message_map(self) -> Dict[str, str]:
        return self.get_system_message_map()

    def get_system_message_map(self) -> Dict[str, str]:
        return {
            'default': (
                'You are the Graph Synthesist, a specialized architect of knowledge structures. '
                'Your goal is to transform a flat list of research data into a coherent, navigable graph. '
                '\n\nCORE RESPONSIBILITIES:\n'
                '1. ANALYZE: Evaluate the semantic labels and functional types (CONCEPT, RESOURCE, INSIGHT, QUERY) '
                'of the provided nodes to identify a central theme.\n'
                '2. ANCHOR: Designate or suggest one FOCUS node as the gravitational center of the map.\n'
                '3. LINK: Recommend 0–3 nodes from the provided list to onboard onto the initial canvas. '
                'For each selected node, assign an "anchor_id" to establish a logical relationship.\n'
                '4. OPTIMIZE: Prioritize a clean topology. Group related evidence (RESOURCES) near their '
                'thematic anchors (CONCEPTS) and position exploratory gaps (QUERIES) at the periphery.'
            )
        }

    def get_tool_map(self) -> Dict[str, Any]:
        """
        Lazily loads and caches the Graphing tools.
        """
        if self._tool_cache is None:
            # Local import to prevent circular dependencies and heavy startup
            from .tools import SpatialLocateTool

            self._tool_cache = {}
            for name, config in self.config.tool_configs.items():
                if name == 'spatial_locate':
                    self._tool_cache['spatial_locate'] = SpatialLocateTool(config)

        return super().get_tool_map()

    def postprocess_tool_output(self, output_string: str) -> Any:
        return json.loads(output_string.replace('```json', '').replace('```', '').strip())
