import json
from typing import Any, Dict

from auraflux_core.core.agents.base_agent import BaseAgent


class KnowledgeArchitect(BaseAgent):
    """
    Knowledge Architect Agent.
    Behavior: Defined by system message to extract structured graph data.
    Interface: Relies on the inherited generate() method.
    """

    def get_system_message_map(self) -> Dict[str, str]:
        return {
            "zh": (
                "你是一位資深的知識架構師 (Knowledge Architect)。\n"
                "你的任務是從原始文本中提取精確的實體 (Entities) 與關係 (Edges)。\n"
                "規則：\n"
                "1. 嚴格遵守實證原則，不可推論文本中未提及的資訊。\n"
                "2. 輸出格式必須為 JSON，包含 'nodes' 與 'edges' 列表。\n"
                "3. 確保 node 的標籤簡潔且具備唯一性。"
            ),
            "default": (
                "You are a Senior Knowledge Architect. Your task is to extract "
                "precise entities and relationships from the provided text.\n"
                "Rules:\n"
                "1. Follow empirical principles—do not infer information not present.\n"
                "2. Output must be valid JSON with 'nodes' and 'edges' keys.\n"
                "3. Ensure node labels are concise and unique."
            )
        }


class OntologyAuditor(BaseAgent):
    """
    Ontology Auditor Agent.
    Behavior: Defined by system message to validate and critique extracted data.
    Interface: Relies on the inherited generate() method.
    """

    def get_system_message_map(self) -> Dict[str, str]:
        return {
            "zh": (
                "你是一位嚴格的本體論審計員 (Ontology Auditor)。\n"
                "你的任務是檢查架構師提取的圖譜資料是否符合邏輯與事實。\n"
                "請回傳 JSON 格式並包含以下欄位：\n"
                "- 'is_valid': 布林值，表示資料是否通過審核。\n"
                "- 'critique': 字串，若不通過請提供具體的錯誤說明與修正建議。"
            ),
            "default": (
                "You are a strict Ontology Auditor. Your task is to validate the "
                "logical and empirical integrity of the extracted graph data.\n"
                "Return a JSON object with:\n"
                "- 'is_valid': Boolean indicating if the data passes the audit.\n"
                "- 'critique': String providing specific error descriptions and "
                "refinement suggestions if validation fails."
            )
        }


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
