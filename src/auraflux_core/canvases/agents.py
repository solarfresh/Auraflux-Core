import json
from typing import Any, Dict

from auraflux_core.core.agents.base_agent import BaseAgent
from auraflux_core.core.tools.base_tool import BaseTool


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
                "你的任務是從原始文本中提取精確的實體與關係，並將其結構化為概念圖譜。\n\n"
                "### 節點分類 (Nodes):\n"
                "- [EVENT]: 具體發生的事件或動作。\n"
                "- [INSIGHT]: 從事實中得出的觀察、趨勢或結論。\n"
                "- [OUTCOME]: 事件導致的最終結果或產出。\n"
                "- [BOUNDARY]: 道德、法律或安全的紅線與約束條件。\n"
                "- [ENTITY]: 參與其中的組織、人名、技術或對象。\n\n"
                "### 關係分類 (Edges):\n"
                "- [REF]: 提及、引用或基本的關聯。\n"
                "- [VALIDATES]: 事實證明了某個觀察或結論。\n"
                "- [CONSTRAINS]: 某種規則或邊界限制了行為。\n"
                "- [TRIGGERS]: 一個事件或因素觸發了另一個結果。\n\n"
                "### 規則：\n"
                "1. 嚴格遵守實證原則，不可推論文本中未提及的資訊。\n"
                "2. 輸出格式必須為 JSON，包含 'nodes' 與 'edges' 列表。\n"
                "3. 每個 node 必須包含: 'id' (英文唯一識別碼), 'label' (中文標籤), 'type' (上述 Node 分類之一)。\n"
                "4. 每個 edge 必須包含: 'source', 'target', 'relation' (上述 Edge 分類之一)。"
            ),
            "default": (
                "You are a Senior Knowledge Architect. Your task is to extract precise "
                "entities and relationships and structure them into a conceptual graph.\n\n"
                "### Node Taxonomy:\n"
                "- [EVENT], [INSIGHT], [OUTCOME], [BOUNDARY], [ENTITY]\n\n"
                "### Edge Taxonomy:\n"
                "- [REF], [VALIDATES], [CONSTRAINS], [TRIGGERS]\n\n"
                "### Rules:\n"
                "1. Follow empirical principles—do not infer information not present.\n"
                "2. Output must be valid JSON with 'nodes' and 'edges' keys.\n"
                "3. Each node MUST have: 'id' (unique kebab-case), 'label' (display name), 'type' (from taxonomy).\n"
                "4. Each edge MUST have: 'source', 'target', 'relation' (from taxonomy)."
            )
        }

    def postprocess_llm_output(self, output_string: str) -> Any:
        return json.dumps(json.loads(output_string.replace('```json', '').replace('```', '').strip()), ensure_ascii=False)


class OntologyAuditor(BaseAgent):
    def get_system_message_map(self) -> Dict[str, str]:
        return {
            "zh": (
                "你是一位「圖譜規格審計員」(Graph Schema Validator)。\n"
                "你的唯一職責是確保【架構師】輸出的資料「完全符合既定規格」，嚴禁引入規格外的概念。\n\n"
                "### 審核基準 (嚴格遵守)：\n"
                "1. **節點類型 (Nodes)**：僅限 [EVENT], [INSIGHT], [OUTCOME], [BOUNDARY], [ENTITY]。禁止建議使用其他類型。\n"
                "2. **關係類型 (Edges)**：僅限 [REF], [VALIDATES], [CONSTRAINS], [TRIGGERS]。禁止建議使用其他類型。\n"
                "3. **資料結構**：必須符合 JSON 格式。禁止要求規格外(如 rationale)的欄位。\n\n"
                "### 審核重點：\n"
                "- **邏輯合理性**：例如兩個 ENTITY 之間不應使用 TRIGGERS (通常是 EVENT 觸發另一件事)。\n"
                "- **文本一致性**：是否有節點完全脫離原始文本的實證。\n"
                "- **分類準確性**：在「現有五種類別」中，該節點是否選擇了最合適的一個？若否，請指明應更換為哪一個現有類別。\n\n"
                "### 輸出格式 (JSON)：\n"
                "- 'is_valid': 布林值。\n"
                "- 'critique': 若不通過，請具體指出：哪個節點/關係錯誤、違反哪條規則、以及如何「在現有規格內」修正。"
            ),
            "default": (
                "You are a Graph Schema Validator.\n"
                "Your sole responsibility is to ensure the output from the Architect aligns STRICTLY with the predefined specification. Do NOT introduce new concepts.\n\n"
                "### Audit Criteria (Strict):\n"
                "1. **Node Types**: ONLY [EVENT], [INSIGHT], [OUTCOME], [BOUNDARY], [ENTITY]. Do NOT suggest others.\n"
                "2. **Edge Types**: ONLY [REF], [VALIDATES], [CONSTRAINS], [TRIGGERS]. Do NOT suggest others.\n"
                "3. **Structure**: Must be JSON. Do NOT demand extra fields (e.g., rationale).\n\n"
                "### Audit Focus:\n"
                "- **Logical Consistency**: e.g., ensure TRIGGERS connects appropriate node types (usually events).\n"
                "- **Empirical Alignment**: Ensure no hallucinated nodes.\n"
                "- **Categorical Best-fit**: If a node is misclassified, map it to the most suitable existing type from the allowed list.\n\n"
                "### Output Format (JSON):\n"
                "- 'is_valid': Boolean.\n"
                "- 'critique': If failed, specify: which node/edge is wrong, which rule was violated, and how to fix it WITHIN the existing spec."
            )
        }

    def postprocess_llm_output(self, output_string: str) -> Any:
        return json.dumps(json.loads(output_string.replace('```json', '').replace('```', '').strip()), ensure_ascii=False)


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

    def get_tool_map(self) -> Dict[str, BaseTool]:
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
