import json
import re
from copy import deepcopy
from typing import Any, Dict, List

from auraflux_core.core.agents.base_agent import BaseAgent
from auraflux_core.core.schemas.messages import Message
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
        return self._parse_json_output(output_string)


class OntologyAuditor(BaseAgent):
    def get_system_message_map(self) -> Dict[str, str]:
        return {
            "zh": (
                "你是一位「圖譜規格與邏輯審計員」(Graph Schema & Logic Auditor)。\n"
                "你的職責是確保【架構師】輸出的資料「完全符合既定規格」，並進行嚴格的邏輯配對審查。結構診斷數據僅作為全域參考，不代表局部邏輯的合法性。\n\n"
                "### 1. 嚴格規格基準 (優先權最高)：\n"
                "1. **節點類型 (Nodes)**：僅限 [EVENT], [INSIGHT], [OUTCOME], [BOUNDARY], [ENTITY]。\n"
                "2. **關係類型 (Edges)**：僅限 [REF], [VALIDATES], [CONSTRAINS], [TRIGGERS]。\n"
                "3. **嚴禁引入**：任何規格外的標籤、概念或 JSON 欄位 (如 rationale)。\n\n"
                "### 2. 深度邏輯審核點 (不可忽略)：\n"
                "- **關係配對邏輯**：審視每一條邊的語義。例如：\n"
                "  * [ENTITY] 之間通常使用 [REF]，不應直接使用 [TRIGGERS] (通常由 EVENT 觸發)。\n"
                "  * [VALIDATES] 應連接實證數據與 [INSIGHT] 或 [OUTCOME]。\n"
                "- **標籤歸一化**：在 Top Hubs 或節點清單中，若發現語義重疊 (如「微軟」與「Microsoft」)，必須要求合併以避免概念稀釋。\n"
                "- **分類準確性**：節點是否被歸類在最合適的類型？(例如：一個具體動作應是 EVENT 而非 ENTITY)。\n\n"
                "### 3. 結構診斷數據的解讀原則：\n"
                "- **Isolation Rate 僅代表密度**：即便顯示健康 (HEALTHY)，你仍需執行上述「深度邏輯審核」。\n"
                "- **高孤島率處理**：若數據顯示為碎片化，你必須在 Critique 中指出邏輯斷裂點，要求增加合理的因果連結。\n\n"
                "### 4. 審核反饋原則：\n"
                "- **拒絕模稜兩可**：若邏輯有瑕疵，即便 JSON 格式正確，也必須設定 'is_valid': false。\n"
                "- **嚴禁 LaTeX 與特殊轉義符號**：禁止使用反斜線 `\\`、字元 `$` 或任何 LaTeX 語法（如 `\\xrightarrow`）。這些符號會造成 JSON 解析崩潰。\n"
                "- **統一關係描述格式**：描述節點關係時，請使用「純文字箭頭」表示。範例：(節點A) -> [關係] -> (節點B)。\n"
                "- **實證導向**：所有修正建議必須基於原始文本，嚴禁幻想不存在的實體。\n\n"
                "### 輸出格式 (JSON)：\n"
                "- **必須確保輸出為標準 JSON，不含任何非法轉義序列。**\n"
                "- 'is_valid': 布林值。\n"
                "- 'critique': {\n"
                "    'violation_details': '違反的規格或邏輯配對細節。',\n"
                "    'structural_issues': '結構性問題 (如標籤合併、連通性優化方向)。',\n"
                "    'correction_suggestions': '修正建議 (請統一使用 -> 描述關係，嚴禁使用反斜線與 LaTeX)。'\n"
                "  }"
            ),
            "default": (
                "You are a Graph Schema & Logic Auditor.\n"
                "Your role is to ensure the Architect's output STRICTLY aligns with predefined specs and to perform rigorous logical pairing audits. Structural stats are background context only and do not imply local validity.\n\n"
                "### 1. Strict Specification Criteria (Highest Priority):\n"
                "1. **Node Types**: ONLY [EVENT], [INSIGHT], [OUTCOME], [BOUNDARY], [ENTITY].\n"
                "2. **Edge Types**: ONLY [REF], [VALIDATES], [CONSTRAINS], [TRIGGERS].\n"
                "3. **No Extra Fields**: Strictly JSON, no 'rationale' or external concepts allowed.\n\n"
                "### 2. Deep Logical Audit Focus (Mandatory):\n"
                "- **Edge Pairing Logic**: Verify semantic pairs. For example:\n"
                "  * [ENTITY] to [ENTITY] should typically use [REF], NOT [TRIGGERS].\n"
                "  * [VALIDATES] should connect evidence to [INSIGHT] or [OUTCOME].\n"
                "- **Label Normalization**: Identify semantic overlaps (e.g., 'Microsoft' vs '微軟') in the node list and demand mergers.\n"
                "- **Categorical Accuracy**: Ensure nodes are mapped to the most precise type (e.g., an action is an EVENT, not an ENTITY).\n\n"
                "### 3. Interpreting Structural Stats:\n"
                "- **Stats != Validity**: Even if connectivity is 'HEALTHY', you MUST still perform the Deep Logical Audit.\n"
                "- **Fragmentation**: If metrics show high isolation, pinpoint logical gaps and demand causal links.\n\n"
                "### 4. Feedback Principles:\n"
                "- **No Compromise**: If logic is flawed, 'is_valid' MUST be false even if JSON is well-formed.\n"
                "- **Strictly Avoid LaTeX**: Do NOT use backslashes (`\\`), dollar signs (`$`), or LaTeX commands (e.g., `\\xrightarrow`). These cause JSON parsing errors.\n"
                "- **Plain Text Relations**: Represent paths using simple text arrows, e.g., (Node A) -> [REL] -> (Node B).\n"
                "- **Grounded in Fact**: All suggestions must be supported by the source text; no hallucinations.\n\n"
                "### Output Format (JSON):\n"
                "- Ensure the output is a standard JSON string without invalid escape sequences.\n"
                "- 'is_valid': Boolean.\n"
                "- 'critique': {\n"
                "    'violation_details': 'Specific rule or logical pairing violations.',\n"
                "    'structural_issues': 'Structural issues like label mergers or connectivity gaps.',\n"
                "    'correction_suggestions': 'Clear instructions using plain text arrows only.'\n"
                "  }"
            )
        }

    async def generate(self, messages: List[Message], tool_args_map: Dict[str, Any] | None = None) -> Message:
        # 1. Deep copy to avoid mutating original history
        copied_messages = [deepcopy(msg) for msg in messages[-self.config.turn_limit:]]

        semantic_report = "Structural diagnostics not executed."
        if self.config.tool_execution_strategy == 'REFLECTIVE':
            # 2. Force tool call to get raw metrics
            tool_message = await self.generate_tool_message(copied_messages, tool_args_map)

            # 3. Translate metrics to semantic report
            semantic_report = self._translate_structural_metrics(tool_message.content)

            # 4. Integrate report into the LAST user message to maintain assistant-user sequence
            # This ensures the LLM sees the diagnostic as part of the context it needs to respond to.
            if copied_messages and copied_messages[-1].role == 'user':
                original_content = copied_messages[-1].content
                copied_messages[-1].content = (
                    f"--- STRUCTURAL DIAGNOSTIC REPORT ---\n"
                    f"{semantic_report}\n"
                    f"-------------------------------------\n\n"
                    f"Please perform the audit based on the context above:\n"
                    f"{original_content}"
                )
            else:
                # Fallback: if last message isn't from user, append a new user context
                copied_messages.append(Message(role='user', content=semantic_report, name="System_Diagnostic"))

        response = await self.generate_llm_message([copied_messages[-1]])
        response.metadata = {
            "diagnostic_conclusion": semantic_report
        }

        # 5. Final LLM generation
        return response

    def postprocess_llm_output(self, output_string: str) -> Any:
        return self._parse_json_output(output_string)

    def get_tool_call(self, messages: List[Message]) -> Dict[str, Any]:
        """
        Forces the agent to call the GraphIsolationRateTool.
        This bypasses LLM decision-making to ensure structural data is always available.
        """
        # The tool name should match the key in your get_tool_map()
        tool_name = "graph_isolation_rate_analyzer"
        if self._tool_cache is None or tool_name not in self._tool_cache:
            self.logger.error(f"Tool '{tool_name}' not found in tool map for agent '{self.name}'. Ensure it is defined in get_tool_map().")
            raise ValueError(f"Tool '{tool_name}' not available.")

        graph_json = json.loads(messages[-2].content)
        return {
            "tool": tool_name,
            "args": {
                "nodes": graph_json.get('nodes', []),
                "edges": graph_json.get('edges', [])
            }
        }

    def _translate_structural_metrics(self, tool_output: str) -> str:
            """
            Translates raw metrics into a structured context for the Auditor.
            This version avoids giving 'pass/fail' conclusions to prevent anchoring bias,
            ensuring the Agent still performs rigorous logic checks.
            """
            try:
                # Parse the tool's raw JSON output
                data = json.loads(tool_output)
                iso_rate = data.get("isolation_rate", 0)

                # Define thresholds and semantic status
                if iso_rate > 0.03:
                    status = "CRITICAL_FRAGMENTATION"
                    # For high isolation, we push for connectivity but maintain logic
                    context_advice = (
                        "High isolation detected. While validating schema, look for missing "
                        "logical bridges. Priority: Connect isolated components using valid types."
                    )
                elif iso_rate > 0.01:
                    status = "MILD_FRAGMENTATION"
                    context_advice = "Connectivity is stable. Focus on logical precision and entity alignment."
                else:
                    status = "OPTIMIZED_CONNECTIVITY"
                    # IMPORTANT: We no longer say 'No changes required'.
                    # We refocus the Agent on the micro-level logic audit.
                    context_advice = (
                        "Global connectivity is healthy. You MUST now perform a deep-dive "
                        "audit on local logic (e.g., Node-Edge type pairing and semantic accuracy)."
                    )

                # Assemble the report with clear boundaries
                return (
                    f"### [STRUCTURAL CONTEXT DATA]\n"
                    f"- Global Status: {status}\n"
                    f"- Isolation Rate: {iso_rate:.2%}\n"
                    f"- Contextual Guidance: {context_advice}\n"
                    f"-------------------------------------\n"
                    f"NOTE: The data above only reflects structural density. You are still "
                    f"REQUIRED to enforce strict schema rules and logical consistency."
                )

            except Exception as e:
                self.logger.error(f"Error translating structural metrics: {e}")
                return f"### [STRUCTURAL CONTEXT DATA]\nWarning: Metrics unavailable. Proceed with standard audit."


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
