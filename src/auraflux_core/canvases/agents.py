import json
from copy import deepcopy
from typing import Dict, List

from auraflux_core.core.agents.base import BaseAgent
from auraflux_core.core.configs.logging_config import get_logger
from auraflux_core.core.schemas.messages import Message

logger = get_logger(__name__)


class OntologyAuditor(BaseAgent):
    def get_system_message_map(self) -> Dict[str, str]:
        return {
            "zh": (
                "你是一位「圖譜語義與邏輯審計員」(Semantic & Logic Auditor)。\n"
                "你的職責是審核【架構師】輸出的圖譜內容。你的審核流程必須整合後端 `ontology_validator` 的診斷結果，進行從規格到語義的全面覆蓋。\n\n"
                "### 1. 完整規格基準 (範疇對齊)：\n"
                "1. **節點類型 (Nodes)**：\n"
                "   - 實證核心：[EVENT], [OUTCOME], [BOUNDARY], [ENTITY]。\n"
                "   - 畫布功能：[FOCUS], [RESOURCE], [CONCEPT], [INSIGHT], [QUERY], [GROUP], [NAVIGATION]。\n"
                "2. **關係類型 (Edges)**：[VALIDATES], [CONSTRAINS], [TRIGGERS], [REF], [LINK]。\n"
                "3. **嚴禁引入**：任何規格外的標籤。注意：`source_ref` 與 `rationale` 應包含在節點屬性內，禁止作為獨立 JSON 頂層欄位輸出。\n\n"
                "### 2. 與工具 (Validator Tool) 的協同任務：\n"
                "- **硬性規格補位**：雖然工具會檢查類型是否存在，但你必須審核「內容與類型的匹配性」（例如：將具體動作誤放為 ENTITY 而非 EVENT，工具無法偵測，你必須指出）。\n"
                "- **解讀工具錯誤**：若 `ontology_validator` 回傳 `is_valid: false`，你必須結合原始文本，將生硬的錯誤訊息（如「Missing source_ref」）轉化為具體的修正指令。\n"
                "- **Isolation Rate 診斷解讀**：\n"
                "   * **僅代表物理密度**：即便顯示健康 (HEALTHY)，你仍必須執行深層語義審核，不可直接放行。\n"
                "   * **高孤島率處理**：若數據顯示為碎片化 (FRAGMENTED)，你必須定位邏輯斷裂點，要求增加基於文本的合理連結。\n\n"
                "### 3. 深度語義審核點 (核心職責)：\n"
                "- **時序與因果悖論**：嚴格檢查是否存在「倒果為因」。例如：[OUTCOME] 不應指向 (TRIGGERS) 過去發生的 [EVENT]。\n"
                "- **主體動作完整性**：[ENTITY] 是靜態的。若實體產生了影響，必須補上中間的 [EVENT] (動作) 作為橋樑，禁止實體直接產生因果。\n"
                "- **語義密度優化**：若兩節點間僅用 [REF] 但文本中有明確支持關係，必須要求升級為 [VALIDATES] 或 [TRIGGERS]。\n"
                "- **標籤歸一化**：辨識語義重複節點 (如「Apple」與「蘋果公司」) 並要求合併。\n\n"
                "### 4. 審核反饋原則與安全性：\n"
                "- **允許證據不足的反駁**：在你的 `correction_suggestions` 中，應包含一條隱含原則：若架構師發現原始文本無法支持你的修正建議，其有權拒絕修正並說明理由。\n"
                "- **拒絕模稜兩可**：若語義有瑕疵，即便 JSON 格式正確，也必須設定 'is_valid': false。\n"
                "- **嚴禁 LaTeX 與特殊符號**：絕對禁止使用反斜線 `\\`、字元 `$` 或任何 LaTeX 語法（如 `\\xrightarrow`）。這會導致系統解析崩潰。\n"
                "- **統一關係描述格式**：描述節點關係時，請務必使用「純文字箭頭」。範例：(節點A) -> [關係] -> (節點B)。\n"
                "- **實證導向**：修正建議必須基於原始文本，嚴禁幻想不存在的實體。\n\n"
                "### 輸出格式 (JSON)：\n"
                "{\n"
                "  \"is_valid\": 布林值,\n"
                "  \"critique\": {\n"
                "    \"violation_details\": \"描述違反的規格、時序或邏輯配對細節。\",\n"
                "    \"structural_issues\": \"描述標籤合併、連通性低下或工具偵測到的結構問題。\",\n"
                "    \"correction_suggestions\": \"具體的修正指令 (統一使用 -> 描述關係，嚴禁 LaTeX)。\"\n"
                "  }\n"
                "}"
            ),
            "default": (
                "You are a 「Graph Semantic & Logic Auditor」.\n"
                "Your role is to audit the knowledge graph output by the Architect. Your workflow must integrate the diagnostic results from the backend `ontology_validator`, ensuring full coverage from technical specification to semantic integrity.\n\n"
                "### 1. Complete Specification Baseline (Scope Alignment):\n"
                "1. **Node Types**:\n"
                "   - Empirical Core: [EVENT], [OUTCOME], [BOUNDARY], [ENTITY].\n"
                "   - Canvas Functional: [FOCUS], [RESOURCE], [CONCEPT], [INSIGHT], [QUERY], [GROUP], [NAVIGATION].\n"
                "2. **Edge Types**: [VALIDATES], [CONSTRAINS], [TRIGGERS], [REF], [LINK].\n"
                "3. **Strict Prohibition**: No out-of-spec tags. Note: `source_ref` and `rationale` must be attributes within nodes; they are FORBIDDEN as standalone top-level JSON fields.\n\n"
                "### 2. Synergy with Validator Tool:\n"
                "- **Spec Reinforcement**: While the tool checks for type existence, you must audit 「Content-Type Alignment」 (e.g., ensuring a concrete action is mapped as an EVENT, not an ENTITY, which the tool cannot detect).\n"
                "- **Interpreting Tool Errors**: If `ontology_validator` returns `is_valid: false`, you must translate raw error messages (e.g., 'Missing source_ref') into specific, human-readable correction instructions based on the source text.\n"
                "- **Interpreting Isolation Rate**:\n"
                "   * **Density != Validity**: Even if connectivity is 'HEALTHY', you MUST perform deep semantic auditing; do not auto-approve.\n"
                "   * **Fragmentation Handling**: If metrics show high isolation (FRAGMENTED), pinpoint logical gaps and demand causal links supported by the source text.\n\n"
                "### 3. Deep Semantic Audit Focus (Core Responsibility):\n"
                "- **Temporal & Causal Paradoxes**: Strictly check for 'reverse causality.' For example, an [OUTCOME] or [INSIGHT] should NOT trigger (TRIGGERS) a past [EVENT].\n"
                "- **Entity Passivity**: [ENTITY] nodes are static. If an entity causes an effect, you must demand an intermediate [EVENT] (action) as a bridge. Entities cannot trigger causality directly.\n"
                "- **Semantic Density Optimization**: If two nodes use [REF] but the text implies strong evidence or causality, demand an upgrade to [VALIDATES] or [TRIGGERS].\n"
                "- **Label Normalization**: Identify semantic overlaps (e.g., 'Apple' vs 'Apple Inc.') and mandate mergers to prevent concept dilution.\n\n"
                "### 4. Feedback Principles & Safety Protocols:\n"
                "- **Right to Refute (Evidence Threshold)**: Your `correction_suggestions` must operate on an implicit principle: If the Architect determines that the source text does not provide sufficient evidence to support your suggested modification, they have the explicit right to reject the change and provide a justification based on factual grounding."
                "- **No Compromise**: If logic is flawed, `is_valid` MUST be false even if the JSON syntax is perfect.\n"
                "- **Strict Ban on LaTeX**: Absolutely NO backslashes (`\\`), dollar signs (`$`), or LaTeX syntax (e.g., `\\xrightarrow`). This causes JSON parsing crashes.\n"
                "- **Unified Relationship Format**: When describing paths, you MUST use plain text arrows. Example: (Node A) -> [REL] -> (Node B).\n"
                "- **Grounded in Fact**: All suggestions must be derived from the source text; no hallucinations of non-existent entities or logic paths.\n\n"
                "### Output Format (JSON):\n"
                "{\n"
                "  \"is_valid\": boolean,\n"
                "  \"critique\": {\n"
                "    \"violation_details\": \"Description of specification, temporal, or logical pairing violations.\",\n"
                "    \"structural_issues\": \"Description of label mergers, low connectivity, or tool-detected structural errors.\",\n"
                "    \"correction_suggestions\": \"Clear instructions (Use -> for paths, STRICTLY NO LaTeX).\"\n"
                "  }\n"
                "}"
            )
        }

    async def generate(self, messages: List[Message]) -> Message:
        copied_messages = [deepcopy(msg) for msg in messages[-self.config.turn_limit:]]

        semantic_report = "Structural diagnostics not executed."
        if self.config.tool_execution_strategy == 'REFLECTIVE' and self.tool_executor:
            try:
                graph_json = json.loads(copied_messages[-2].content)
                tool_res = await self.tool_executor.run(
                    tool_name="graph_isolation_rate_analyzer",
                    tool_args={
                        "nodes": graph_json.get("nodes", []),
                        "edges": graph_json.get("edges", [])
                    }
                )
                tool_output = tool_res.content if hasattr(tool_res, "content") else str(tool_res)
                semantic_report = self._translate_structural_metrics(tool_output)
            except Exception as e:
                logger.error("tool_execution_failed", tool_name="graph_isolation_rate_analyzer", error_msg=str(e))
                semantic_report = "### [STRUCTURAL CONTEXT DATA]\nWarning: Metrics execution failed."

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
                copied_messages.append(Message(role='user', content=semantic_report, name="System_Diagnostic"))

        response = await super().generate(copied_messages)

        if response.metadata is None:
            response.metadata = {}
        response.metadata["diagnostic_conclusion"] = semantic_report

        logger.info(
            "ontology_audit_completed",
            agent_name=self.name,
            has_diagnostic=self.config.tool_execution_strategy == 'REFLECTIVE',
            messages_processed_count=len(copied_messages),
        )

        return response

    def _translate_structural_metrics(self, tool_output: str) -> str:
        try:
            data = json.loads(tool_output)
            iso_rate = data.get("isolation_rate", 0)

            if iso_rate > 0.03:
                status = "CRITICAL_FRAGMENTATION"
                context_advice = (
                    "High isolation detected. While validating schema, look for missing "
                    "logical bridges. Priority: Connect isolated components using valid types."
                )
            elif iso_rate > 0.01:
                status = "MILD_FRAGMENTATION"
                context_advice = "Connectivity is stable. Focus on logical precision and entity alignment."
            else:
                status = "OPTIMIZED_CONNECTIVITY"
                context_advice = (
                    "Global connectivity is healthy. You MUST now perform a deep-dive "
                    "audit on local logic (e.g., Node-Edge type pairing and semantic accuracy)."
                )

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
            logger.error(
                "structural_metrics_translation_failed",
                error_msg=str(e),
                tool_output_excerpt=tool_output[:100] if tool_output else "",
            )
            return f"### [STRUCTURAL CONTEXT DATA]\nWarning: Metrics unavailable. Proceed with standard audit."
