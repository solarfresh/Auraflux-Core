import json
from typing import Any, Dict, List, Optional

from auraflux_core.alignment.pipelines import AlignmentHandler
from auraflux_core.alignment.threshold_claim.schemas import (
    NormalizedMetric, ThresholdClaimVerdict, ThresholdDiagnosticAnalysis,
    TripleItem)
from auraflux_core.core.agents.base_agent import BaseAgent
from auraflux_core.core.schemas.messages import Message


class ThresholdClaimAgent(BaseAgent, AlignmentHandler):
    """
    Specialized Alignment Agent for verifying quantitative thresholds and boundary claims (Gate 2/3).

    Acts as a Domain Provider:
    1. Inherits infrastructure capabilities from BaseAgent (LLM generation, ToolExecutor)[cite: 2].
    2. Inherits shared retrieval spec logic from BaseAlignmentHandler.
    3. Implements PlanAndExecuteHandler via BaseAlignmentHandler to execute a deterministic
       two-stage workflow (RAG Retrieval -> Deterministic Calculator) without LLM hallucination.
    """

    def get_system_message_map(self) -> Dict[str, str]:
        return {
            "zh": (
                "你是一名數據門檻與量化對焦稽核專家（Threshold & Quantitative Verification Specialist）。\n"
                "你的核心任務是針對傳入命題進行數據門檻與邊界衝突分析：\n"
                "1. 實施「單位標準化與量化診斷」，提取明確可稽核的數據指標（如金額、期限、SLA 等）與硬性條件邊界。\n"
                "2. 構建精準的事證檢索計畫，撈取相關基線條款。\n"
                "在驗證階段，你必須嚴格依據精確運算的結果進行事實導向的判決（PASS / FAIL / PARADOX / BORDERLINE），絕不允許主觀臆測。"
            ),
            "default": (
                "You are a Threshold & Quantitative Verification Specialist responsible for metric normalization "
                "and boundary checking of input claims.\n"
                "Your core responsibilities:\n"
                "1. Perform unit normalization and quantitative diagnostics to extract precise metrics and boundary criteria.\n"
                "2. Formulate targeted retrieval strategies to fetch relevant baseline policies or clauses.\n"
                "During verification, you must strictly adhere to exact mathematical computation results "
                "to render fact-based determinations (PASS / FAIL / PARADOX / BORDERLINE) without subjective hallucination."
            ),
        }

    # =========================================================================
    # PlanAndExecuteHandler Implementation (Domain Hooks for Pipeline)
    # =========================================================================

    def build_plan_messages(self, payload: Dict[str, Any]) -> List[Message]:
        """Stage 1 Hook: Builds prompt to analyze threshold claim and extract quantitative requirements matching NormalizedMetric schema."""
        proposition_id = payload.get("proposition_id", "")
        claim_text = payload.get("claim_text", "")

        prompt = (
            f"Proposition ID: {proposition_id}\n"
            f"Claim: {claim_text}\n\n"
            "Task:\n"
            "1. Extract semantic triples (Subject -> Predicate -> Object).\n"
            "2. Perform quantitative threshold diagnosis:\n"
            "   - implicit_premises: Unstated baseline assumptions or environmental factors.\n"
            "   - quantification_requirements: Extract explicit metric targets and normalize them strictly according to NormalizedMetric schema.\n"
            "   - boundary_conflicts: Detect potential paradoxes or mathematical boundary conflicts.\n"
            "3. Formulate targeted retrieval queries (`queries`). Generate 1 to 2 distinct query items "
            "tailored to fetch baseline clauses or policy rules:\n"
            "   - `query_text`: The rewritten query string optimized for retrieval (MUST BE IN ENGLISH).\n"
            "   - `target_type`: Chosen strictly from ['question', 'concept_title', 'concept_desc', 'evidence'].\n"
            "   - `text_field`: Exactly ONE text field name from ['target_question', 'concept_title', 'concept_description', 'evidence_text'].\n"
            "   - `vector_field`: Exactly ONE vector field name from ['question_vector', 'concept_vector', 'evidence_vector'].\n\n"
            "CRITICAL FORMAT RULES:\n"
            "- `query_text` MUST BE IN ENGLISH regardless of the input claim language.\n"
            "- `quantification_requirements` MUST BE A LIST of NormalizedMetric objects.\n"
            "- `queries` MUST BE A NON-EMPTY ARRAY containing 1 or 2 targeted query objects.\n\n"
            "Required Output JSON Format:\n"
            "{\n"
            '  "triples": [\n'
            '    {"subject": "...", "predicate": "...", "object": "..."}\n'
            '  ],\n'
            '  "diagnostics": {\n'
            '    "implicit_premises": ["..."],\n'
            '    "quantification_requirements": [\n'
            '      {\n'
            '        "raw_text": "500萬TWD",\n'
            '        "metric_name": "budget",\n'
            '        "normalized_value": 5000000.0,\n'
            '        "unit": "TWD",\n'
            '        "operator": "<="\n'
            '      }\n'
            '    ],\n'
            '    "boundary_conflicts": {\n'
            '      "has_conflict": false\n'
            '    }\n'
            '  },\n'
            '  "queries": [\n'
            '    {\n'
            '      "target_type": "evidence",\n'
            '      "query_text": "...",\n'
            '      "text_field": "evidence_text",\n'
            '      "vector_field": "evidence_vector"\n'
            '    }\n'
            '  ]\n'
            "}"
        )
        return [Message(role="user", content=prompt, name=self.name)]

    async def execute_tool_workflow(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any]
    ) -> List[Message]:
        """Stage 2 Hook: Deterministic code-controlled execution pipeline (RAG -> Calculator)."""
        tool_results: List[Message] = []

        # Step A: Execute hybrid_retriever to pull baseline policy/clause context
        retriever_spec = self._build_retriever_spec(payload, plan_output)
        if retriever_spec and self.tool_executor:
            rag_msg = await self.tool_executor.run(
                tool_name=retriever_spec["tool_name"],
                tool_args=retriever_spec["tool_args"]
            )
            tool_results.append(rag_msg)

        # Step B: Deterministic pass to threshold_calculator to eliminate LLM arithmetic errors
        norm_metrics = plan_output.get("diagnostics", {}).get("quantification_requirements", [])

        if self.tool_executor and "threshold_calculator" in self.tool_executor.tool_registry:
            calc_msg = await self.tool_executor.run(
                tool_name="threshold_calculator",
                tool_args={
                    "normalized_metrics": norm_metrics,
                    "retrieved_context": [m.content for m in tool_results]
                }
            )
            tool_results.append(calc_msg)

        return tool_results

    def build_synthesis_messages(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any], tool_results: List[Message]
    ) -> List[Message]:
        """Stage 3 Hook: Synthesizes deterministic calculator results and evidence into a structured ThresholdClaimVerdict."""
        proposition_id = payload.get("proposition_id", "")
        claim_text = payload.get("claim_text", "")

        extracted_docs: List[str] = []
        calc_summary: str = "No calculator result returned."

        for msg in tool_results:
            if msg.role != "tool":
                continue

            # Process calculator output separately if available
            if "calculator" in msg.name.lower() or "overflow" in msg.content.lower():
                calc_summary = msg.content
                continue

            raw_content = msg.content
            if isinstance(raw_content, str):
                try:
                    parsed_data = json.loads(raw_content)
                except json.JSONDecodeError:
                    parsed_data = [{"evidence_text": raw_content}]
            else:
                continue

            if isinstance(parsed_data, dict):
                retrieval_results = [parsed_data]
            elif isinstance(parsed_data, list):
                retrieval_results = parsed_data
            else:
                retrieval_results = []

            for result in retrieval_results:
                content = result.get('content', {})
                if not isinstance(content, dict):
                    continue

                excerpt_text = content.get("evidence_text", "N/A")
                location = content.get("location", "N/A")

                doc_block = (
                    f"• Source Location: {location}\n"
                    f"• Baseline Policy Excerpt: \"{excerpt_text}\"\n"
                )
                extracted_docs.append(doc_block)

        formatted_evidence_text = "\n".join(extracted_docs) if extracted_docs else "No relevant policy records found."

        prompt = (
            f"Proposition ID: {proposition_id}\n"
            f"Claim: {claim_text}\n"
            f"Stage 1 Quantification Requirements: {json.dumps(plan_output.get('diagnostics', {}).get('quantification_requirements', []), ensure_ascii=False)}\n\n"
            f"Deterministic Calculator Computation Result (HARD GROUND TRUTH):\n{calc_summary}\n\n"
            f"Retrieved Baseline Policy Context:\n{formatted_evidence_text}\n\n"
            "Task:\n"
            "1. Evaluate claim against the Deterministic Calculator Computation Result.\n"
            "2. YOU MUST STRICTLY ADHERE to the calculator result status:\n"
            "   - 'PASS': Met boundary requirements.\n"
            "   - 'FAIL': Conflict or out-of-bounds violation.\n"
            "   - 'PARADOX': Direct contradiction in rules or dual-metric paradox.\n"
            "   - 'BORDERLINE': Vague or boundary-adjacent value.\n"
            "3. If status is 'BORDERLINE' or 'PARADOX', provide trade-off choices in `preset_options`.\n"
            "4. Fill `compliance_gap` with detailed explanation of the numeric paradox or boundary overflow, or null if PASS.\n\n"
            "Required Output JSON Format:\n"
            "{\n"
            '  "status": "PASS" | "FAIL" | "PARADOX" | "BORDERLINE",\n'
            '  "preset_options": ["Option 1...", "Option 2..."],\n'
            '  "compliance_gap": "Detailed explanation string or null"\n'
            "}"
        )

        return [Message(role="user", content=prompt, name=self.name)]

    def parse_final_output(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any], raw_llm_output: str
    ) -> ThresholdClaimVerdict:
        """Domain Transformation Hook: Maps Stage 1 & Stage 3 outputs into ThresholdClaimVerdict Schema."""
        parsed_data = self.output_parser.parse_json(raw_llm_output)

        triples = [
            TripleItem(**item) for item in plan_output.get("triples", [])
        ]

        diagnostics_data = plan_output.get("diagnostics", {})

        # Ensure quantification_requirements matches List[NormalizedMetric]
        raw_metrics = diagnostics_data.get("quantification_requirements", [])
        normalized_metrics = [
            NormalizedMetric(**m) if isinstance(m, dict) else m for m in raw_metrics
        ]

        diagnostics = ThresholdDiagnosticAnalysis(
            implicit_premises=diagnostics_data.get("implicit_premises", []),
            quantification_requirements=normalized_metrics,
            boundary_conflicts=diagnostics_data.get("boundary_conflicts", {})
        )

        status = parsed_data.get("status", "FAIL")
        preset_options = parsed_data.get("preset_options", [])
        raw_compliance_gap = parsed_data.get("compliance_gap")

        if isinstance(raw_compliance_gap, (dict, list)):
            compliance_gap = json.dumps(raw_compliance_gap, ensure_ascii=False)
        elif raw_compliance_gap is not None:
            compliance_gap = str(raw_compliance_gap)
        else:
            compliance_gap = None

        if status in ["FAIL", "PARADOX", "BORDERLINE"] and not compliance_gap:
            compliance_gap = "Threshold condition failed or triggered boundary conflict based on hard calculation."
        elif status == "PASS":
            compliance_gap = None

        return ThresholdClaimVerdict(
            proposition_id=payload.get("proposition_id", ""),
            claim_text=payload.get("claim_text", ""),
            triples=triples,
            diagnostics=diagnostics,
            status=status,
            preset_options=preset_options,
            compliance_gap=compliance_gap,
        )

    # =========================================================================
    # Facade / Entrypoint
    # =========================================================================

    async def verify_threshold(
        self,
        proposition_id: str,
        claim_text: str,
        routing_key: Optional[str] = None,
        index_name: Optional[str] = None,
        tool_args_map: Optional[Dict[str, Any]] = None,
    ) -> ThresholdClaimVerdict:
        """
        Main execution facade for verifying a threshold/quantitative claim.
        Delegates execution directly to the bound Pipeline strategy via agent.run()[cite: 2].
        """
        payload: Dict[str, Any] = {
            "proposition_id": proposition_id,
            "claim_text": claim_text,
            "tool_args_map": tool_args_map,
        }

        if routing_key is not None:
            payload["routing_key"] = routing_key
        if index_name is not None:
            payload["index_name"] = index_name

        # Executes via BaseAgent.run(), which delegates directly to self.pipeline[cite: 2]
        return await self.run(payload)
