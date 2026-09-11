import json
from typing import Any, Dict, List, Optional

from auraflux_core.alignment.objective_claim.schemas import (
    ObjectiveDiagnosticAnalysis, ObjectiveClaimVerdict, TripleItem)
from auraflux_core.alignment.pipelines import AlignmentHandler
from auraflux_core.core.agents.base_agent import BaseAgent
from auraflux_core.core.schemas.messages import Message


class ObjectiveClaimAgent(BaseAgent, AlignmentHandler):
    """
    Specialized Alignment Agent for diagnosing and verifying objective claims.

    Acts as a Domain Provider:
    1. Inherits infrastructure capabilities from BaseAgent (LLM generation, ToolExecutor)[cite: 2].
    2. Inherits shared retrieval spec logic from BaseAlignmentHandler.
    3. Implements PlanAndExecuteHandler via BaseAlignmentHandler to supply domain prompts,
       a deterministic tool execution workflow, and output parsing to PlanAndExecutePipeline[cite: 2].
    """

    def get_system_message_map(self) -> Dict[str, str]:
        return {
            "zh": (
                "你是一名客觀聲明稽核專家（Objective Claim Audit Specialist）。\n"
                "你的核心任務是針對傳入命題進行正交診斷分析（Orthogonal Diagnostics）：\n"
                "1. 剖析隱性前提（資源授權、團隊能力與因果依賴）。\n"
                "2. 實施「標準與量化診斷」，剝離主觀形容與抽象描述，提取明確可稽核的數據指標與所需產物類型（Required Artifact Types, 如 Log、 Commit、驗收報告）。\n"
                "3. 檢測足跡衝突（Footprint Conflicts）並構建精準的事證檢索計畫。\n"
                "在驗證階段，你必須嚴格基於檢索到的客觀佐證進行事實導向的判決（VERIFIED / PARTIALLY_VERIFIED / UNSUPPORTED），絕不允許主觀臆測。"
            ),
            "default": (
                "You are an Objective Claim Audit Specialist responsible for orthogonal diagnostic analysis "
                "and evidence-based verification of input claims.\n"
                "Your core responsibilities:\n"
                "1. Analyze implicit premises (resource authorization, capability assumptions, and causality).\n"
                "2. Execute standardization and quantification diagnostics to strip subjective phrasing and extract audit-ready metrics alongside required artifact types (e.g., logs, commits, spec docs).\n"
                "3. Detect footprint conflicts and formulate targeted evidence retrieval strategies.\n"
                "During verification, you must strictly evaluate claims against retrieved objective evidence "
                "to render fact-based determinations (VERIFIED / PARTIALLY_VERIFIED / UNSUPPORTED) without subjective hallucination."
            ),
        }

    # =========================================================================
    # PlanAndExecuteHandler Implementation (Domain Hooks for Pipeline)
    # =========================================================================

    def build_plan_messages(self, payload: Dict[str, Any]) -> List[Message]:
        """Stage 1 Hook: Builds prompt to analyze claim and specify 1-to-1 targeted retrieval fields."""
        proposition_id = payload.get("proposition_id", "")
        claim_text = payload.get("claim_text", "")

        prompt = (
            f"Proposition ID: {proposition_id}\n"
            f"Claim: {claim_text}\n\n"
            "Task:\n"
            "1. Extract semantic triples (Subject -> Predicate -> Object).\n"
            "2. Perform orthogonal diagnosis:\n"
            "   - implicit_premises: Unstated assumptions regarding Capability, Resources, Causality, or Environment.\n"
            "   - quantification_requirements: Clear metrics, required artifact types, and acceptance criteria.\n"
            "   - boundary_conflicts: Potential resource or rule conflicts.\n"
            "3. Formulate targeted retrieval queries (`queries`). Generate 1 to 2 distinct query items "
            "tailored to specific search intents. Each item must strictly bind to ONE text field and ONE vector field:\n"
            "   - `query_text`: The rewritten query string optimized for retrieval (MUST BE IN ENGLISH to match knowledge base).\n"
            "   - `target_type`: The retrieval intent, chosen strictly from ['question', 'concept_title', 'concept_desc', 'evidence']:\n"
            "       * 'question': Target underlying core questions. (text_field: 'target_question', vector_field: 'question_vector')\n"
            "       * 'concept_title': Target specific concept names/terms. (text_field: 'concept_title', vector_field: 'concept_vector')\n"
            "       * 'concept_desc': Target concept definitions and mechanisms. (text_field: 'concept_description', vector_field: 'concept_vector')\n"
            "       * 'evidence': Target concrete empirical facts or evidence. (text_field: 'evidence_text', vector_field: 'evidence_vector')\n"
            "   - `text_field`: Exactly ONE text field name from ['target_question', 'concept_title', 'concept_description', 'evidence_text'].\n"
            "   - `vector_field`: Exactly ONE vector field name from ['question_vector', 'concept_vector', 'evidence_vector'].\n\n"
            "CRITICAL FORMAT RULES:\n"
            "- `query_text` MUST BE IN ENGLISH regardless of the input claim language.\n"
            "- `quantification_requirements` MUST BE A DICTIONARY OBJECT (NOT A LIST/ARRAY).\n"
            "- `queries` MUST BE A NON-EMPTY ARRAY containing 1 or 2 targeted query objects.\n\n"
            "Required Output JSON Format:\n"
            "{\n"
            '  "triples": [\n'
            '    {"subject": "...", "predicate": "...", "object": "..."}\n'
            '  ],\n'
            '  "diagnostics": {\n'
            '    "implicit_premises": ["..."],\n'
            '    "quantification_requirements": {\n'
            '      "required_artifact_types": ["..."],\n'
            '      "acceptance_criteria": "..."\n'
            '    },\n'
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
        """Stage 2 Hook: Deterministic execution of the retrieval tool workflow."""
        tool_results: List[Message] = []
        retriever_spec = self._build_retriever_spec(payload, plan_output)

        if retriever_spec and self.tool_executor:
            rag_msg = await self.tool_executor.run(
                tool_name=retriever_spec["tool_name"],
                tool_args=retriever_spec["tool_args"]
            )
            tool_results.append(rag_msg)

        return tool_results

    def build_synthesis_messages(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any], tool_results: List[Message]
    ) -> List[Message]:
        """Stage 3 Hook: Extracts structured 4-layer context from tool results and injects for synthesis."""
        proposition_id = payload.get("proposition_id", "")
        claim_text = payload.get("claim_text", "")

        extracted_docs: List[str] = []
        tool_messages = [msg for msg in tool_results if msg.role == "tool"]

        doc_idx = 1
        for msg in tool_messages:
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

                target_question = content.get("target_question", "N/A")
                concept_title = content.get("concept_title", "N/A")
                concept_desc = content.get("concept_description", "N/A")
                triples = content.get("triples", [])

                excerpt_text = content.get("evidence_text", "N/A")
                location = content.get("location", "N/A")
                tags = content.get("tags", [])

                doc_block = (
                    f"--- [Document #{doc_idx}] ---\n"
                    f"• Source Location (Layer 4): {location}\n"
                    f"• Tags: {', '.join(tags) if tags else 'None'}\n"
                    f"• Target Question (Layer 1): {target_question}\n"
                    f"• Concept Description (Layer 2): [{concept_title}] {concept_desc}\n"
                    f"• Semantic Triples (Layer 3): {json.dumps(triples, ensure_ascii=False)}\n"
                    f"• Direct Excerpt Proof (Layer 4): \"{excerpt_text}\"\n"
                )
                extracted_docs.append(doc_block)
                doc_idx += 1

        if extracted_docs:
            formatted_evidence_text = "\n".join(extracted_docs)
        else:
            formatted_evidence_text = "No relevant context or records found in Core Context."

        prompt = (
            f"Proposition ID: {proposition_id}\n"
            f"Claim: {claim_text}\n"
            f"Extracted Claim Triples (Stage 1): {json.dumps(plan_output.get('triples', []), ensure_ascii=False)}\n"
            f"Diagnostics (Stage 1): {json.dumps(plan_output.get('diagnostics', {}), ensure_ascii=False)}\n\n"
            f"Retrieved Structured Core Context Evidence:\n{formatted_evidence_text}\n\n"
            "Task:\n"
            "1. Perform a thorough cross-examination of the Claim and Stage 1 Diagnostics against the Retrieved Evidence.\n"
            "2. Evaluate and EXPLICITLY OUTPUT the `diagnostic_evaluation` for each diagnostic item:\n"
            "   - `implicit_premises_eval`: Assess if the unstated assumptions are supported by evidence.\n"
            "   - `quantification_eval`: Assess if the quantitative requirements/metrics are fully satisfied by evidence.\n"
            "   - `boundary_conflicts_eval`: Detail whether the boundary conflicts identified in Stage 1 are resolved, mitigated, or violated by the evidence.\n"
            "3. Determine the final `status`:\n"
            "   - 'VERIFIED': Fully supported by evidence, all quantitative criteria met, and ALL boundary conflicts are fully resolved.\n"
            "   - 'PARTIALLY_VERIFIED': Main claim is supported, BUT missing quantitative metrics, unfulfilled criteria, or UNRESOLVED boundary conflicts remain.\n"
            "   - 'UNSUPPORTED': Insufficient evidence, direct factual contradiction, or major boundary conflict violation.\n"
            "4. Fill `verification_proofs` with exact supporting quotes, locations, or triples from evidence.\n"
            "5. Fill `compliance_gap`:\n"
            "   - If status is 'VERIFIED': MUST be set to `null`.\n"
            "   - If status is 'PARTIALLY_VERIFIED' or 'UNSUPPORTED': MUST be a string explicitly summarizing the gaps, unfulfilled metrics, or unresolved boundary conflicts.\n\n"
            "CRITICAL FORMAT RULES:\n"
            "- `diagnostic_evaluation` MUST be a nested object with text descriptions for all 3 diagnostic dimensions.\n"
            "- `compliance_gap` MUST BE A PLAIN STRING OR `null` (NEVER AN OBJECT OR LIST).\n\n"
            "Required Output JSON Format:\n"
            "{\n"
            '  "diagnostic_evaluation": {\n'
            '    "implicit_premises_eval": "Evaluation of assumptions against evidence...",\n'
            '    "quantification_eval": "Evaluation of metrics and criteria against evidence...",\n'
            '    "boundary_conflicts_eval": "Evaluation of boundary conflicts (whether resolved or remaining as gaps)..."\n'
            '  },\n'
            '  "status": "VERIFIED" | "PARTIALLY_VERIFIED" | "UNSUPPORTED",\n'
            '  "verification_proofs": ["..."],\n'
            '  "compliance_gap": "Brief description string of missing criteria/gaps/conflicts, or null if fully verified"\n'
            "}"
        )

        return [Message(role="user", content=prompt, name=self.name)]

    def parse_final_output(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any], raw_llm_output: str
    ) -> ObjectiveClaimVerdict:
        """Domain Transformation Hook: Maps Stage 1 & Stage 3 outputs into Pydantic Schema."""
        parsed_data = self.output_parser.parse_json(raw_llm_output)

        triples = [
            TripleItem(**item) for item in plan_output.get("triples", [])
        ]
        diagnostics = ObjectiveDiagnosticAnalysis(**plan_output.get("diagnostics", {}))

        diagnostic_eval = parsed_data.get("diagnostic_evaluation", {})
        boundary_eval = diagnostic_eval.get("boundary_conflicts_eval", "")

        status = parsed_data.get("status", "UNSUPPORTED")
        verification_proofs = parsed_data.get("verification_proofs", [])
        raw_compliance_gap = parsed_data.get("compliance_gap")

        if isinstance(raw_compliance_gap, (dict, list)):
            compliance_gap = json.dumps(raw_compliance_gap, ensure_ascii=False)
        elif raw_compliance_gap is not None:
            compliance_gap = str(raw_compliance_gap)
        else:
            compliance_gap = None

        if status == "UNSUPPORTED" and not compliance_gap:
            compliance_gap = f"Core Context contains no supporting evidence. Boundary evaluation: {boundary_eval}"
        elif status == "PARTIALLY_VERIFIED" and not compliance_gap:
            compliance_gap = f"Claim is partially verified. Remaining issues: {boundary_eval}"
        elif status == "VERIFIED":
            compliance_gap = None

        return ObjectiveClaimVerdict(
            proposition_id=payload.get("proposition_id", ""),
            claim_text=payload.get("claim_text", ""),
            triples=triples,
            diagnostics=diagnostics,
            status=status,
            verification_proofs=verification_proofs,
            compliance_gap=compliance_gap,
        )

    # =========================================================================
    # Facade / Entrypoint
    # =========================================================================

    async def diagnose_and_verify(
        self,
        proposition_id: str,
        claim_text: str,
        routing_key: Optional[str] = None,
        index_name: Optional[str] = None,
        tool_args_map: Optional[Dict[str, Any]] = None,
    ) -> ObjectiveClaimVerdict:
        """
        Main execution facade for verifying an individual objective claim.
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
