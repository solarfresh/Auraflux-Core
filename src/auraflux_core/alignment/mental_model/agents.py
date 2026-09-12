import json
from typing import Any, Dict, List, Optional

from auraflux_core.alignment.mental_model.schemas import (
    FourDimensionAssumptionMatrix, MentalModelClaimVerdict,
    MentalModelDiagnosticAnalysis)
from auraflux_core.alignment.schemas import TripleItem
from auraflux_core.alignment.pipelines import AlignmentHandler
from auraflux_core.core.agents.base_agent import BaseAgent
from auraflux_core.core.schemas.messages import Message


class MentalModelAgent(BaseAgent, AlignmentHandler):
    """
    Specialized Alignment Agent for diagnosing subjective mental-model
    (expectation/belief) claims.

    Acts as a Domain Provider:
    1. Inherits infrastructure capabilities from BaseAgent (LLM generation, ToolExecutor).
    2. Inherits shared retrieval spec logic from BaseAlignmentHandler (retrieval-only path,
       same as ObjectiveClaimAgent — no deterministic calculation step is involved).
    3. Implements PlanAndExecuteHandler via BaseAlignmentHandler to supply domain prompts,
       a retrieval-based tool execution workflow, and output parsing to PlanAndExecutePipeline.
    """

    def get_system_message_map(self) -> Dict[str, str]:
        return {
            "zh": (
                "你是一名心智模型與主觀期待稽核專家（Mental Model Audit Specialist）。\n"
                "你的核心任務是針對傳入命題進行正交診斷分析（Orthogonal Diagnostics）：\n"
                "1. 【主戰場】發動四維度防遺漏矩陣（Capability, Resources, Causality, Environment），"
                "剖析主體對另一實體隱含且未經證實的認知假設。\n"
                "2. 標註主觀期待中的模糊形容詞與抽象概念，指出待量化的指標（如 SLA 時間、驗收標準）。\n"
                "3. 捕捉主觀期待與客觀規章/限制間可能存在的認知落差，並構建精準的事證檢索計畫。\n"
                "在驗證階段，你必須嚴格基於檢索到的客觀證據判斷主觀期待是否成立，"
                "判決結果為 VERIFIED（認知吻合）/ VIOLATED（證實落差）/ UNSUPPORTED（關鍵假設無佐證），絕不允許臆測評分。"
            ),
            "default": (
                "You are a Mental Model Audit Specialist responsible for orthogonal diagnostic analysis "
                "of subjective expectations, beliefs, and implicit assumptions held by one party about another.\n"
                "Your core responsibilities:\n"
                "1. [Main battlefield] Run the four-dimension assumption matrix "
                "(Capability, Resources, Causality, Environment) to surface implicit, "
                "unverified cognitive assumptions.\n"
                "2. Flag vague subjective adjectives and abstract expectations, identifying which "
                "metrics still require quantification (e.g., SLA duration, acceptance criteria).\n"
                "3. Detect potential gaps between the subjective expectation and objective policy/"
                "constraints, and formulate a targeted evidence retrieval strategy.\n"
                "During verification, you must judge whether the subjective expectation holds strictly "
                "against retrieved objective evidence, rendering VERIFIED (aligned) / VIOLATED (confirmed gap) / "
                "UNSUPPORTED (key assumption has no evidence) without subjective hallucination."
            ),
        }

    # =========================================================================
    # PlanAndExecuteHandler Implementation (Domain Hooks for Pipeline)
    # =========================================================================

    def build_plan_messages(self, payload: Dict[str, Any]) -> List[Message]:
        """Stage 1 Hook: Builds prompt to analyze the expectation and specify targeted retrieval fields."""
        proposition_id = payload.get("proposition_id", "")
        claim_text = payload.get("claim_text", "")

        prompt = (
            f"Proposition ID: {proposition_id}\n"
            f"Claim: {claim_text}\n\n"
            "Task:\n"
            "1. Extract semantic triples (Subject -> Predicate -> Object).\n"
            "2. Perform orthogonal diagnosis:\n"
            "   - implicit_premises (MAIN BATTLEFIELD): Run the 4D assumption matrix. For EACH of the "
            "four dimensions below, list unverified assumptions found in the claim (empty array if none):\n"
            "       * capability_assumptions: assumptions about the counterparty's technical/execution capability.\n"
            "       * resource_assumptions: assumptions about resource, budget, or authorization scope.\n"
            "       * causality_assumptions: unexamined links taken for granted in the causal chain.\n"
            "       * environment_assumptions: assumptions that the external environment stays static.\n"
            "   - quantification_requirements: Vague subjective adjectives/abstract expectations mapped to "
            "the metric that still needs to be quantified (e.g. {'quickly': 'needs SLA in days'}).\n"
            "   - boundary_conflicts: Potential gaps between this subjective expectation and known policy/"
            "constraints (has_conflict flag + description).\n"
            "3. Formulate targeted retrieval queries (`queries`) to search Core Context for evidence that "
            "supports or refutes this expectation (e.g., design docs, prior commitments, org policy). "
            "Generate 1 to 2 distinct query items tailored to specific search intents. Each item must "
            "strictly bind to ONE text field and ONE vector field:\n"
            "   - `query_text`: The rewritten query string optimized for retrieval (MUST BE IN ENGLISH).\n"
            "   - `target_type`: chosen strictly from ['question', 'concept_title', 'concept_desc', 'evidence']:\n"
            "       * 'question': (text_field: 'target_question', vector_field: 'question_vector')\n"
            "       * 'concept_title': (text_field: 'concept_title', vector_field: 'concept_vector')\n"
            "       * 'concept_desc': (text_field: 'concept_description', vector_field: 'concept_vector')\n"
            "       * 'evidence': (text_field: 'evidence_text', vector_field: 'evidence_vector')\n"
            "   - `text_field`: Exactly ONE text field name from ['target_question', 'concept_title', "
            "'concept_description', 'evidence_text'].\n"
            "   - `vector_field`: Exactly ONE vector field name from ['question_vector', 'concept_vector', "
            "'evidence_vector'].\n\n"
            "CRITICAL FORMAT RULES:\n"
            "- `query_text` MUST BE IN ENGLISH regardless of the input claim language.\n"
            "- `quantification_requirements` and `boundary_conflicts` MUST BE DICTIONARY OBJECTS (NOT ARRAYS).\n"
            "- `implicit_premises` MUST be an object with exactly the four keys listed above, each an array.\n"
            "- `queries` MUST BE A NON-EMPTY ARRAY containing 1 or 2 targeted query objects.\n\n"
            "Required Output JSON Format:\n"
            "{\n"
            '  "triples": [\n'
            '    {"subject": "...", "predicate": "...", "object": "..."}\n'
            '  ],\n'
            '  "diagnostics": {\n'
            '    "implicit_premises": {\n'
            '      "capability_assumptions": ["..."],\n'
            '      "resource_assumptions": ["..."],\n'
            '      "causality_assumptions": ["..."],\n'
            '      "environment_assumptions": ["..."]\n'
            '    },\n'
            '    "quantification_requirements": {"...": "..."},\n'
            '    "boundary_conflicts": {"has_conflict": false}\n'
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
        """Stage 2 Hook: Deterministic execution of the retrieval tool workflow.

        NOTE: Identical to ObjectiveClaimAgent's implementation — mental_model,
        like objective_claim, is a pure retrieval consumer with no extra
        rule-based calculation step (unlike threshold_claim_agent, which layers
        a deterministic calculator on top of retrieval). If a third retrieval-only
        agent is ever added, this method should be hoisted into a shared
        RetrievalOnlyAlignmentHandler mixin instead of copy-pasted again.
        """
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

        # NOTE: This extraction/formatting block is byte-for-byte identical to
        # ObjectiveClaimAgent's version. It is agent-agnostic (pure retrieval-result
        # formatting) and should be extracted into a shared
        # AlignmentHandler.format_retrieved_evidence(tool_results) helper rather
        # than duplicated a second time here.
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
            f"Claim (Subjective Expectation): {claim_text}\n"
            f"Extracted Claim Triples (Stage 1): {json.dumps(plan_output.get('triples', []), ensure_ascii=False)}\n"
            f"Diagnostics (Stage 1): {json.dumps(plan_output.get('diagnostics', {}), ensure_ascii=False)}\n\n"
            f"Retrieved Structured Core Context Evidence:\n{formatted_evidence_text}\n\n"
            "Task:\n"
            "1. Perform a thorough cross-examination of the subjective expectation and Stage 1 4D "
            "assumption matrix against the Retrieved Evidence.\n"
            "2. Evaluate and EXPLICITLY OUTPUT `diagnostic_evaluation` for each diagnostic dimension:\n"
            "   - `capability_eval`: Is the capability assumption supported by evidence?\n"
            "   - `resource_eval`: Is the resource/authorization assumption supported by evidence?\n"
            "   - `causality_eval`: Does evidence support the assumed causal chain?\n"
            "   - `environment_eval`: Does evidence confirm the environment assumption still holds?\n"
            "   - `quantification_eval`: Are the flagged vague expectations now quantifiable from evidence?\n"
            "   - `boundary_conflicts_eval`: Is the subjective expectation in conflict with objective "
            "policy/constraints found in evidence?\n"
            "3. Determine the final `status`:\n"
            "   - 'VERIFIED': The subjective expectation is fully aligned with retrieved evidence "
            "(equivalent to ALIGNED) — no unresolved assumption or conflict remains.\n"
            "   - 'VIOLATED': Evidence confirms a cognitive gap between the expectation and actual "
            "state (equivalent to MISALIGNED).\n"
            "   - 'UNSUPPORTED': A key assumption in the 4D matrix has no supporting or refuting "
            "evidence at all (equivalent to OPAQUE_RISK) — this MUST trigger a BLOCK requiring "
            "human clarification.\n"
            "4. Fill `verification_proofs` with exact supporting quotes, locations, or triples from evidence.\n"
            "5. Fill `compliance_gap`:\n"
            "   - If status is 'VERIFIED': MUST be set to `null`.\n"
            "   - If status is 'VIOLATED' or 'UNSUPPORTED': MUST be a string explicitly summarizing the "
            "gap, unresolved assumption, or conflict.\n\n"
            "CRITICAL FORMAT RULES:\n"
            "- `diagnostic_evaluation` MUST be a nested object with text descriptions for all 6 dimensions above.\n"
            "- `compliance_gap` MUST BE A PLAIN STRING OR `null` (NEVER AN OBJECT OR LIST).\n\n"
            "Required Output JSON Format:\n"
            "{\n"
            '  "diagnostic_evaluation": {\n'
            '    "capability_eval": "...",\n'
            '    "resource_eval": "...",\n'
            '    "causality_eval": "...",\n'
            '    "environment_eval": "...",\n'
            '    "quantification_eval": "...",\n'
            '    "boundary_conflicts_eval": "..."\n'
            '  },\n'
            '  "status": "VERIFIED" | "VIOLATED" | "UNSUPPORTED",\n'
            '  "verification_proofs": ["..."],\n'
            '  "compliance_gap": "Brief description string of the gap/unresolved assumption, or null"\n'
            "}"
        )

        return [Message(role="user", content=prompt, name=self.name)]

    def parse_final_output(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any], raw_llm_output: str
    ) -> MentalModelClaimVerdict:
        """Domain Transformation Hook: Maps Stage 1 & Stage 3 outputs into Pydantic Schema."""
        parsed_data = self.output_parser.parse_json(raw_llm_output)

        triples = [
            TripleItem(**item) for item in plan_output.get("triples", [])
        ]

        raw_diagnostics = plan_output.get("diagnostics", {})
        raw_implicit_premises = raw_diagnostics.get("implicit_premises", {})
        diagnostics = MentalModelDiagnosticAnalysis(
            implicit_premises=FourDimensionAssumptionMatrix(**raw_implicit_premises),
            quantification_requirements=raw_diagnostics.get("quantification_requirements", {}),
            boundary_conflicts=raw_diagnostics.get("boundary_conflicts", {}),
        )

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
            compliance_gap = f"A key assumption in the 4D matrix has no supporting evidence. Boundary evaluation: {boundary_eval}"
        elif status == "VIOLATED" and not compliance_gap:
            compliance_gap = f"Subjective expectation conflicts with evidence. Boundary evaluation: {boundary_eval}"
        elif status == "VERIFIED":
            compliance_gap = None

        return MentalModelClaimVerdict(
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
    ) -> MentalModelClaimVerdict:
        """
        Main execution facade for verifying an individual mental-model (subjective
        expectation) claim. Delegates execution directly to the bound Pipeline
        strategy via agent.run().
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

        # Executes via BaseAgent.run(), which delegates directly to self.pipeline
        return await self.run(payload)
