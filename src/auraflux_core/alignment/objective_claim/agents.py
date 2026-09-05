import json
from typing import Any, Dict, List, Optional

from auraflux_core.alignment.objective_claim.schemas import (
    DiagnosticAnalysis, ObjectiveClaimVerdict, TripleItem)
from auraflux_core.core.agents.base_agent import BaseAgent
from auraflux_core.core.agents.pipelines.plan_and_execute import \
    PlanAndExecuteHandler
from auraflux_core.core.schemas.messages import Message


class ObjectiveClaimAgent(BaseAgent, PlanAndExecuteHandler):
    """
    Specialized Alignment Agent for diagnosing and verifying objective claims.

    Acts as a Domain Provider:
    1. Inherits infrastructure capabilities from BaseAgent (LLM generation, ToolExecutor)[cite: 2].
    2. Implements PlanAndExecuteHandler to supply domain prompts, tool mapping, and output parsing
       to the decoupled PlanAndExecutePipeline without polluting BaseAgent.
    """

    def get_system_message_map(self) -> Dict[str, str]:
        return {
            "zh": (
                "你是一名客觀聲明稽核專家，負責針對單一原子命題進行正交診斷分析"
                "（隱性前提、標準與量化需求、邊界衝突），並根據檢索到的佐證資料進行最終驗證判決。"
            ),
            "default": (
                "You are an objective claim verification expert responsible for orthogonal "
                "diagnostic analysis and evidence-based verification."
            ),
        }

    # =========================================================================
    # PlanAndExecuteHandler Implementation (Domain Hooks for Pipeline)
    # =========================================================================

    def build_plan_messages(self, payload: Dict[str, Any]) -> List[Message]:
        """Stage 1 Hook: Builds the prompt to analyze claim and generate dynamic search query."""
        proposition_id = payload.get("proposition_id", "")
        claim_text = payload.get("claim_text", "")

        prompt = (
            f"Proposition ID: {proposition_id}\n"
            f"Claim: {claim_text}\n\n"
            "Task:\n"
            "1. Extract semantic triples (Subject -> Predicate -> Object).\n"
            "2. Perform orthogonal diagnosis (implicit_premises, quantification_requirements, boundary_conflicts).\n"
            "3. Generate an optimal search query (`query_text`) to retrieve core context evidence.\n\n"
            "CRITICAL FORMAT RULES:\n"
            "- `quantification_requirements` MUST BE A DICTIONARY OBJECT (NOT A LIST/ARRAY).\n\n"
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
            '  "query_text": "..."\n'
            "}"
        )
        return [Message(role="user", content=prompt, name=self.name)]

    def extract_tool_call_spec(self, plan_output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Stage 2 Hook: Maps Stage 1 LLM plan decision to concrete Tool execution specs."""
        query_text = plan_output.get("query_text")
        if not query_text:
            return None

        # Maps query intent to the registered 'hybrid_retriever' tool
        return {
            "tool_name": "hybrid_retriever",
            "tool_args": {
                "query_text": query_text,
                "top_k": 3
            }
        }

    def build_synthesis_messages(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any], tool_results: List[Message]
    ) -> List[Message]:
        """Stage 3 Hook: Injects retrieved evidence context for cross-checking synthesis."""
        proposition_id = payload.get("proposition_id", "")
        claim_text = payload.get("claim_text", "")

        evidence_text = "\n".join([msg.content for msg in tool_results if msg.role == "tool"])
        if not evidence_text:
            evidence_text = "No relevant context or records found in Core Context."

        prompt = (
            f"Proposition ID: {proposition_id}\n"
            f"Claim: {claim_text}\n"
            f"Diagnostics: {json.dumps(plan_output.get('diagnostics', {}), ensure_ascii=False)}\n"
            f"Retrieved Evidence:\n{evidence_text}\n\n"
            "Task:\n"
            "1. Cross-check the claim against the retrieved evidence.\n"
            "2. If fully supported, set status to 'VERIFIED' and list proofs in `verification_proofs`.\n"
            "3. If unsupported, set status to 'UNSUPPORTED' and specify `compliance_gap`.\n\n"
            "Output strict JSON format with keys: `status`, `verification_proofs`, `compliance_gap`."
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
        diagnostics = DiagnosticAnalysis(**plan_output.get("diagnostics", {}))

        status = parsed_data.get("status", "UNSUPPORTED")
        verification_proofs = parsed_data.get("verification_proofs", [])
        compliance_gap = parsed_data.get("compliance_gap")

        if status == "UNSUPPORTED" and not compliance_gap:
            compliance_gap = "Core Context contains no verifiable artifact or record supporting this claim."

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
        tool_args_map: Optional[Dict[str, Any]] = None,
    ) -> ObjectiveClaimVerdict:
        """
        Main execution facade for verifying an individual objective claim.
        Delegates the execution directly to the injected/configured Pipeline strategy.
        """
        payload = {
            "proposition_id": proposition_id,
            "claim_text": claim_text,
            "tool_args_map": tool_args_map,
        }

        # Executes the Strategy pipeline configured on the Agent instance
        return await self.pipeline.execute(agent=self, payload=payload)
