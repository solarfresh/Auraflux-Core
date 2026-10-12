import json
from typing import Any, Dict, List, Optional, cast

from auraflux_core.alignment.objective_claim.schemas import (
    ObjectiveClaimVerdict, ObjectiveDiagnosticAnalysis)
from auraflux_core.alignment.pipelines import AlignmentHandler
from auraflux_core.alignment.schemas import TripleItem
from auraflux_core.core.agents.base_agent import BaseAgent
from auraflux_core.core.configs.logging_config import get_logger
from auraflux_core.core.schemas.messages import Message

logger = get_logger(__name__)


class ObjectiveClaimAgent(BaseAgent, AlignmentHandler):
    """
    Specialized Alignment Agent for diagnosing and verifying objective claims.

    Acts as a Domain Provider:
    1. Inherits infrastructure capabilities from BaseAgent (LLM generation, ToolExecutor)[cite: 11].
    2. Inherits shared retrieval spec logic from BaseAlignmentHandler.
    3. Implements PlanAndExecuteHandler via BaseAlignmentHandler to supply domain prompts,
       a deterministic tool execution workflow, and output parsing to PlanAndExecutePipeline[cite: 11].
    """

    def get_plan_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            """Extracts context variables needed for Stage 1 planning template."""
            return {
                "proposition_id": payload.get("proposition_id", ""),
                "claim_text": payload.get("claim_text", "")
            }

    def build_plan_messages(
        self,
        payload: Dict[str, Any],
        agent: "BaseAgent"
    ) -> List[Message]:
        """Stage 1: Constructs user messages for LLM keyword & triple extraction.

        Attempts dynamic lookup via `prompt_formatter.format_messages`. If absent or
        not callable, falls back to the default Stage 1 extraction user message.
        """
        return super().build_plan_messages(payload, agent)

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

    def get_synthesis_context(
            self,
            payload: Dict[str, Any],
            plan_output: Dict[str, Any],
            tool_results: List[Message]
        ) -> Dict[str, Any]:
            return {
                "proposition_id": payload.get("proposition_id", ""),
                "claim_text": payload.get("claim_text", ""),
                "formatted_evidence_text": self._format_retrieved_evidence(tool_results),
                "triples": json.dumps(plan_output.get('triples', []), ensure_ascii=False),
                "diagnostics": json.dumps(plan_output.get('diagnostics', {}), ensure_ascii=False)
            }

    def build_synthesis_messages(
        self,
        payload: Dict[str, Any],
        plan_output: Dict[str, Any],
        tool_results: List[Message],
        agent: "BaseAgent",
    ) -> List[Message]:
        """Stage 3 Hook: Extracts structured 4-layer context from tool results and injects for synthesis."""
        return super().build_synthesis_messages(payload, plan_output, tool_results, agent)

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

        verdict = ObjectiveClaimVerdict(
            proposition_id=payload.get("proposition_id", ""),
            claim_text=payload.get("claim_text", ""),
            triples=triples,
            diagnostics=diagnostics,
            status=status,
            verification_proofs=verification_proofs,
            compliance_gap=compliance_gap,
        )

        logger.info(
            "parse_final_output_completed",
            proposition_id=verdict.proposition_id,
            status=verdict.status,
            has_compliance_gap=verdict.compliance_gap is not None,
        )

        return verdict

    def _format_retrieved_evidence(self, tool_results: List[Message]) -> str:
        """Helper to format structured 4-layer context documents from tool execution messages."""
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
            return "\n".join(extracted_docs)
        return "No relevant context or records found in Core Context."

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
        Delegates execution directly to the bound Pipeline strategy via agent.run()[cite: 11].
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

        # Executes via BaseAgent.run(), which delegates directly to self.pipeline[cite: 11]
        return await self.run(payload)
