import json
from typing import Any, Dict, List, Optional, cast

from auraflux_core.alignment.pipelines import AlignmentHandler
from auraflux_core.alignment.schemas import TripleItem
from auraflux_core.alignment.threshold_claim.schemas import (
    NormalizedMetric, ThresholdClaimVerdict, ThresholdDiagnosticAnalysis)
from auraflux_core.core.agents.base_agent import BaseAgent
from auraflux_core.core.configs.logging_config import get_logger
from auraflux_core.core.schemas.messages import Message

logger = get_logger(__name__)


class ThresholdClaimAgent(BaseAgent, AlignmentHandler):
    """
    Specialized Alignment Agent for verifying quantitative thresholds and boundary claims (Gate 2/3).

    Acts as a Domain Provider:
    1. Inherits infrastructure capabilities from BaseAgent (LLM generation, ToolExecutor)[cite: 12].
    2. Inherits shared retrieval spec logic from BaseAlignmentHandler[cite: 12].
    3. Implements PlanAndExecuteHandler via BaseAlignmentHandler to execute a deterministic
       two-stage workflow (RAG Retrieval -> Deterministic Calculator) without LLM hallucination[cite: 12].
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
        """Stage 1 Hook: Builds prompt to analyze threshold claim and extract quantitative requirements matching NormalizedMetric schema."""
        return super().build_plan_messages(payload, agent)

    async def execute_tool_workflow(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any]
    ) -> List[Message]:
        """Stage 2 Hook: Code-controlled execution pipeline (RAG -> Calculator)."""
        tool_results: List[Message] = []

        # Step A: Execute hybrid_retriever to pull baseline policy/clause context
        retriever_spec = self._build_retriever_spec(payload, plan_output)
        if retriever_spec and self.tool_executor:
            rag_msg = await self.tool_executor.run(
                tool_name=retriever_spec["tool_name"],
                tool_args=retriever_spec["tool_args"]
            )
            tool_results.append(rag_msg)

        # Step B: Parse pre-normalized metrics from RAG tool output into standard NormalizedMetric objects
        claim_metrics = plan_output.get("diagnostics", {}).get("quantification_requirements", [])
        baseline_metrics = self._parse_baseline_metrics_from_rag(tool_results)

        # Step C: Pass explicitly decoupled claim_metrics and baseline_metrics to threshold_calculator
        if self.tool_executor and "threshold_calculator" in self.tool_executor.tool_registry:
            calc_msg = await self.tool_executor.run(
                tool_name="threshold_calculator",
                tool_args={
                    "claim_metrics": claim_metrics,
                    "baseline_metrics": baseline_metrics
                }
            )
            tool_results.append(calc_msg)

        return tool_results

    def _parse_baseline_metrics_from_rag(self, tool_results: List[Message]) -> List[Dict[str, Any]]:
        """Extracts normalized metrics directly attached to semantic triples from RAG context."""
        extracted_metrics: List[Dict[str, Any]] = []

        for msg in tool_results:
            if msg.role != "tool":
                continue

            raw_content = msg.content
            if not raw_content:
                continue

            if isinstance(raw_content, str):
                try:
                    parsed_data = json.loads(raw_content)
                except json.JSONDecodeError:
                    continue
            else:
                parsed_data = raw_content

            items = parsed_data if isinstance(parsed_data, list) else [parsed_data]

            for item in items:
                content = item.get("content", item) if isinstance(item, dict) else {}
                triples_list = content.get("triples", []) if isinstance(content, dict) else []

                for t in triples_list:
                    if isinstance(t, dict) and t.get("metric_name") and t.get("normalized_value") is not None:
                        raw_val = t.get("normalized_value")
                        extracted_metrics.append({
                            "raw_text": t.get("object", ""),
                            "metric_name": t.get("metric_name"),
                            "normalized_value": float(raw_val) if raw_val is not None else 0.0,
                            "unit": t.get("unit", ""),
                            "operator": t.get("operator", "<=")
                        })

        return extracted_metrics

    def get_synthesis_context(
        self,
        payload: Dict[str, Any],
        plan_output: Dict[str, Any],
        tool_results: List[Message]
    ) -> Dict[str, Any]:
        quantification_requirements = json.dumps(
            plan_output.get('diagnostics', {}).get('quantification_requirements', []),
            ensure_ascii=False
        )
        calc_summary, formatted_evidence_text = self._format_retrieved_evidence(tool_results)

        return {
            "proposition_id": payload.get("proposition_id", ""),
            "claim_text": payload.get("claim_text", ""),
            "calc_summary": calc_summary,
            "formatted_evidence_text": formatted_evidence_text,
            "quantification_requirements": quantification_requirements
        }

    def build_synthesis_messages(
        self,
        payload: Dict[str, Any],
        plan_output: Dict[str, Any],
        tool_results: List[Message],
        agent: "BaseAgent",
    ) -> List[Message]:
        """Stage 3 Hook: Synthesizes deterministic calculator results and evidence into a structured ThresholdClaimVerdict."""
        return super().build_synthesis_messages(payload, plan_output, tool_results, agent)

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

        status = parsed_data.get("status", "UNSUPPORTED")
        preset_options = parsed_data.get("preset_options", [])
        raw_compliance_gap = parsed_data.get("compliance_gap")
        verification_proofs = parsed_data.get("verification_proofs", [])

        if isinstance(raw_compliance_gap, (dict, list)):
            compliance_gap = json.dumps(raw_compliance_gap, ensure_ascii=False)
        elif raw_compliance_gap is not None:
            compliance_gap = str(raw_compliance_gap)
        else:
            compliance_gap = None

        if status in ["VIOLATED", "PARTIALLY_VERIFIED", "UNSUPPORTED", "FAIL", "PARADOX", "BORDERLINE"] and not compliance_gap:
            compliance_gap = "Threshold condition failed, unsupported, or triggered boundary conflict based on hard calculation."
        elif status in ["VERIFIED", "PASS"]:
            compliance_gap = None

        verdict = ThresholdClaimVerdict(
            proposition_id=payload.get("proposition_id", ""),
            claim_text=payload.get("claim_text", ""),
            triples=triples,
            diagnostics=diagnostics,
            status=status,
            preset_options=preset_options,
            compliance_gap=compliance_gap,
            verification_proofs=verification_proofs,
        )

        logger.info(
            "parse_final_output_completed",
            proposition_id=verdict.proposition_id,
            status=verdict.status,
            has_compliance_gap=verdict.compliance_gap is not None,
            preset_options_count=len(verdict.preset_options),
        )

        return verdict

    def _format_retrieved_evidence(self, tool_results: List[Message]) -> tuple[str, str]:
        """Helper to separate calculator output and format retrieved baseline policy context."""
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
            if not raw_content:
                continue

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

        formatted_evidence = "\n".join(extracted_docs) if extracted_docs else "No relevant policy records found."
        return calc_summary, formatted_evidence

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
    ) -> ThresholdClaimVerdict:
        """
        Main execution facade for verifying a threshold/quantitative claim.
        Delegates execution directly to the bound Pipeline strategy via agent.run()[cite: 12].
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

        # Executes via BaseAgent.run(), which delegates directly to self.pipeline[cite: 12]
        return await self.run(payload)
