from typing import Any, Dict, List

from auraflux_core.canvases.tools import OntologyValidatorTool
from auraflux_core.core.agents.pipelines.plan_and_execute import \
    PlanAndExecuteHandler
from auraflux_core.core.configs.logging_config import get_logger
from auraflux_core.core.schemas.messages import Message
from auraflux_core.core.schemas.pipelines import ValidationResult

logger = get_logger(__name__)


class CanvasPlanAndExecuteHandler(PlanAndExecuteHandler):
    """
    Canvas pipeline handler extending PlanAndExecuteHandler.
    Integrates OntologyValidatorTool to validate nodes and edges,
    converting validation outputs into ValidationResult with actionable LLM guidance.
    """

    async def inspect_plan_output(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any]
    ) -> ValidationResult:
        """
        Stage 1.5 Inspection Hook:
        Validates nodes and edges using OntologyValidatorTool.
        Returns ValidationResult containing is_valid, reasons (formatted as English LLM instructions), and metadata.
        """
        try:
            nodes = plan_output.get("nodes", [])
            edges = plan_output.get("edges", [])

            validator = OntologyValidatorTool()
            validation_data = await validator.run(nodes=nodes, edges=edges)

            is_valid = validation_data.get("is_valid", True)
            errors = validation_data.get("errors", [])
            stats = validation_data.get("stats", {})

            if not is_valid:
                formatted_reasons = self.format_validation_errors_to_instructions(errors)
                logger.warning(
                    "canvas_ontology_validation_failed",
                    error_count=len(errors),
                    errors=formatted_reasons,
                )
                return ValidationResult(
                    is_valid=False,
                    reasons=formatted_reasons,
                    metadata={
                        "stats": stats,
                        "validation_data": validation_data,
                    },
                )

            return ValidationResult(
                is_valid=True,
                reasons=[],
                metadata={
                    "stats": stats,
                    "validation_data": validation_data,
                },
            )

        except Exception as e:
            logger.error(
                "canvas_inspect_plan_output_failed",
                error_type=type(e).__name__,
                error_msg=str(e),
                exc_info=True,
            )
            return ValidationResult(
                is_valid=False,
                reasons=[f"Failed to validate canvas ontology: {str(e)}"],
            )

    def format_validation_errors_to_instructions(self, errors: List[str]) -> List[str]:
        """
        Transforms raw OntologyValidatorTool error messages into specific,
        actionable, step-by-step English correction instructions for the LLM during refinement.
        """
        instructions = []
        for err in errors:
            lower_err = err.lower()
            if "source_ref" in lower_err:
                instruction = f"[Correction Required]: {err}. Action: Please provide a valid, non-empty 'source_ref' (at least 3 characters) for empirical grounding."
            elif "rationale" in lower_err:
                instruction = f"[Correction Required]: {err}. Action: Please provide a clear 'rationale' (at least 10 characters) explaining the synthesis logic."
            elif "orphaned edge" in lower_err:
                instruction = f"[Correction Required]: {err}. Action: Ensure all edge source and target IDs reference existing nodes in the node map."
            elif "self-referential loop" in lower_err or "self-loops" in lower_err:
                instruction = f"[Correction Required]: {err}. Action: Remove self-loops where source and target are identical."
            elif "directional error" in lower_err and "focus" in lower_err:
                instruction = f"[Correction Required]: {err}. Action: Central FOCUS nodes cannot be driven backwards by TRIGGERS, CONSTRAINS, or VALIDATES from other nodes. Reverse the edge direction or change relation type."
            elif "static" in lower_err and "cannot triggers" in lower_err:
                instruction = f"[Correction Required]: {err}. Action: Static entities or resources cannot directly trigger causality; introduce an intermediary EVENT node or change the relation."
            elif "temporal paradox" in lower_err:
                instruction = f"[Correction Required]: {err}. Action: Correct the causal direction; downstream synthesis nodes (OUTCOME/INSIGHT) cannot trigger past observation events."
            elif "terminal violation" in lower_err or ("outcome" in lower_err and "cannot act as a causal source" in lower_err):
                instruction = f"[Correction Required]: {err}. Action: OUTCOME nodes are terminal goals and cannot act as a causal source for TRIGGERS."
            elif "boundary" in lower_err and "use constrains instead of triggers" in lower_err:
                instruction = f"[Correction Required]: {err}. Action: BOUNDARY nodes are static constraints; use CONSTRAINS instead of TRIGGERS."
            elif "logical flaw" in lower_err and "query" in lower_err:
                instruction = f"[Correction Required]: {err}. Action: Unresolved QUERY nodes cannot activate or trigger other nodes until resolved."
            elif "fact categorization error" in lower_err:
                instruction = f"[Correction Required]: {err}. Action: EVENT nodes are raw observations and cannot be validated; use REF instead."
            elif "lacks empirical weight to validates" in lower_err:
                instruction = f"[Correction Required]: {err}. Action: Abstract nodes (QUERY, BOUNDARY, NAVIGATION) lack empirical weight to VALIDATE; use EVENT, RESOURCE, or INSIGHT."
            elif "validates must target" in lower_err:
                instruction = f"[Correction Required]: {err}. Action: VALIDATES edges must target a claim, concept, or unresolved question (INSIGHT, OUTCOME, CONCEPT, QUERY)."
            elif "is not an allowed source for constrains" in lower_err:
                instruction = f"[Correction Required]: {err}. Action: CONSTRAINS source must be a scope-defining component (BOUNDARY, CONCEPT, GROUP, FOCUS)."
            elif "boundary constraint cannot restrict static references" in lower_err:
                instruction = f"[Correction Required]: {err}. Action: Boundary constraints cannot restrict static references or portals like RESOURCE or NAVIGATION."
            elif "spatial container" in lower_err or "group" in lower_err:
                instruction = f"[Correction Required]: {err}. Action: GROUP nodes are spatial containers; only REF or LINK relationships are permitted."
            else:
                instruction = f"[Correction Required]: {err}. Action: Please fix this ontology violation to comply with empirical science standards."
            instructions.append(instruction)
        return instructions

    async def execute_tool_workflow(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any]
    ) -> List[Message]:
        """
        Stage 2 Implementation: Executes deterministic tool workflows for canvas decomposition.
        """
        # For decomposition, Stage 2 currently relies on structural validation or no-op workflow,
        # integration points for graph growth tools can be added here if needed.
        return []

    def parse_final_output(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any], raw_llm_output: str
    ) -> Any:
        return raw_llm_output
