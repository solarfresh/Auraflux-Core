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
            elif "self-referential" in lower_err or "self-loops" in lower_err:
                instruction = f"[Correction Required]: {err}. Action: Remove self-loops where source and target are identical."
            elif "orphaned" in lower_err:
                instruction = f"[Correction Required]: {err}. Action: Ensure all edge source and target IDs reference existing nodes in the node map."
            elif "temporal paradox" in lower_err:
                instruction = f"[Correction Required]: {err}. Action: Correct causal direction to prevent future outcomes/insights from triggering past events."
            elif "static" in lower_err or "boundary" in lower_err:
                instruction = f"[Correction Required]: {err}. Action: Adjust edge types (e.g. use CONSTRAINS or LINK instead of TRIGGERS for static boundaries/entities)."
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
