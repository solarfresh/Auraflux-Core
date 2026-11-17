import json
from typing import Any, Dict, List, Optional

from auraflux_core.canvases.decomposition.schemas import (
    DecompositionResponse,
)
from auraflux_core.canvases.schemas import ConceptualEdge, ConceptualNode
from auraflux_core.canvases.pipelines import CanvasPlanAndExecuteHandler
from auraflux_core.core.agents.base import BaseAgent
from auraflux_core.core.configs.logging_config import get_logger
from auraflux_core.core.schemas.messages import Message
from auraflux_core.core.schemas.pipelines import ValidationResult

logger = get_logger(__name__)


class DecompositionAgent(BaseAgent, CanvasPlanAndExecuteHandler):
    """
    DecompositionAgent breaks down a complex conceptual node into sub-nodes and causal links
    using the PlanAndExecutePipeline, integrated with CanvasPlanAndExecuteHandler for ontology validation.
    """

    def get_plan_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts context variables needed for Stage 1 planning template."""
        source_node = payload.get("source_node", {})
        if hasattr(source_node, "model_dump"):
            source_node_dict = source_node.model_dump()
        else:
            source_node_dict = source_node

        return {
            "source_node": json.dumps(source_node_dict, ensure_ascii=False, indent=2),
        }

    def get_plan_refinement_context(
        self,
        payload: Dict[str, Any],
        plan_output: Dict[str, Any],
        validation_result: ValidationResult,
    ) -> Optional[Dict[str, Any]]:
        """
        Extracts context variables for plan refinement stage based on validation errors.
        """
        reasons = getattr(validation_result, "reasons", [])
        if not reasons:
            return None

        source_node = payload.get("source_node", {})
        if hasattr(source_node, "model_dump"):
            source_node_dict = source_node.model_dump()
        else:
            source_node_dict = source_node

        return {
            "source_node": json.dumps(source_node_dict, ensure_ascii=False, indent=2),
            "plan_output": json.dumps(plan_output, ensure_ascii=False, indent=2),
            "validation_warnings": json.dumps(reasons, ensure_ascii=False, indent=2),
        }

    def get_synthesis_context(
        self,
        payload: Dict[str, Any],
        plan_output: Dict[str, Any],
        tool_results: List[Message]
    ) -> Dict[str, Any]:
        source_node = payload.get("source_node", {})
        if hasattr(source_node, "model_dump"):
            source_node_dict = source_node.model_dump()
        else:
            source_node_dict = source_node

        return {
            "source_node": json.dumps(source_node_dict, ensure_ascii=False, indent=2),
            "plan_output": json.dumps(plan_output, ensure_ascii=False, indent=2),
        }

    def parse_final_output(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any], raw_llm_output: str
    ) -> DecompositionResponse:
        """Domain Transformation Hook: Maps plan and synthesis/raw outputs into DecompositionResponse."""
        try:
            parsed_data = self.output_parser.parse_json(raw_llm_output)
        except Exception:
            # Fallback to plan_output if raw_llm_output parsing fails
            parsed_data = plan_output

        sub_nodes = [ConceptualNode(**item) for item in parsed_data.get("nodes", [])]
        causal_links = [ConceptualEdge(**item) for item in parsed_data.get("edges", [])]

        result = DecompositionResponse(
            nodes=sub_nodes,
            edges=causal_links,
        )

        source_node = payload.get("source_node", {})
        source_id = source_node.get("id", "unknown") if isinstance(source_node, dict) else getattr(source_node, "id", "unknown")

        logger.info(
            "decomposition_completed",
            source_node_id=source_id,
            sub_nodes_count=len(sub_nodes),
            causal_links_count=len(causal_links),
        )

        return result
