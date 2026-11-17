import pytest
from unittest.mock import AsyncMock, patch
from auraflux_core.canvases.pipelines import CanvasPlanAndExecuteHandler
from auraflux_core.core.schemas.pipelines import ValidationResult


class DummyCanvasHandler(CanvasPlanAndExecuteHandler):
    async def execute_tool_workflow(self, payload, plan_output):
        return []

    def parse_final_output(self, payload, plan_output, raw_llm_output):
        return raw_llm_output


@pytest.fixture
def handler():
    return DummyCanvasHandler()


@pytest.mark.asyncio
async def test_inspect_plan_output_valid(handler):
    plan_output = {
        "nodes": [{"id": "node1", "label": "Focus Node", "source_ref": "ref1", "rationale": "rationale1"}],
        "edges": []
    }

    with patch("auraflux_core.canvases.pipelines.OntologyValidatorTool") as MockTool:
        mock_instance = MockTool.return_value
        mock_instance.run = AsyncMock(return_value={
            "is_valid": True,
            "errors": [],
            "stats": {"node_count": 1, "edge_count": 0}
        })

        result = await handler.inspect_plan_output({}, plan_output)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.reasons == []
        assert result.metadata["stats"]["node_count"] == 1


@pytest.mark.asyncio
async def test_inspect_plan_output_invalid(handler):
    plan_output = {
        "nodes": [{"id": "node1"}],
        "edges": []
    }

    with patch("auraflux_core.canvases.pipelines.OntologyValidatorTool") as MockTool:
        mock_instance = MockTool.return_value
        mock_instance.run = AsyncMock(return_value={
            "is_valid": False,
            "errors": ["Missing source_ref in node node1"],
            "stats": {"node_count": 1, "edge_count": 0}
        })

        result = await handler.inspect_plan_output({}, plan_output)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert len(result.reasons) == 1
        assert "source_ref" in result.reasons[0].lower()


def test_format_validation_errors_to_instructions(handler):
    errors = [
        "Node[1]: EVENT is missing mandatory 'source_ref' for grounding.",
        "Node[2]: INSIGHT is missing mandatory 'rationale' for synthesis.",
        "Edge[1]: Orphaned edge (n1->n2) - one or both nodes are undefined in node_map.",
        "Edge[2]: Self-referential loop detected on Node[n1].",
        "Edge[3]: Directional Error - Central FOCUS cannot be driven by TRIGGERS from Node[n1].",
        "Edge[4]: Static ENTITY[n1] cannot TRIGGERS causality.",
        "Edge[5]: Temporal Paradox - Synthesis INSIGHT[n1] cannot TRIGGERS a past observation EVENT[n2].",
        "Edge[6]: Terminal Violation - OUTCOME[n1] is a research goal and cannot act as a causal source.",
        "Edge[7]: BOUNDARY[n1] is a static constraint. Use CONSTRAINS instead of TRIGGERS.",
        "Edge[8]: Logical Flaw - Unresolved QUERY[n1] cannot activate or TRIGGERS other nodes.",
        "Edge[9]: Fact Categorization Error - EVENT[n1] is a raw observation and cannot be validated.",
        "Edge[10]: QUERY[n1] lacks empirical weight to VALIDATES.",
        "Edge[11]: Methodological Flaw - QUERY[n1] cannot directly VALIDATES OUTCOME[n2].",
        "Edge[12]: VALIDATES must target a claim, concept, or unresolved question. Invalid target: ENTITY.",
        "Edge[13]: ENTITY[n1] is not an allowed source for CONSTRAINS.",
        "Edge[14]: Boundary Constraint cannot restrict static references or portals like RESOURCE[n1].",
        "Edge[15]: GROUP[n1] is a spatial container; only REF or LINK is permitted."
    ]

    instructions = handler.format_validation_errors_to_instructions(errors)

    assert len(instructions) == 17
    assert any("source_ref" in inst for inst in instructions)
    assert any("rationale" in inst for inst in instructions)
    assert any("reference existing nodes" in inst for inst in instructions)
    assert any("self-loops" in inst for inst in instructions)
    assert any("Central FOCUS nodes cannot be driven backwards" in inst for inst in instructions)
    assert any("Static entities or resources" in inst for inst in instructions)
    assert any("downstream synthesis nodes" in inst for inst in instructions)
    assert any("OUTCOME nodes are terminal goals" in inst for inst in instructions)
    assert any("BOUNDARY nodes are static constraints" in inst for inst in instructions)
    assert any("Unresolved QUERY nodes" in inst for inst in instructions)
    assert any("EVENT nodes are raw observations" in inst for inst in instructions)
    assert any("lack empirical weight to VALIDATE" in inst for inst in instructions)
    assert any("QUERY nodes cannot directly VALIDATES" in inst for inst in instructions)
    assert any("VALIDATES edges must target" in inst for inst in instructions)
    assert any("CONSTRAINS source must be" in inst for inst in instructions)
    assert any("Boundary constraints cannot restrict static references" in inst for inst in instructions)
    assert any("GROUP nodes are spatial containers" in inst for inst in instructions)
