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
        "Missing source_ref",
        "Missing rationale",
        "Missing focus node",
        "Self-referential loop detected",
        "Orphaned edge found",
        "Temporal paradox in edges",
        "Static boundary violation",
        "Unknown custom error"
    ]

    instructions = handler.format_validation_errors_to_instructions(errors)

    assert len(instructions) == 8
    assert any("source_ref" in inst for inst in instructions)
    assert any("rationale" in inst for inst in instructions)
    assert any("self-loops" in inst for inst in instructions)
    assert any("reference existing nodes" in inst for inst in instructions)
    assert any("causal direction" in inst for inst in instructions)
    assert any("edge types" in inst for inst in instructions)
    assert any("empirical science standards" in inst for inst in instructions)
