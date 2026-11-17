from unittest.mock import MagicMock

import pytest

from auraflux_core.canvases.decomposition.agents import DecompositionAgent
from auraflux_core.canvases.decomposition.schemas import (
    DecompositionAgentConfig)
from auraflux_core.canvases.schemas import ConceptualNode, ConceptualNodeType, ConceptualGraph
from auraflux_core.core.schemas.pipelines import ValidationResult


class DummyDecompositionAgent(DecompositionAgent):
    async def execute_tool_workflow(self, payload, plan_output):
        return []


@pytest.fixture
def agent(mock_client_manager):
    config = DecompositionAgentConfig(provider="openai", model="gpt-4")
    return DummyDecompositionAgent(config=config, client_manager=mock_client_manager)


def test_get_plan_context_dict(agent):
    payload = {"source_node": {"id": "n1", "label": "Test Node"}}
    context = agent.get_plan_context(payload)
    assert "source_node" in context
    assert "Test Node" in context["source_node"]


def test_get_plan_context_model(agent):
    node = ConceptualNode(
        id="n1",
        type=ConceptualNodeType.CONCEPT,
        content="Model Node",
        anchor_id="a1",
        label="Model Node",
        source_ref="ref",
        rationale="rat"
    )
    payload = {"source_node": node}
    context = agent.get_plan_context(payload)
    assert "source_node" in context
    assert "Model Node" in context["source_node"]


def test_get_plan_refinement_context_valid(agent):
    payload = {"source_node": {"id": "n1", "label": "Test Node"}}
    plan_output = {"nodes": [], "edges": []}
    validation_result = ValidationResult(is_valid=False, reasons=["Missing source_ref"])

    context = agent.get_plan_refinement_context(payload, plan_output, validation_result)
    assert context is not None
    assert "validation_warnings" in context
    assert "Missing source_ref" in context["validation_warnings"]


def test_get_plan_refinement_context_empty(agent):
    payload = {"source_node": {"id": "n1", "label": "Test Node"}}
    plan_output = {"nodes": [], "edges": []}
    validation_result = ValidationResult(is_valid=True, reasons=[])

    context = agent.get_plan_refinement_context(payload, plan_output, validation_result)
    assert context is None


def test_parse_final_output_success(agent):
    payload = {"source_node": {"id": "root"}}
    plan_output = {}
    raw_llm_output = '{"nodes": [{"id": "sub1", "type": "CONCEPT", "content": "Sub 1", "anchor_id": "a1", "label": "Sub 1", "source_ref": "ref", "rationale": "rat"}], "edges": []}'

    agent.output_parser = MagicMock()
    agent.output_parser.parse_json.return_value = {
        "nodes": [{"id": "sub1", "type": "CONCEPT", "content": "Sub 1", "anchor_id": "a1", "label": "Sub 1", "source_ref": "ref", "rationale": "rat"}],
        "edges": []
    }

    result = agent.parse_final_output(payload, plan_output, raw_llm_output)
    assert isinstance(result, ConceptualGraph)
    assert len(result.nodes) == 1
    assert "sub1" in result.nodes


def test_parse_final_output_fallback(agent):
    payload = {"source_node": {"id": "root"}}
    plan_output = {
        "nodes": [{"id": "sub_fallback", "type": "CONCEPT", "content": "Fallback", "anchor_id": "a1", "label": "Fallback", "source_ref": "ref", "rationale": "rat"}],
        "edges": []
    }
    raw_llm_output = 'invalid json'

    agent.output_parser = MagicMock()
    agent.output_parser.parse_json.side_effect = Exception("Parsing error")

    result = agent.parse_final_output(payload, plan_output, raw_llm_output)
    assert isinstance(result, ConceptualGraph)
    assert len(result.nodes) == 1
    assert "sub_fallback" in result.nodes
