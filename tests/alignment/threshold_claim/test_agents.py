import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from auraflux_core.alignment.threshold_claim.agents import ThresholdClaimAgent
from auraflux_core.alignment.threshold_claim.schemas import \
    ThresholdClaimVerdict
from auraflux_core.core.schemas.messages import Message


@pytest.fixture
def agent():
    """Fixture to instantiate ThresholdClaimAgent with mocked dependencies."""
    mock_config = MagicMock()
    mock_config.name = "ThresholdClaimAgent"
    mock_config.pipeline_name = "plan_and_execute"
    mock_config.language = "zh"
    mock_config.llm.provider = "openai"
    mock_config.llm.model = "gpt-4o"
    mock_config.llm.temperature = 0.0
    mock_config.llm.max_tokens = 2000

    mock_client_manager = MagicMock()

    agent_instance = ThresholdClaimAgent(
        config=mock_config, client_manager=mock_client_manager
    )

    # Mock low-level LLM message generation capability
    agent_instance.generate = AsyncMock()

    # Mock tool executor capability
    agent_instance.tool_executor = MagicMock()
    agent_instance.tool_executor.tool_registry = {"hybrid_retriever": MagicMock()}
    agent_instance.tool_executor.run = AsyncMock()

    return agent_instance


@pytest.mark.asyncio
async def test_diagnose_and_verify_threshold_verified_with_normalized_metrics(agent):
    """Test Case 1: Verifies a quantitative threshold claim (budget <= 5M TWD) fully satisfied by context."""
    # Arrange
    proposition_id = "PROP-THRESH-001"
    claim_text = "專案總預算控制在 500 萬元新台幣以內。"

    # Stage 1 LLM Response (Plan with metric extraction)
    stage1_response = Message(
        role="assistant",
        content=json.dumps({
            "queries": [{
                "query_text": "專案 預算 經費",
                "target_type": "evidence",
                "text_field": "evidence_text",
                "vector_field": "evidence_vector"
            }],
            "triples": [{
                "subject": "專案總預算",
                "predicate": "<=",
                "object": "500萬元",
                "metric_name": "budget",
                "normalized_value": 5000000.0,
                "unit": "TWD",
                "operator": "<="
            }]
        }),
        name="ThresholdClaimAgent",
    )

    # Stage 3 LLM Response (Synthesis Verdict)
    stage3_response = Message(
        role="assistant",
        content="""{
            "diagnostic_evaluation": {
                "implicit_premises_eval": "Budget allocation explicitly defined in contract.",
                "quantification_eval": "Actual budget is 4.5M TWD, satisfying the <= 5.0M TWD constraint.",
                "boundary_conflicts_eval": "No boundary conflict detected."
            },
            "status": "VERIFIED",
            "verification_proofs": ["Project_Contract.pdf#Page12"],
            "compliance_gap": null
        }""",
        name="ThresholdClaimAgent",
    )

    agent.generate.side_effect = [stage1_response, stage3_response]

    # Tool Execution Result (Returned evidence with normalized metrics)
    tool_payload = [
        {
            "content": {
                "evidence_text": "本專案核定總預算為新台幣 450 萬元整。",
                "location": "Project_Contract.pdf#Page12",
                "triples": [
                    {
                        "subject": "專案核定總預算",
                        "predicate": "等於",
                        "object": "450萬元",
                        "metric_name": "budget",
                        "normalized_value": 4500000.0,
                        "unit": "TWD",
                        "operator": "=="
                    }
                ]
            }
        }
    ]
    agent.tool_executor.run.return_value = Message(
        role="tool",
        content=json.dumps(tool_payload),
        name="hybrid_retriever",
    )

    def mock_parse_json(content):
        parsed = json.loads(content)
        if "queries" in parsed or "triples" in parsed:
            return {
                "triples": parsed.get("triples", []),
                "diagnostics": {
                    "implicit_premises": ["Budget covers all sub-contractors."],
                    "quantification_requirements": [
                        {
                            "raw_text": "500萬元",
                            "metric_name": "budget",
                            "normalized_value": 5000000.0,
                            "unit": "TWD",
                            "operator": "<="
                        }
                    ],
                    "boundary_conflicts": {"has_conflict": False},
                },
                "queries": parsed.get("queries", []),
            }
        return parsed

    agent.output_parser.parse_json = MagicMock(side_effect=mock_parse_json)

    # Act
    result = await agent.diagnose_and_verify(proposition_id, claim_text)

    # Assert
    assert isinstance(result, ThresholdClaimVerdict)
    assert result.proposition_id == proposition_id
    assert result.status == "VERIFIED"
    assert len(result.triples) == 1
    assert result.triples[0].metric_name == "budget"
    assert result.triples[0].normalized_value == 5000000.0
    assert result.triples[0].unit == "TWD"
    assert result.diagnostics.boundary_conflicts["has_conflict"] is False
    assert "Project_Contract.pdf#Page12" in result.verification_proofs
    assert result.compliance_gap is None


@pytest.mark.asyncio
async def test_diagnose_and_verify_threshold_violation_boundary_conflict(agent):
    """Test Case 2: Tests PARTIALLY_VERIFIED/VIOLATED status when SLA threshold (latency <= 200ms) fails boundary check."""
    # Arrange
    proposition_id = "PROP-THRESH-002"
    claim_text = "API 響應時間（SLA）需在 200ms 以內。"

    stage1_response = Message(
        role="assistant",
        content=json.dumps({
            "queries": [{
                "query_text": "API response time SLA 響應時間",
                "target_type": "evidence",
                "text_field": "evidence_text",
                "vector_field": "evidence_vector"
            }],
            "triples": [{
                "subject": "API 響應時間",
                "predicate": "<=",
                "object": "200ms",
                "metric_name": "sla_latency",
                "normalized_value": 0.2,
                "unit": "second",
                "operator": "<="
            }]
        }),
        name="ThresholdClaimAgent",
    )

    stage3_response = Message(
        role="assistant",
        content="""{
            "diagnostic_evaluation": {
                "implicit_premises_eval": "SLA applies to p99 percentile.",
                "quantification_eval": "Actual p99 response time is 0.45 seconds (450ms).",
                "boundary_conflicts_eval": "Boundary Conflict: Actual latency 0.45s exceeds maximum allowed threshold of 0.2s."
            },
            "status": "PARTIALLY_VERIFIED",
            "verification_proofs": ["SLA_Report.json#Line18"],
            "compliance_gap": "SLA requirement failed: Average response time is acceptable, but p99 latency (450ms) exceeds 200ms constraint."
        }""",
        name="ThresholdClaimAgent",
    )

    agent.generate.side_effect = [stage1_response, stage3_response]

    agent.tool_executor.run.return_value = Message(
        role="tool",
        content=json.dumps([
            {
                "content": {
                    "evidence_text": "System benchmark report: p99 API response time measured at 450ms.",
                    "location": "SLA_Report.json#Line18",
                    "triples": [{
                        "subject": "p99 API response time",
                        "predicate": "==","object": "450ms",
                        "metric_name": "sla_latency",
                        "normalized_value": 0.45,
                        "unit": "second",
                        "operator": "=="
                    }]
                }
            }
        ]),
        name="ThresholdClaimAgent",
    )

    def mock_parse_json(content):
        parsed = json.loads(content)
        if "queries" in parsed or "triples" in parsed:
            return {
                "triples": parsed.get("triples", []),
                "diagnostics": {
                    "implicit_premises": ["Peak load condition"],
                    "quantification_requirements": [
                        {
                            "raw_text": "200ms",
                            "metric_name": "sla_latency",
                            "normalized_value": 0.2,
                            "unit": "second",
                            "operator": "<="
                        }
                    ],
                    "boundary_conflicts": {
                        "has_conflict": True,
                        "conflict_reason": "450ms > 200ms limit"
                    },
                },
                "queries": parsed.get("queries", []),
            }
        return parsed

    agent.output_parser.parse_json = MagicMock(side_effect=mock_parse_json)

    # Act
    result = await agent.diagnose_and_verify(proposition_id, claim_text)

    # Assert
    assert isinstance(result, ThresholdClaimVerdict)
    assert result.status == "PARTIALLY_VERIFIED"
    assert result.triples[0].unit == "second"
    assert result.diagnostics.boundary_conflicts["has_conflict"] is True
    assert "SLA_Report.json#Line18" in result.verification_proofs
    assert result.compliance_gap is not None
    assert "exceeds 200ms constraint" in result.compliance_gap


@pytest.mark.asyncio
async def test_diagnose_and_verify_threshold_unsupported_fallback(agent):
    """Test Case 3: Verifies fallback behavior when no threshold metrics or evidence can be retrieved."""
    # Arrange
    proposition_id = "PROP-THRESH-003"
    claim_text = "系統維護中斷時間每年不超過 2 小時。"

    stage1_response = Message(
        role="assistant",
        content=json.dumps({
            "queries": [{
                "query_text": "系統維護 中斷時間 每年 2小時",
                "target_type": "evidence",
                "text_field": "evidence_text",
                "vector_field": "evidence_vector"
            }],
            "triples": [{
                "subject": "系統維護中斷時間",
                "predicate": "<=",
                "object": "2小時",
                "metric_name": "downtime",
                "normalized_value": 7200.0,
                "unit": "second",
                "operator": "<="
            }]
        }),
        name="ThresholdClaimAgent",
    )

    stage3_response = Message(
        role="assistant",
        content="""{
            "diagnostic_evaluation": {
                "implicit_premises_eval": "No downtime log found.",
                "quantification_eval": "Cannot verify annual maintenance limit.",
                "boundary_conflicts_eval": "Unable to evaluate threshold boundary conditions due to lack of metrics."
            },
            "status": "UNSUPPORTED",
            "verification_proofs": [],
            "compliance_gap": null
        }""",
        name="ThresholdClaimAgent",
    )

    agent.generate.side_effect = [stage1_response, stage3_response]

    # Empty retriever response
    agent.tool_executor.run.return_value = Message(
        role="tool", content="", name="hybrid_retriever"
    )

    def mock_parse_json(content):
        parsed = json.loads(content)
        if "queries" in parsed or "triples" in parsed:
            return {
                "triples": parsed.get("triples", []),
                "diagnostics": {
                    "implicit_premises": [],
                    "quantification_requirements": {},
                    "boundary_conflicts": {},
                },
                "queries": parsed.get("queries", []),
            }
        return parsed

    agent.output_parser.parse_json = MagicMock(side_effect=mock_parse_json)

    # Act
    result = await agent.diagnose_and_verify(proposition_id, claim_text)

    # Assert
    assert isinstance(result, ThresholdClaimVerdict)
    assert result.status == "UNSUPPORTED"
    assert len(result.verification_proofs) == 0
    assert result.compliance_gap is not None
    assert "unsupported" in result.compliance_gap.lower()
