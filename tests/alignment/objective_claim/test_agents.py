import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from auraflux_core.alignment.objective_claim.agents import ObjectiveClaimAgent
from auraflux_core.alignment.objective_claim.schemas import \
    ObjectiveClaimVerdict
from auraflux_core.core.schemas.messages import Message


@pytest.fixture
def agent():
    """Fixture to instantiate ObjectiveClaimAgent with mocked dependencies."""
    mock_config = MagicMock()
    mock_config.name = "ObjectiveClaimAgent"
    mock_config.pipeline_name = "plan_and_execute"
    mock_config.language = "zh"
    mock_config.llm.provider = "openai"
    mock_config.llm.model = "gpt-4o"
    mock_config.llm.temperature = 0.0
    mock_config.llm.max_tokens = 2000

    mock_client_manager = MagicMock()

    agent_instance = ObjectiveClaimAgent(
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
async def test_diagnose_and_verify_verified_status(agent):
    """Test Case 1: Verifies a claim fully supported by evidence with structured Stage 3 diagnostic evaluation."""
    # Arrange
    proposition_id = "PROP-001"
    claim_text = "The target entity obtained ISO 27001 certification in 2025."

    # Stage 1 LLM Response (Plan / Diagnostics)
    stage1_response = Message(
        role="assistant",
        content='{"query_text": "ISO 27001 certificate 2025"}',
        name="ObjectiveClaimAgent",
    )

    # Stage 3 LLM Response (Synthesis Verdict including diagnostic_evaluation)
    stage3_response = Message(
        role="assistant",
        content="""{
            "diagnostic_evaluation": {
                "implicit_premises_eval": "Certificate validity confirmed in 2025 document.",
                "quantification_eval": "Artifact requirement met with certificate copy.",
                "boundary_conflicts_eval": "No resource or rule conflicts detected."
            },
            "status": "VERIFIED",
            "verification_proofs": ["ISO_27001_Certificate_2025.pdf#Page2"],
            "compliance_gap": null
        }""",
        name="ObjectiveClaimAgent",
    )

    # Pipeline executes LLM twice: Stage 1 -> Stage 3
    agent.generate.side_effect = [stage1_response, stage3_response]

    # Mock tool execution response (JSON structure simulating layer 1-4 context)
    tool_payload = [
        {
            "content": {
                "target_question": "Does entity hold ISO 27001 in 2025?",
                "concept_title": "ISO 27001",
                "concept_description": "Information security management standard.",
                "triples": [
                    {
                        "subject": "target entity",
                        "predicate": "holds",
                        "object": "ISO 27001",
                    }
                ],
                "evidence_text": "Document confirms ISO 27001 was issued in 2025.",
                "location": "ISO_27001_Certificate_2025.pdf#Page2",
                "tags": ["certification", "security"],
            }
        }
    ]
    agent.tool_executor.run.return_value = Message(
        role="tool",
        content=json.dumps(tool_payload),
        name="hybrid_retriever",
    )

    # Mock OutputParser for Stage 1 and Stage 3 JSON parsing
    def mock_parse_json(content):
        parsed = json.loads(content)
        if "query_text" in parsed:
            return {
                "triples": [
                    {
                        "subject": "target entity",
                        "predicate": "obtained",
                        "object": "ISO 27001 certification",
                    }
                ],
                "diagnostics": {
                    "implicit_premises": [
                        "Certification remains valid throughout 2025."
                    ],
                    "quantification_requirements": {
                        "required_artifact_types": ["Certificate"],
                        "acceptance_criteria": "Issued in 2025",
                    },
                    "boundary_conflicts": {"has_conflict": False},
                },
                "query_text": parsed["query_text"],
            }
        return parsed

    agent.output_parser.parse_json = MagicMock(side_effect=mock_parse_json)

    # Act
    result = await agent.diagnose_and_verify(proposition_id, claim_text)

    # Assert
    assert isinstance(result, ObjectiveClaimVerdict)
    assert result.proposition_id == proposition_id
    assert result.status == "VERIFIED"
    assert len(result.triples) == 1
    assert result.triples[0].subject == "target entity"
    assert result.diagnostics.boundary_conflicts["has_conflict"] is False
    assert "ISO_27001_Certificate_2025.pdf#Page2" in result.verification_proofs
    assert result.compliance_gap is None


@pytest.mark.asyncio
async def test_diagnose_and_verify_partially_verified_with_boundary_conflict(agent):
    """Test Case 2: Tests PARTIALLY_VERIFIED status when boundary conflicts remain unresolved."""
    # Arrange
    proposition_id = "PROP-002"
    claim_text = (
        "Deployment completed in 1 day with zero downtime under restricted network."
    )

    stage1_response = Message(
        role="assistant",
        content='{"query_text": "deployment restricted network downtime"}',
        name="ObjectiveClaimAgent",
    )

    stage3_response = Message(
        role="assistant",
        content="""{
            "diagnostic_evaluation": {
                "implicit_premises_eval": "Network access was assumed to be granted.",
                "quantification_eval": "Downtime metric partially confirmed.",
                "boundary_conflicts_eval": "Restricted network policy prevents 1-day deployment constraint."
            },
            "status": "PARTIALLY_VERIFIED",
            "verification_proofs": ["Deployment_Log.txt#Line45"],
            "compliance_gap": "Deployment completed but required 3 days due to restricted network security review."
        }""",
        name="ObjectiveClaimAgent",
    )

    agent.generate.side_effect = [stage1_response, stage3_response]

    agent.tool_executor.run.return_value = Message(
        role="tool",
        content=json.dumps(
            [
                {
                    "content": {
                        "evidence_text": "Deployment took 3 days under restricted policy.",
                        "location": "Deployment_Log.txt#Line45",
                    }
                }
            ]
        ),
        name="hybrid_retriever",
    )

    def mock_parse_json(content):
        parsed = json.loads(content)
        if "query_text" in parsed:
            return {
                "triples": [],
                "diagnostics": {
                    "implicit_premises": ["Full network access"],
                    "quantification_requirements": {"time_limit": "1 day"},
                    "boundary_conflicts": {
                        "has_conflict": True,
                        "conflict_reason": "Security policy delays",
                    },
                },
                "query_text": parsed["query_text"],
            }
        return parsed

    agent.output_parser.parse_json = MagicMock(side_effect=mock_parse_json)

    # Act
    result = await agent.diagnose_and_verify(proposition_id, claim_text)

    # Assert
    assert isinstance(result, ObjectiveClaimVerdict)
    assert result.status == "PARTIALLY_VERIFIED"
    assert result.diagnostics.boundary_conflicts["has_conflict"] is True
    assert "Deployment_Log.txt#Line45" in result.verification_proofs
    assert result.compliance_gap is not None
    assert "restricted network security review" in result.compliance_gap


@pytest.mark.asyncio
async def test_diagnose_and_verify_unsupported_status_with_fallback_gap(agent):
    """Test Case 3: Verifies fallback handling when context has no proofs and status is UNSUPPORTED."""
    # Arrange
    proposition_id = "PROP-003"
    claim_text = "System achieved 99.999% uptime in Q1."

    stage1_response = Message(
        role="assistant",
        content='{"query_text": "system uptime Q1 99.999%"}',
        name="ObjectiveClaimAgent",
    )

    stage3_response = Message(
        role="assistant",
        content="""{
            "diagnostic_evaluation": {
                "implicit_premises_eval": "No records found.",
                "quantification_eval": "99.999% metric cannot be verified.",
                "boundary_conflicts_eval": "Unable to evaluate boundary conditions."
            },
            "status": "UNSUPPORTED",
            "verification_proofs": [],
            "compliance_gap": null
        }""",
        name="ObjectiveClaimAgent",
    )

    agent.generate.side_effect = [stage1_response, stage3_response]

    agent.tool_executor.run.return_value = Message(
        role="tool", content="", name="hybrid_retriever"
    )

    def mock_parse_json(content):
        parsed = json.loads(content)
        if "query_text" in parsed:
            return {
                "triples": [],
                "diagnostics": {
                    "implicit_premises": [],
                    "quantification_requirements": {},
                    "boundary_conflicts": {},
                },
                "query_text": parsed["query_text"],
            }
        return parsed

    agent.output_parser.parse_json = MagicMock(side_effect=mock_parse_json)

    # Act
    result = await agent.diagnose_and_verify(proposition_id, claim_text)

    # Assert
    assert isinstance(result, ObjectiveClaimVerdict)
    assert result.status == "UNSUPPORTED"
    assert len(result.verification_proofs) == 0
    # Fallback explanation should be attached automatically when compliance_gap is returned as null
    assert result.compliance_gap is not None
    assert "Core Context contains no supporting evidence" in result.compliance_gap
    assert "Unable to evaluate boundary conditions" in result.compliance_gap
