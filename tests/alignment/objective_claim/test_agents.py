import pytest
from unittest.mock import AsyncMock, MagicMock

from auraflux_core.alignment.objective_claim.agents import ObjectiveClaimAgent
from auraflux_core.alignment.objective_claim.schemas import ObjectiveClaimVerdict
from auraflux_core.core.schemas.messages import Message


@pytest.fixture
def agent():
    """Fixture to instantiate ObjectiveClaimAgent with mocked dependencies."""
    mock_config = MagicMock()
    mock_client_manager = MagicMock()

    agent_instance = ObjectiveClaimAgent(
        config=mock_config,
        client_manager=mock_client_manager
    )
    # Mock the LLM generation and output parser
    agent_instance.generate = AsyncMock()
    agent_instance.output_parser = MagicMock()
    return agent_instance


@pytest.mark.asyncio
async def test_diagnose_and_verify_verified_status(agent):
    """Test Case 1: Verifies a claim with sufficient supporting evidence in context."""
    # Arrange
    proposition_id = "PROP-001"
    claim_text = "The target entity obtained ISO 27001 certification in 2025."

    mock_llm_response = Message(
        role='assistant',
        content='{"status": "VERIFIED"}',
        name="ObjectiveClaimAgent"
    )
    agent.generate.return_value = mock_llm_response

    agent.output_parser.parse_json.return_value = {
        "triples": [
            {
                "subject": "target entity",
                "predicate": "obtained",
                "object": "ISO 27001 certification"
            }
        ],
        "diagnostics": {
            "implicit_premises": ["Certification remains valid throughout 2025."],
            "quantification_requirements": {
                "required_artifact_types": ["Certificate"],
                "acceptance_criteria": "Issued in 2025"
            },
            "boundary_conflicts": {
                "has_conflict": False
            }
        },
        "status": "VERIFIED",
        "verification_proofs": ["ISO_27001_Certificate_2025.pdf#Page2"],
        "compliance_gap": None
    }

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
async def test_diagnose_and_verify_unsupported_status_with_fallback_gap(agent):
    """Test Case 2: Verifies fallback handling when context has no proofs and status is UNSUPPORTED."""
    # Arrange
    proposition_id = "PROP-002"
    claim_text = "System achieved 99.999% uptime in Q1."

    mock_llm_response = Message(
        role="assistant",
        content='{"status": "UNSUPPORTED"}',
        name="ObjectiveClaimAgent"
    )
    agent.generate.return_value = mock_llm_response

    agent.output_parser.parse_json.return_value = {
        "triples": [],
        "diagnostics": {
            "implicit_premises": [],
            "quantification_requirements": {},
            "boundary_conflicts": {}
        },
        "status": "UNSUPPORTED",
        "verification_proofs": [],
        "compliance_gap": None  # Trigger fallback handling
    }

    # Act
    result = await agent.diagnose_and_verify(proposition_id, claim_text)

    # Assert
    assert isinstance(result, ObjectiveClaimVerdict)
    assert result.status == "UNSUPPORTED"
    assert len(result.verification_proofs) == 0
    # Fallback explanation should be attached automatically
    assert result.compliance_gap == "Core Context contains no verifiable artifact or record supporting this claim."
