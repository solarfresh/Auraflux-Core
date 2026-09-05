from unittest.mock import AsyncMock, MagicMock

import pytest

from auraflux_core.alignment.objective_claim.agents import ObjectiveClaimAgent
from auraflux_core.alignment.objective_claim.orchestrators import \
    AlignmentOrchestrator
from auraflux_core.alignment.objective_claim.schemas import (
    DiagnosticAnalysis, ObjectiveClaimVerdict, TripleItem)
from auraflux_core.core.orchestrators.state import OrchestratorStatus


@pytest.fixture
def mock_agent():
    """Fixture providing a mocked ObjectiveClaimAgent."""
    agent = MagicMock(spec=ObjectiveClaimAgent)
    agent.diagnose_and_verify = AsyncMock()
    return agent


@pytest.fixture
def orchestrator(mock_agent):
    """Fixture initializing AlignmentOrchestrator with mapped agent."""
    agents_map = {"ObjectiveClaimAgent": mock_agent}
    return AlignmentOrchestrator(agents=agents_map)


@pytest.mark.asyncio
async def test_orchestrator_execute_all_verified(orchestrator, mock_agent):
    """Test Case 1: Verifies parallel execution when all claims pass verification."""
    # Arrange
    claims_input = [
        {"type": "objective_claim", "id": "PROP-001", "text": "Claim 1 statement"},
        {"type": "objective_claim", "id": "PROP-002", "text": "Claim 2 statement"},
    ]

    verdict_1 = ObjectiveClaimVerdict(
        proposition_id="PROP-001",
        claim_text="Claim 1 statement",
        triples=[TripleItem(subject="S1", predicate="P1", object="O1")],
        diagnostics=DiagnosticAnalysis(),
        status="VERIFIED",
        verification_proofs=["proof1.pdf"],
    )
    verdict_2 = ObjectiveClaimVerdict(
        proposition_id="PROP-002",
        claim_text="Claim 2 statement",
        triples=[TripleItem(subject="S2", predicate="P2", object="O2")],
        diagnostics=DiagnosticAnalysis(),
        status="VERIFIED",
        verification_proofs=["proof2.pdf"],
    )

    mock_agent.diagnose_and_verify.side_effect = [verdict_1, verdict_2]

    # Act
    final_state = await orchestrator.run(input_data=claims_input)

    # Assert
    assert final_state.status == OrchestratorStatus.COMPLETED
    assert final_state.metadata.get("is_locked") is True
    assert len(final_state.metadata.get("verdicts", [])) == 2
    assert mock_agent.diagnose_and_verify.call_count == 2


@pytest.mark.asyncio
async def test_orchestrator_execute_with_unsupported_triggers_block(orchestrator, mock_agent):
    """Test Case 2: Verifies that an UNSUPPORTED claim sets is_locked to False (BLOCK state)."""
    # Arrange
    claims_input = [
        {"type": "objective_claim", "id": "PROP-001", "text": "Supported claim"},
        {"type": "objective_claim", "id": "PROP-002", "text": "Unsupported claim"},
    ]

    verdict_1 = ObjectiveClaimVerdict(
        proposition_id="PROP-001",
        claim_text="Supported claim",
        diagnostics=DiagnosticAnalysis(),
        status="VERIFIED",
    )
    verdict_2 = ObjectiveClaimVerdict(
        proposition_id="PROP-002",
        claim_text="Unsupported claim",
        diagnostics=DiagnosticAnalysis(),
        status="UNSUPPORTED",
        compliance_gap="No proof found",
    )

    mock_agent.diagnose_and_verify.side_effect = [verdict_1, verdict_2]

    # Act
    final_state = await orchestrator.run(input_data=claims_input)

    # Assert
    assert final_state.status == OrchestratorStatus.COMPLETED
    assert final_state.metadata.get("is_locked") is False
    assert len(final_state.metadata.get("verdicts", [])) == 2


@pytest.mark.asyncio
async def test_orchestrator_handles_empty_or_unmatched_claims(orchestrator):
    """Test Case 3: Handles input with no matching objective_claim types cleanly."""
    # Arrange
    claims_input = [
        {"type": "mental_model", "id": "MM-001", "text": "Unmatched type"}
    ]

    # Act
    final_state = await orchestrator.run(input_data=claims_input)

    # Assert
    assert final_state.status == OrchestratorStatus.COMPLETED
    assert final_state.metadata.get("is_locked") is True
    assert final_state.metadata.get("verdicts") == []
