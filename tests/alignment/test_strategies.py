import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from auraflux_core.alignment.agents import BaseAlignmentAgent
from auraflux_core.alignment.objective_claim.schemas import (
    DiagnosticAnalysis, ObjectiveClaimVerdict)
from auraflux_core.alignment.strategies import AlignmentOrchestrationStrategy
from auraflux_core.core.orchestrators.state import OrchestratorState


@pytest.fixture
def mock_agent():
    """Mock agent implementing BaseAlignmentAgent and required capabilities."""
    agent = MagicMock(spec=BaseAlignmentAgent)
    agent.register_tools = MagicMock()
    agent.diagnose_and_verify = AsyncMock(
        return_value=ObjectiveClaimVerdict(
            status="VERIFIED",
            proposition_id="c1",
            claim_text="Medical claim",
            diagnostics=DiagnosticAnalysis.model_construct(),
        )
    )
    return agent


@pytest.fixture
def initial_state():
    """Provides an initial clean OrchestratorState."""
    return OrchestratorState()


# =============================================================================
# AlignmentOrchestrationStrategy Tests
# =============================================================================


@pytest.mark.asyncio
async def test_alignment_orchestration_tool_registration(
    mock_agent, initial_state, dummy_tool
):
    """Verify strategy registers tools across mapped alignment agents prior to task execution."""
    tools = {"dummy_tool": dummy_tool}
    agents = {"ObjectiveClaimAgent": mock_agent}
    strategy = AlignmentOrchestrationStrategy()

    input_data = [
        {"type": "objective_claim", "id": "c1", "text": "Medical claim"}
    ]

    await strategy.execute(
        input_data=input_data, tools=tools, agents=agents, state=initial_state
    )

    mock_agent.register_tools.assert_called_once_with(tools)


@pytest.mark.asyncio
async def test_alignment_orchestration_dynamic_routing(mock_agent, initial_state):
    """Verify strategy routes input claims to target agent and collects verdicts."""
    agents = {"ObjectiveClaimAgent": mock_agent}
    strategy = AlignmentOrchestrationStrategy()

    input_data = [
        {"type": "objective_claim", "id": "c1", "text": "Verify claim text"}
    ]

    result_state = await strategy.execute(
        input_data=input_data, tools={}, agents=agents, state=initial_state
    )

    mock_agent.diagnose_and_verify.assert_called_once_with(
        proposition_id="c1", claim_text="Verify claim text"
    )
    assert len(result_state.metadata["verdicts"]) == 1
    assert result_state.metadata["is_locked"] is True


@pytest.mark.asyncio
async def test_alignment_orchestration_unsupported_blocks(
    mock_agent, initial_state
):
    """Verify unsupported claim verdicts force state metadata is_locked to False."""
    mock_agent.diagnose_and_verify.return_value = ObjectiveClaimVerdict(
        status="UNSUPPORTED",
        proposition_id="c2",
        claim_text="Unsupported text",
        diagnostics=DiagnosticAnalysis.model_construct(),
    )

    agents = {"ObjectiveClaimAgent": mock_agent}
    strategy = AlignmentOrchestrationStrategy()

    input_data = [
        {"type": "objective_claim", "id": "c2", "text": "Unsupported text"}
    ]

    result_state = await strategy.execute(
        input_data=input_data, tools={}, agents=agents, state=initial_state
    )

    assert result_state.metadata["is_locked"] is False


@pytest.mark.asyncio
async def test_alignment_orchestration_unmapped_claim_type_handling(
    mock_agent, initial_state, caplog
):
    """Verify unmapped claim types trigger a warning and skip dispatching gracefully."""
    agents = {"ObjectiveClaimAgent": mock_agent}
    strategy = AlignmentOrchestrationStrategy()

    input_data = [
        {"type": "unknown_claim_type", "id": "c3", "text": "Unmapped text"}
    ]

    with caplog.at_level(logging.WARNING):
        result_state = await strategy.execute(
            input_data=input_data, tools={}, agents=agents, state=initial_state
        )

    mock_agent.diagnose_and_verify.assert_not_called()
    assert result_state.metadata["verdicts"] == []
    assert result_state.metadata["is_locked"] is True
    assert (
        "No agent configured/found for claim_type 'unknown_claim_type'"
        in caplog.text
    )
