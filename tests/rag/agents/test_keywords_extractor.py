import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from auraflux_core.rag.agents.keywords_extractor import ExtractKeywordsAgent


@pytest.fixture
def mock_extract_keywords_agent(mock_agent_config, mock_client_manager):
    """
    Fixture providing an ExtractKeywordsAgent instance configured
    using standard conftest fixtures.
    """
    agent = ExtractKeywordsAgent(
        config=mock_agent_config,
        client_manager=mock_client_manager
    )
    # Mock internal tool_executor to isolate agent logic from actual tools
    agent.tool_executor = MagicMock()
    agent.tool_executor.run = AsyncMock()
    return agent


@pytest.mark.asyncio
async def test_process_incremental_triples_success_with_flagged_data(
    mock_extract_keywords_agent
):
    """
    Test Case 1: Verify state updates when ExtractKeywordsAgent handles flagged triples.
    Ensures clean_triples, flagged_triples, and flagged_reasons are populated.
    """
    # 1. Arrange
    plan_output = {
        "triples": [
            {"subject": "it", "predicate": "has_value", "object": "100"},
            {"subject": "Apple", "predicate": "located_in", "object": "USA"}
        ]
    }

    processor_res = MagicMock(content=plan_output["triples"])
    checker_payload = {
        "has_flagged": True,
        "clean_triples": [
            {"subject": "Apple", "predicate": "located_in", "object": "USA"}
        ],
        "flagged_triples": [
            {
                "subject": "it",
                "predicate": "has_value",
                "object": "100",
                "_flag_reasons": ["Missing subject_resolved for anaphoric pronoun 'it'"]
            }
        ]
    }
    checker_res = MagicMock(content=json.dumps(checker_payload))

    # Configure mock responses for sequential tool calls
    mock_extract_keywords_agent.tool_executor.run.side_effect = [
        processor_res,
        checker_res
    ]

    # 2. Act
    has_flagged, reasons = await mock_extract_keywords_agent._process_incremental_triples(
        plan_output
    )

    # 3. Assert
    assert has_flagged is True
    assert len(reasons) == 1
    assert reasons[0] == "Missing subject_resolved for anaphoric pronoun 'it'"

    # Verify state write-back in plan_output
    assert "clean_triples" in plan_output
    assert len(plan_output["clean_triples"]) == 1
    assert plan_output["clean_triples"] == checker_payload["clean_triples"]

    assert plan_output["flagged_triples"] == checker_payload["flagged_triples"]
    assert plan_output["_flagged_triples"] == checker_payload["flagged_triples"]
    assert plan_output["flagged_reasons"] == ["Missing subject_resolved for anaphoric pronoun 'it'"]
    assert len(plan_output["triples"]) == 2  # Total combined triples (clean + flagged)


@pytest.mark.asyncio
async def test_process_incremental_triples_deduplication(
    mock_extract_keywords_agent
):
    """
    Test Case 2: Verify that clean triples are properly deduplicated and updated
    based on the (subject, predicate, object) tuple identity across passes.
    """
    # 1. Arrange
    plan_output = {
        "_accumulated_clean_triples": [
            {"subject": "Apple", "predicate": "located_in", "object": "USA", "version": 1}
        ],
        "triples": [
            {"subject": "Apple", "predicate": "located_in", "object": "USA", "version": 2}
        ]
    }

    processor_res = MagicMock(content=plan_output["triples"])
    checker_payload = {
        "has_flagged": False,
        "clean_triples": [
            {"subject": "Apple", "predicate": "located_in", "object": "USA", "version": 2}
        ],
        "flagged_triples": []
    }
    checker_res = MagicMock(content=json.dumps(checker_payload))

    mock_extract_keywords_agent.tool_executor.run.side_effect = [
        processor_res,
        checker_res
    ]

    # 2. Act
    has_flagged, reasons = await mock_extract_keywords_agent._process_incremental_triples(
        plan_output
    )

    # 3. Assert
    assert has_flagged is False
    assert len(reasons) == 0
    assert len(plan_output["clean_triples"]) == 1
    # Verify that the older triple version was updated/overwritten in-place
    assert plan_output["clean_triples"][0]["version"] == 2
