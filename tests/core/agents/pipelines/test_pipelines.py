import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from auraflux_core.core.agents.base_agent import BaseAgent
from auraflux_core.core.agents.pipelines.direct import DirectPipeline
from auraflux_core.core.agents.pipelines.plan_and_execute import (
    PlanAndExecuteHandler, PlanAndExecutePipeline)
from auraflux_core.core.schemas.messages import Message


class MockAgentWithHandler(BaseAgent, PlanAndExecuteHandler):
    """Mock Agent inheriting BaseAgent and PlanAndExecuteHandler for type compliance."""

    generate: AsyncMock

    def __init__(self):
        mock_config = MagicMock()
        mock_config.name = "MockPlanAgent"
        mock_config.pipeline_name = "plan_and_execute"
        mock_client_manager = MagicMock()

        super().__init__(config=mock_config, client_manager=mock_client_manager)

        self.generate = AsyncMock()
        self.tool_executor = MagicMock()
        self.output_parser = MagicMock()

    @property
    def name(self) -> str:
        return "MockPlanAgent"

    def get_system_message_map(self):
        return {"default": "System Prompt"}

    def build_plan_messages(self, payload):
        return [Message(role="user", content="Plan stage prompt", name=self.name)]

    def extract_tool_call_spec(self, payload, plan_output):
        if plan_output.get("needs_tool"):
            return {
                "tool_name": "search_tool",
                "tool_args": {"query": plan_output.get("query")},
            }
        return None

    def build_synthesis_messages(self, payload, plan_output, tool_results):
        evidence = tool_results[0].content if tool_results else "No evidence"
        return [
            Message(
                role="user",
                content=f"Synthesis stage prompt with evidence: {evidence}",
                name=self.name,
            )
        ]

    def parse_final_output(self, payload, plan_output, raw_llm_output):
        return {"status": "SUCCESS", "result": raw_llm_output}


class MockAgentWithoutHandler(BaseAgent):
    """Mock Agent inheriting BaseAgent without PlanAndExecuteHandler."""

    generate: AsyncMock

    def __init__(self):
        mock_config = MagicMock()
        mock_config.name = "MockSimpleAgent"
        mock_config.pipeline_name = "direct"
        mock_client_manager = MagicMock()

        super().__init__(config=mock_config, client_manager=mock_client_manager)

        self.generate = AsyncMock()

    @property
    def name(self) -> str:
        return "MockSimpleAgent"

    def get_system_message_map(self):
        return {"default": "System Prompt"}


# =============================================================================
# DirectPipeline Tests
# =============================================================================


@pytest.mark.asyncio
async def test_direct_pipeline_with_messages_payload():
    """Verify DirectPipeline extracts messages list directly from payload and calls agent.generate."""
    pipeline = DirectPipeline()
    mock_agent = MockAgentWithoutHandler()
    messages = [Message(role="user", content="Hello world", name="User")]
    mock_agent.generate.return_value = Message(
        role="assistant", content="Direct response", name=mock_agent.name
    )

    payload = {"messages": messages}
    result = await pipeline.execute(agent=mock_agent, payload=payload)

    mock_agent.generate.assert_called_once_with(messages)
    assert result == "Direct response"


@pytest.mark.asyncio
async def test_direct_pipeline_with_prompt_payload():
    """Verify DirectPipeline wraps prompt string into Message list when given a prompt payload."""
    pipeline = DirectPipeline()
    mock_agent = MockAgentWithoutHandler()
    mock_agent.generate.return_value = Message(
        role="assistant", content="Prompt response", name=mock_agent.name
    )

    payload = {"prompt": "Single prompt string"}
    result = await pipeline.execute(agent=mock_agent, payload=payload)

    call_args = mock_agent.generate.call_args[0][0]
    assert len(call_args) == 1
    assert call_args[0].content == "Single prompt string"
    assert result == "Prompt response"


@pytest.mark.asyncio
async def test_direct_pipeline_with_generic_dict_payload():
    """Verify DirectPipeline serializes dict payload to JSON string when no explicit key exists."""
    pipeline = DirectPipeline()
    mock_agent = MockAgentWithoutHandler()
    mock_agent.generate.return_value = Message(
        role="assistant", content="Dict response", name=mock_agent.name
    )

    payload = {"key": "value", "count": 10}
    result = await pipeline.execute(agent=mock_agent, payload=payload)

    call_args = mock_agent.generate.call_args[0][0]
    assert len(call_args) == 1
    assert json.loads(call_args[0].content) == payload
    assert result == "Dict response"


# =============================================================================
# PlanAndExecutePipeline Tests
# =============================================================================


@pytest.mark.asyncio
async def test_plan_and_execute_pipeline_full_flow():
    """Verify PlanAndExecutePipeline executes Stage 1 (Plan), Stage 2 (Tool), and Stage 3 (Synthesis)."""
    pipeline = PlanAndExecutePipeline()
    agent = MockAgentWithHandler()

    stage1_msg = Message(
        role="assistant",
        content='{"needs_tool": true, "query": "test query"}',
        name=agent.name,
    )
    stage3_msg = Message(
        role="assistant", content="Final synthesized answer", name=agent.name
    )

    agent.generate.side_effect = [stage1_msg, stage3_msg]
    agent.output_parser.parse_json.return_value = {
        "needs_tool": True,
        "query": "test query",
    }

    agent.tool_executor.tool_registry = {"search_tool": MagicMock()}
    agent.tool_executor.run = AsyncMock(
        return_value=Message(
            role="tool", content="Found test evidence", name="search_tool"
        )
    )

    payload = {"task": "verify claim"}
    result = await pipeline.execute(agent=agent, payload=payload)

    assert agent.generate.call_count == 2
    agent.tool_executor.run.assert_called_once_with(
        tool_name="search_tool", tool_args={"query": "test query"}
    )
    assert result == {"status": "SUCCESS", "result": "Final synthesized answer"}


@pytest.mark.asyncio
async def test_plan_and_execute_pipeline_without_tool_execution():
    """Verify PlanAndExecutePipeline skips tool execution when extract_tool_call_spec returns None."""
    pipeline = PlanAndExecutePipeline()
    agent = MockAgentWithHandler()

    stage1_msg = Message(
        role="assistant", content='{"needs_tool": false}', name=agent.name
    )
    stage3_msg = Message(
        role="assistant", content="Direct synthesized answer", name=agent.name
    )

    agent.generate.side_effect = [stage1_msg, stage3_msg]
    agent.output_parser.parse_json.return_value = {"needs_tool": False}

    payload = {"task": "verify claim without tool"}
    result = await pipeline.execute(agent=agent, payload=payload)

    assert agent.generate.call_count == 2
    agent.tool_executor.run.assert_not_called()
    assert result == {"status": "SUCCESS", "result": "Direct synthesized answer"}


@pytest.mark.asyncio
async def test_plan_and_execute_pipeline_raises_type_error_for_invalid_agent():
    """Verify PlanAndExecutePipeline raises TypeError if agent does not implement PlanAndExecuteHandler."""
    pipeline = PlanAndExecutePipeline()
    invalid_agent = MockAgentWithoutHandler()

    payload = {"task": "test"}
    with pytest.raises(TypeError) as exc_info:
        await pipeline.execute(agent=invalid_agent, payload=payload)

    assert "must implement PlanAndExecuteHandler interface" in str(exc_info.value)
