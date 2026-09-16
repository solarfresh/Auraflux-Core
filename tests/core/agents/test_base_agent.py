from unittest.mock import AsyncMock, MagicMock

import pytest

from auraflux_core.core.agents.pipelines.base import BaseAgentPipeline
from auraflux_core.core.agents.pipelines.direct import DirectPipeline
from auraflux_core.core.schemas.agents import AgentConfig


# =============================================================================
# BaseAgent & Tool Registration Tests
# =============================================================================


def test_base_agent_init_binds_correct_pipeline(
    default_config, mock_client_manager, concrete_agent_factory
):
    """Verify BaseAgent binds the appropriate Pipeline instance based on config.pipeline_name."""
    default_config.pipeline_name = "direct"
    agent = concrete_agent_factory(config=default_config, client_manager=mock_client_manager)

    assert isinstance(agent.pipeline, BaseAgentPipeline)
    assert isinstance(agent.pipeline, DirectPipeline)


def test_register_tools_via_agent_delegates_to_executor(
    default_config, mock_client_manager, dummy_tool, concrete_agent_factory
):
    """Verify calling agent.register_tools updates tool_registry in ToolExecutor."""
    agent = concrete_agent_factory(config=default_config, client_manager=mock_client_manager)

    # Register via list
    agent.register_tools([dummy_tool])
    assert "dummy_tool" in agent.tool_executor.tool_registry
    assert agent.tool_executor.tool_registry["dummy_tool"] == dummy_tool

    # Register via dict
    dict_tools = {"dummy_tool_dict": dummy_tool}
    agent.register_tools(dict_tools)
    assert "dummy_tool" in agent.tool_executor.tool_registry


@pytest.mark.asyncio
async def test_agent_run_delegates_to_pipeline(
    default_config, mock_client_manager, concrete_agent_factory
):
    """Verify calling agent.run(payload) delegates execution flow directly to bound Pipeline."""
    agent = concrete_agent_factory(config=default_config, client_manager=mock_client_manager)

    mock_pipeline = MagicMock(spec=BaseAgentPipeline)
    mock_pipeline.execute = AsyncMock(return_value="pipeline_result")
    agent.pipeline = mock_pipeline

    payload = {"prompt": "Execute task"}
    result = await agent.run(payload)

    mock_pipeline.execute.assert_called_once_with(agent=agent, payload=payload)
    assert result == "pipeline_result"


def test_base_tool_executor_register_tools_formats(
    default_config, mock_client_manager, dummy_tool, concrete_agent_factory
):
    """Verify ToolExecutor handles both List[BaseTool] and Dict[str, BaseTool] registration formats."""
    agent = concrete_agent_factory(config=default_config, client_manager=mock_client_manager)
    executor = agent.tool_executor

    # 1. Register List format
    executor.register_tools([dummy_tool])
    assert "dummy_tool" in executor.tool_registry

    # 2. Register Dict format
    executor.tool_registry.clear()
    executor.register_tools({"tool_key": dummy_tool})
    assert "dummy_tool" in executor.tool_registry
