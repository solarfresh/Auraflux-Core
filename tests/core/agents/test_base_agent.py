from unittest.mock import AsyncMock, MagicMock

import pytest

from auraflux_core.core.agents.base_agent import BaseAgent
from auraflux_core.core.agents.pipelines.base import BaseAgentPipeline
from auraflux_core.core.agents.pipelines.direct import DirectPipeline
from auraflux_core.core.schemas.agents import AgentConfig


class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing purposes."""

    generate: AsyncMock  # Explicit type annotation for Pylance

    def __init__(self, config: AgentConfig, client_manager: MagicMock):
        super().__init__(config=config, client_manager=client_manager)
        self.generate = AsyncMock()

    def get_system_message_map(self):
        return {"default": "Default System Prompt", "zh": "預設系統提示詞"}


@pytest.fixture
def mock_client_manager():
    return MagicMock()


@pytest.fixture
def default_config():
    config = MagicMock(spec=AgentConfig)
    config.name = "TestAgent"
    config.provider = "openai"
    config.model = "gpt-4o"
    config.pipeline_name = "direct"
    config.turn_limit = 10
    config.output_format = "TEXT"
    config.lang = "default"
    config.system_message = None
    config.tools = None
    config.tool_call_protocol = "native"  # Set default protocol to prevent AttributeError
    return config


# =============================================================================
# BaseAgent & Tool Registration Tests
# =============================================================================


def test_base_agent_init_binds_correct_pipeline(default_config, mock_client_manager):
    """Verify BaseAgent binds the appropriate Pipeline instance based on config.pipeline_name."""
    default_config.pipeline_name = "direct"
    agent = ConcreteAgent(config=default_config, client_manager=mock_client_manager)

    assert isinstance(agent.pipeline, BaseAgentPipeline)
    assert isinstance(agent.pipeline, DirectPipeline)


def test_register_tools_via_agent_delegates_to_executor(default_config, mock_client_manager, dummy_tool):
    """Verify calling agent.register_tools updates tool_registry in ToolExecutor."""
    agent = ConcreteAgent(config=default_config, client_manager=mock_client_manager)

    # Register via list
    agent.register_tools([dummy_tool])
    assert "dummy_tool" in agent.tool_executor.tool_registry
    assert agent.tool_executor.tool_registry["dummy_tool"] == dummy_tool

    # Register via dict
    dict_tools = {"dummy_tool_dict": dummy_tool}
    agent.register_tools(dict_tools)
    assert "dummy_tool" in agent.tool_executor.tool_registry


@pytest.mark.asyncio
async def test_agent_run_delegates_to_pipeline(default_config, mock_client_manager):
    """Verify calling agent.run(payload) delegates execution flow directly to bound Pipeline."""
    agent = ConcreteAgent(config=default_config, client_manager=mock_client_manager)

    mock_pipeline = MagicMock(spec=BaseAgentPipeline)
    mock_pipeline.execute = AsyncMock(return_value="pipeline_result")
    agent.pipeline = mock_pipeline

    payload = {"prompt": "Execute task"}
    result = await agent.run(payload)

    mock_pipeline.execute.assert_called_once_with(agent=agent, payload=payload)
    assert result == "pipeline_result"


def test_base_tool_executor_register_tools_formats(default_config, mock_client_manager, dummy_tool):
    """Verify ToolExecutor handles both List[BaseTool] and Dict[str, BaseTool] registration formats."""
    agent = ConcreteAgent(config=default_config, client_manager=mock_client_manager)
    executor = agent.tool_executor

    # 1. Register List format
    executor.register_tools([dummy_tool])
    assert "dummy_tool" in executor.tool_registry

    # 2. Register Dict format
    executor.tool_registry.clear()
    executor.register_tools({"tool_key": dummy_tool})
    assert "dummy_tool" in executor.tool_registry
