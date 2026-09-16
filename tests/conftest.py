from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from auraflux_core.core.agents.base_agent import BaseAgent
from auraflux_core.core.agents.pipelines.base import BaseAgentPipeline
from auraflux_core.core.schemas.agents import AgentConfig
from auraflux_core.core.tools.base_tool import BaseTool


class DummyPipeline(BaseAgentPipeline):
    """Reusable DummyPipeline for testing pipeline execution and registry."""

    async def _run_pipeline(self, agent: Any, payload: Dict[str, Any]) -> Any:
        return {"status": "success", "payload": payload}


class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing purposes."""
    pass


class MockAgentConfig(AgentConfig):
    """Mock agent configuration inheriting from AgentConfig for testing."""
    name: str = "mock_agent"
    model: str = "mock-model"
    provider: str = "mock-provider"


@pytest.fixture
def dummy_pipeline():
    """Fixture providing a fresh DummyPipeline instance."""
    return DummyPipeline()


@pytest.fixture
def concrete_agent_factory():
    """Factory fixture to create ConcreteAgent instances."""
    def _create_agent(config: AgentConfig, client_manager: Any) -> ConcreteAgent:
        return ConcreteAgent(config=config, client_manager=client_manager)
    return _create_agent


@pytest.fixture
def mock_agent_config() -> AgentConfig:
    """Provides a default MockAgentConfig instance for unit tests."""
    return MockAgentConfig()


@pytest.fixture
def mock_agent_config_factory():
    """Provides a factory function to build MockAgentConfig with custom attributes."""
    def _create_config(**kwargs) -> AgentConfig:
        return MockAgentConfig(**kwargs)
    return _create_config


@pytest.fixture
def mock_client_manager():
    """Mock fixture for ClientManager across agent tests."""
    return MagicMock()


@pytest.fixture
def default_config():
    """Provides a fully-mocked AgentConfig with standard defaults."""
    config = MagicMock(spec=AgentConfig)
    config.name = "TestAgent"
    config.provider = "openai"
    config.model = "gpt-4o"
    config.pipeline_name = "direct"
    config.turn_limit = 10
    config.output_format = "TEXT"
    config.lang = "default"
    config.system_message = None
    config.prompt_config = None
    config.tools = None
    config.tool_call_protocol = "native"
    return config


class DummyTool(BaseTool):
    """Reusable DummyTool for testing agent and tool executor behaviors."""

    def get_name(self) -> str:
        return "dummy_tool"

    def get_description(self) -> str:
        return "A dummy tool for unit testing."

    def get_parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def run(self, **kwargs) -> str:
        return "dummy output"


@pytest.fixture
def dummy_tool():
    """Fixture providing a fresh DummyTool instance."""
    return DummyTool()


@pytest.fixture
def mock_embedding_model():
    """Mock for the BaseEmbedding model instance."""
    model = MagicMock()
    model.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return model


@pytest.fixture
def mock_opensearch_client():
    """Mock for the low-level OpenSearch SDK client."""
    return MagicMock()


@pytest.fixture
def sample_opensearch_response():
    """Sample raw OpenSearch search response payload."""
    return {
        "hits": {
            "hits": [
                {
                    "_id": "doc_1",
                    "_score": 0.95,
                    "_source": {
                        "text": "This is a test context chunk.",
                        "category": "ai_safety",
                    },
                },
                {
                    "_id": "doc_2",
                    "_score": 0.82,
                    "_source": {
                        "evidence_text": "Secondary evidence content.",
                    },
                },
            ]
        }
    }
