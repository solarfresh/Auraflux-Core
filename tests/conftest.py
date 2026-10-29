from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from auraflux_core.core.agents.base import BaseAgent
from auraflux_core.core.agents.pipelines.base import BaseAgentPipeline
from auraflux_core.core.orchestrators.state import OrchestratorState
from auraflux_core.core.schemas.agents import AgentConfig
from auraflux_core.core.schemas.messages import PromptConfig, StageSpec
from auraflux_core.core.tools.base_tool import BaseTool

DEFAULT_AGENT_PARAMS = {
    "name": "test_agent",
    "provider": "openai",
    "model": "gpt-4",
    "pipeline_name": "direct",
    "turn_limit": 10,
    "output_format": "TEXT",
    "lang": "default",
    "system_message": None,
    "prompt_config": None,
    "tools": None,
    "tool_call_protocol": "NATIVE",
}


class DummyPipeline(BaseAgentPipeline):
    """Reusable DummyPipeline for testing pipeline execution and registry."""

    async def _execute_pipeline(self, agent: Any, payload: Dict[str, Any]) -> Any:
        return {"status": "success", "payload": payload}

    def get_execute_pipeline(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Declares dummy fine-grained execution steps for testing purposes."""
        return [
            {"method_name": "_dummy_step", "kwargs": {"payload": payload}}
        ]


class DummyPipelineForStrategy(BaseAgentPipeline):
    """Dummy pipeline providing fine-grained steps for strategy execution testing."""

    async def _execute_pipeline(self, agent: Any, payload: Dict[str, Any]) -> Any:
        return {"status": "success", "payload": payload}

    def get_execute_pipeline(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {"method_name": "step_one", "kwargs": {"extra": "val_1"}},
            {"method_name": "step_two", "kwargs": {}},
        ]


class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing purposes."""
    pass


class MockAgentConfig(AgentConfig):
    """Mock agent configuration inheriting from AgentConfig for testing."""
    name: str = "mock_agent"
    model: str = "mock-model"
    provider: str = "mock-provider"


@pytest.fixture
def base_agent_params():
    from operator import itemgetter

    keys_to_extract = ['name', 'provider', 'model']
    getter = itemgetter(*keys_to_extract)
    return dict(zip(keys_to_extract, getter(DEFAULT_AGENT_PARAMS)))


@pytest.fixture
def dummy_pipeline():
    """Fixture providing a fresh DummyPipeline instance."""
    return DummyPipeline()


@pytest.fixture
def dummy_pipeline_for_strategy():
    """Fixture providing a DummyPipelineForStrategy instance."""
    return DummyPipelineForStrategy()


@pytest.fixture
def concrete_agent_factory():
    """Factory fixture to create ConcreteAgent instances."""
    def _create_agent(config: AgentConfig, client_manager: Any) -> ConcreteAgent:
        return ConcreteAgent(config=config, client_manager=client_manager)
    return _create_agent


@pytest.fixture
def mock_agent_config() -> AgentConfig:
    """Provides a default MockAgentConfig instance for unit tests."""
    stage_spec = StageSpec(
        template_variables={"dummy": "description"},
        templates={"en": "some template"}
    )
    object.__setattr__(stage_spec, "get_template", MagicMock(return_value=None))

    prompt_config = PromptConfig(
        stages={
            "stage_without_lang_template": stage_spec
        }
    )

    return MockAgentConfig(prompt_config=prompt_config)


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
    for key, value in DEFAULT_AGENT_PARAMS.items():
        setattr(config, key, value)
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
def mock_orchestrator_state() -> OrchestratorState:
    """Provides a fresh OrchestratorState instance for testing."""
    return OrchestratorState(session_id="test_session_123")


@pytest.fixture
def mock_agents_registry() -> Dict[str, Any]:
    """Provides a mock dictionary of available agents for routing."""
    class DummyAgent:
        def __init__(self, name: str):
            self.name = name

    return {
        "agent_a": DummyAgent("agent_a"),
        "agent_b": DummyAgent("agent_b"),
        "agent_c": DummyAgent("agent_c"),
    }


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
