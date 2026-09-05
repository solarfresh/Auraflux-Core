from unittest.mock import AsyncMock, MagicMock

import pytest

from auraflux_core.core.tools.base_tool import BaseTool


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