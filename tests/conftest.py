from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_embedding_client():
    """Mock for the generic asynchronous embedding client."""
    client = MagicMock()
    client.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return client


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