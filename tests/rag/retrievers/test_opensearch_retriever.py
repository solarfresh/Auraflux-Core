import pytest

from auraflux_core.rag.retrievers.opensearch_retriever import (
    OpenSearchDSLBuilder, OpenSearchHybridRetriever)
from auraflux_core.rag.schemas.retrievers import (OpenSearchHybridConfig,
                                                  RetrievalResult)


class TestOpenSearchDSLBuilder:
    def test_build_hybrid_query_basic(self):
        config = OpenSearchHybridConfig(
            query_text="python async",
            query_vector=[0.1, 0.2, 0.3],
            top_k=3,
            text_fields=["title^1.5", "content"],
            vector_fields=["content_vector"],
        )

        dsl = OpenSearchDSLBuilder.build_hybrid_query(config)

        assert dsl["size"] == 3
        assert "hybrid" in dsl["query"]
        queries = dsl["query"]["hybrid"]["queries"]
        assert len(queries) == 2
        assert queries[0]["multi_match"]["query"] == "python async"
        assert queries[0]["multi_match"]["fields"] == ["title^1.5", "content"]
        assert queries[1]["knn"]["content_vector"]["k"] == 9

    def test_build_hybrid_query_with_filters(self):
        config = OpenSearchHybridConfig(
            query_text="test",
            query_vector=[0.1, 0.2],
            top_k=5,
            filters={"project_id": "proj_123", "tags": ["v1", "v2"]},
            text_fields=["text"],
            vector_fields=["vec"],
        )

        dsl = OpenSearchDSLBuilder.build_hybrid_query(config)

        assert "bool" in dsl["query"]
        assert "filter" in dsl["query"]["bool"]
        filters = dsl["query"]["bool"]["filter"]
        assert len(filters) == 2
        assert {"term": {"project_id": "proj_123"}} in filters
        assert {"terms": {"tags": ["v1", "v2"]}} in filters


@pytest.mark.asyncio
class TestOpenSearchHybridRetriever:
    async def test_retrieve_success(
        self, mock_opensearch_client, mock_embedding_model, sample_opensearch_response
    ):
        mock_opensearch_client.search.return_value = sample_opensearch_response

        retriever = OpenSearchHybridRetriever(
            client=mock_opensearch_client,
            embedding_model=mock_embedding_model,
            default_index_name="test_index",
        )

        results = await retriever.retrieve(
            query_text="safety standards",
            top_k=2,
            filters={"project_id": "proj_abc"},
        )

        # 1. Verify embedding model invocation via BaseEmbedding interface
        mock_embedding_model.embed_query.assert_awaited_once_with("safety standards")

        # 2. Verify OpenSearch client call
        assert mock_opensearch_client.search.called
        call_kwargs = mock_opensearch_client.search.call_args.kwargs
        assert call_kwargs["index"] == "test_index"
        assert call_kwargs["params"]["search_pipeline"] == "rrf_question_oriented"

        # 3. Verify standard result mapping
        assert len(results) == 2
        assert isinstance(results[0], RetrievalResult)
        assert results[0].id == "doc_1"
        assert results[0].content['text'] == "This is a test context chunk."
        assert results[0].score == 0.95
        assert results[0].metadata["category"] == "ai_safety"

        assert results[1].id == "doc_2"
        assert results[1].content['evidence_text'] == "Secondary evidence content."

    async def test_retrieve_empty_results(
        self, mock_opensearch_client, mock_embedding_model
    ):
        mock_opensearch_client.search.return_value = {"hits": {"hits": []}}

        retriever = OpenSearchHybridRetriever(
            client=mock_opensearch_client,
            embedding_model=mock_embedding_model,
            default_index_name="empty_index",
        )

        results = await retriever.retrieve(query_text="nonexistent query")

        assert results == []
