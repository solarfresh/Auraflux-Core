import pytest

from auraflux_core.rag.retrievers.opensearch_retriever import (
    OpenSearchDSLBuilder, OpenSearchHybridRetriever)
from auraflux_core.rag.schemas.retrievers import (HybridQueryItem,
                                                  OpenSearchHybridConfig,
                                                  RetrievalResult)


class TestOpenSearchDSLBuilder:
    def test_build_hybrid_query_basic(self):
        items = [
            HybridQueryItem(
                query_text="python async",
                query_vector=[0.1, 0.2, 0.3],
                text_field="title",
                vector_field="content_vector",
            )
        ]
        config = OpenSearchHybridConfig(
            query_items=items,
            top_k=3,
        )

        dsl = OpenSearchDSLBuilder.build_hybrid_query(config)

        assert dsl["size"] == 3
        assert "hybrid" in dsl["query"]

        hybrid_body = dsl["query"]["hybrid"]
        queries = hybrid_body["queries"]

        assert len(queries) == 2
        assert queries[0]["multi_match"]["query"] == "python async"
        assert queries[0]["multi_match"]["fields"] == ["title"]
        assert queries[1]["knn"]["content_vector"]["vector"] == [0.1, 0.2, 0.3]
        assert queries[1]["knn"]["content_vector"]["k"] == 9

    def test_build_hybrid_query_with_filters(self):
        items = [
            HybridQueryItem(
                query_text="test",
                query_vector=[0.1, 0.2],
                text_field="text",
                vector_field="vec",
            )
        ]
        config = OpenSearchHybridConfig(
            query_items=items,
            top_k=5,
            filters={"project_id": "proj_123", "tags": ["v1", "v2"]},
        )

        dsl = OpenSearchDSLBuilder.build_hybrid_query(config)

        assert "hybrid" in dsl["query"]
        hybrid_body = dsl["query"]["hybrid"]

        assert "filter" in hybrid_body
        filter_clause = hybrid_body["filter"]
        assert "bool" in filter_clause
        assert "filter" in filter_clause["bool"]

        filters = filter_clause["bool"]["filter"]
        assert len(filters) == 2
        assert {"term": {"project_id": "proj_123"}} in filters
        assert {"terms": {"tags": ["v1", "v2"]}} in filters

    def test_build_hybrid_query_exceeds_max_queries_raises_error(self):
        items = [
            HybridQueryItem(
                query_text=f"query {i}",
                query_vector=[0.1, 0.2],
                text_field=f"text_{i}",
                vector_field=f"vec_{i}",
            )
            for i in range(3)
        ]
        config = OpenSearchHybridConfig(query_items=items, top_k=5)

        with pytest.raises(ValueError) as exc_info:
            OpenSearchDSLBuilder.build_hybrid_query(config)

        assert "exceeds the OpenSearch hybrid limit of 5" in str(exc_info.value)


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

        query_items = [
            HybridQueryItem(
                query_text="safety standards",
                query_vector=None,
                text_field="evidence_text",
                vector_field="evidence_vector",
            )
        ]

        results = await retriever.retrieve(
            query_items=query_items,
            top_k=2,
            filters={"project_id": "proj_abc"},
        )

        mock_embedding_model.embed_query.assert_awaited_once_with("safety standards")

        assert mock_opensearch_client.search.called
        call_kwargs = mock_opensearch_client.search.call_args.kwargs
        assert call_kwargs["index"] == "test_index"
        assert call_kwargs["params"]["search_pipeline"] == "rrf_question_oriented"

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

        query_items = [
            HybridQueryItem(
                query_text="nonexistent query",
                query_vector=None,
                text_field="text",
                vector_field="vector",
            )
        ]

        results = await retriever.retrieve(query_items=query_items)

        assert results == []

    async def test_retrieve_empty_query_items(
        self, mock_opensearch_client, mock_embedding_model
    ):
        retriever = OpenSearchHybridRetriever(
            client=mock_opensearch_client,
            embedding_model=mock_embedding_model,
            default_index_name="test_index",
        )

        results = await retriever.retrieve(query_items=[])

        assert results == []
        mock_embedding_model.embed_query.assert_not_called()
        mock_opensearch_client.search.assert_not_called()
