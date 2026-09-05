import asyncio
from typing import Any, Dict, List, Optional

from opensearchpy import OpenSearch

from auraflux_core.core.embeddings.base_embedding import BaseEmbedding
from auraflux_core.rag.retrievers.base import BaseRetriever
from auraflux_core.rag.schemas.retrievers import (OpenSearchHybridConfig,
                                                  RetrievalResult)


class OpenSearchDSLBuilder:
    """Internal helper class translating OpenSearchHybridConfig into standard OpenSearch DSL."""

    @staticmethod
    def _build_filter_clause(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        filter_clauses = []
        for key, value in filters.items():
            if isinstance(value, list):
                filter_clauses.append({"terms": {key: value}})
            else:
                filter_clauses.append({"term": {key: value}})
        return filter_clauses

    @classmethod
    def build_hybrid_query(cls, config: OpenSearchHybridConfig) -> Dict[str, Any]:
        hybrid_queries: List[Dict[str, Any]] = []

        if config.text_fields:
            hybrid_queries.append({
                "multi_match": {
                    "query": config.query_text,
                    "fields": config.text_fields
                }
            })

        for vec_field in config.vector_fields:
            hybrid_queries.append({
                "knn": {
                    vec_field: {
                        "vector": config.query_vector,
                        "k": config.top_k * 3
                    }
                }
            })

        query_body: Dict[str, Any] = {"size": config.top_k}

        if config.filters:
            filter_clauses = cls._build_filter_clause(config.filters)
            query_body["query"] = {
                "bool": {
                    "must": [{"hybrid": {"queries": hybrid_queries}}],
                    "filter": filter_clauses
                }
            }
        else:
            query_body["query"] = {"hybrid": {"queries": hybrid_queries}}

        return query_body


class OpenSearchService:
    """Low-level infrastructure driver wrapper executing async requests."""

    def __init__(self, client: OpenSearch):
        self.client = client

    async def search(
        self,
        index_name: str,
        body: Dict[str, Any],
        routing: Optional[str] = None,
        search_pipeline: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        params = {}
        if search_pipeline:
            params["search_pipeline"] = search_pipeline
        if routing:
            params["routing"] = routing

        response = await asyncio.to_thread(
            self.client.search, index=index_name, body=body, params=params
        )
        return response.get("hits", {}).get("hits", [])


class OpenSearchHybridRetriever(BaseRetriever):
    """Configurable OpenSearch retriever accepting customizable text and vector fields."""

    def __init__(
        self,
        client: OpenSearch,
        embedding_model: BaseEmbedding,
        default_index_name: str,
        text_fields: Optional[List[str]] = None,
        vector_fields: Optional[List[str]] = None,
        default_search_pipeline: Optional[str] = "rrf_question_oriented"
    ):
        self.service = OpenSearchService(client)
        self.embedding_model = embedding_model
        self.default_index_name = default_index_name
        self.text_fields = text_fields or ["target_question", "concept_title^1.5", "evidence_text"]
        self.vector_fields = vector_fields or ["question_vector", "concept_vector", "evidence_vector"]
        self.default_search_pipeline = default_search_pipeline

    async def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        index_name: Optional[str] = None,
        routing: Optional[str] = None
    ) -> List[RetrievalResult]:
        query_vector = await self.embedding_model.embed_query(query_text)

        config = OpenSearchHybridConfig(
            query_text=query_text,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            text_fields=self.text_fields,
            vector_fields=self.vector_fields,
            search_pipeline=self.default_search_pipeline
        )
        dsl_body = OpenSearchDSLBuilder.build_hybrid_query(config)

        hits = await self.service.search(
            index_name=index_name or self.default_index_name,
            body=dsl_body,
            routing=routing,
            search_pipeline=config.search_pipeline
        )

        results: List[RetrievalResult] = []
        for hit in hits:
            source = hit.get("_source", {})
            extracted_text = (
                source.get("text")
                or source.get("content")
                or str(source.get("evidence_text", ""))
                or source.get("target_question")
                or ""
            )

            results.append(
                RetrievalResult(
                    id=str(hit.get("_id", "")),
                    text=extracted_text,
                    score=hit.get("_score"),
                    metadata=source
                )
            )

        return results
