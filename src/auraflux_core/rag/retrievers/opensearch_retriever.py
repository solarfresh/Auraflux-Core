import asyncio
import json
from typing import Any, Callable, Dict, List, Optional

from opensearchpy import OpenSearch

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
    """Generic & configurable OpenSearch hybrid retriever.

    Agnostic to domain schemas, business layers, or specific Index structures.
    """

    def __init__(
        self,
        client: Any,
        embedding_model: Any,
        default_index_name: str,
        text_fields: Optional[List[str]] = None,
        vector_fields: Optional[List[str]] = None,
        default_search_pipeline: Optional[str] = "rrf_question_oriented",
        formatter_fn: Optional[Callable[[Dict[str, Any]], str]] = None
    ):
        """
        Args:
            client: OpenSearch client instance.
            embedding_model: Text embedding model instance.
            default_index_name: Target OpenSearch index name.
            text_fields: Fields for BM25 full-text search (e.g., ["title^2.0", "content"]).
            vector_fields: Fields for k-NN vector search (e.g., ["title_vector", "content_vector"]).
            default_search_pipeline: OpenSearch Hybrid Search Pipeline name.
            formatter_fn: Optional custom callable to transform `_source` dict into the target text string.
                          Defaults to a generic JSON sanitizer (stripping huge vector arrays).
        """
        self.service = OpenSearchService(client)
        self.embedding_model = embedding_model
        self.default_index_name = default_index_name

        # Generic defaults without hardcoded business schema paths
        self.text_fields = text_fields or ["title^2.0", "content", "text"]
        self.vector_fields = vector_fields or ["vector", "embedding"]
        self.default_search_pipeline = default_search_pipeline

        # Pluggable doc formatter function (defaults to generic JSON sanitizer)
        self.formatter_fn = formatter_fn or self._default_doc_formatter

    async def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        index_name: Optional[str] = None,
        routing: Optional[str] = None
    ) -> List[RetrievalResult]:
        """Executes hybrid vector + BM25 search and returns standardized RetrievalResults."""

        # Generate query vector asynchronously
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

            # Format raw document via customizable formatter without hardcoding business fields
            formatted_content = self.formatter_fn(source)

            results.append(
                RetrievalResult(
                    id=str(hit.get("_id", "")),
                    content=formatted_content,
                    score=hit.get("_score"),
                    metadata=source  # Unaltered raw _source preserved for downstream extraction
                )
            )

        return results

    @staticmethod
    def _default_doc_formatter(source: Dict[str, Any]) -> Dict[str, Any]:
        """Generic fallback doc sanitizer:

        Serializes `_source` to JSON while stripping large vector/embedding arrays
        to reduce LLM prompt token consumption.
        """
        # Automatically strip high-dimensional array fields containing 'vector' or 'embedding'
        sanitizing_keys = [
            k for k, v in source.items()
            if isinstance(v, list) and ("vector" in k.lower() or "embedding" in k.lower())
        ]

        clean_source = {k: v for k, v in source.items() if k not in sanitizing_keys}

        # Defensive cleanup if 'vectors' sub-object exists
        if "vectors" in clean_source and isinstance(clean_source["vectors"], dict):
            del clean_source["vectors"]

        return clean_source
