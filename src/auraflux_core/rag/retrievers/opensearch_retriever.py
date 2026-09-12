import asyncio
from typing import Any, Callable, Dict, List, Optional

from opensearchpy import OpenSearch

from auraflux_core.rag.retrievers.base import BaseRetriever
from auraflux_core.rag.schemas.retrievers import (HybridQueryItem,
                                                  OpenSearchHybridConfig,
                                                  RetrievalResult)


class OpenSearchDSLBuilder:
    """Internal helper class translating OpenSearchHybridConfig into standard OpenSearch DSL."""

    MAX_HYBRID_QUERIES = 5

    @classmethod
    def _build_filter_clause(cls, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        filter_clauses: List[Dict[str, Any]] = []
        nested_group: Dict[str, List[Dict[str, Any]]] = {}

        for key, value in filters.items():
            if value is None:
                continue

            if "." in key:
                path = key.split(".")[0]
            else:
                path = None

            # 建立單一條件句
            if isinstance(value, list):
                clause = {"terms": {key: value}}
            elif isinstance(value, dict):
                clause = {"range": {key: value}}
            else:
                clause = {"term": {key: value}}

            if path:
                if path not in nested_group:
                    nested_group[path] = []
                nested_group[path].append(clause)
            else:
                filter_clauses.append(clause)

        for path, clauses in nested_group.items():
            if len(clauses) == 1:
                nested_query = clauses[0]
            else:
                nested_query = {"bool": {"must": clauses}}

            filter_clauses.append({"nested": {"path": path, "query": nested_query}})

        return filter_clauses

    @classmethod
    def build_hybrid_query(cls, config: OpenSearchHybridConfig) -> Dict[str, Any]:
        filter_clauses = cls._build_filter_clause(config.filters) if config.filters else []
        hybrid_queries: List[Dict[str, Any]] = []

        for item in config.query_items:

            if item.query_text and item.text_field:
                hybrid_queries.append({
                    "multi_match": {
                        "query": item.query_text,
                        "fields": [item.text_field]
                    }
                })

            if item.query_vector and item.vector_field:
                hybrid_queries.append({
                    "knn": {
                        item.vector_field: {
                            "vector": item.query_vector,
                            "k": config.top_k * 3
                        }
                    }
                })

        total_sub_queries = len(hybrid_queries)
        if total_sub_queries > cls.MAX_HYBRID_QUERIES:
            raise ValueError(
                f"Generated {total_sub_queries} sub-queries, which exceeds the OpenSearch hybrid "
                f"limit of {cls.MAX_HYBRID_QUERIES}. Reduce the number of query_items or field mappings."
            )

        hybrid_body: Dict[str, Any] = {"queries": hybrid_queries}

        if filter_clauses:
            if len(filter_clauses) == 1:
                hybrid_body["filter"] = filter_clauses[0]
            else:
                hybrid_body["filter"] = {"bool": {"filter": filter_clauses}}

        return {
            "size": config.top_k,
            "query": {
                "hybrid": hybrid_body
            }
        }


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
        default_search_pipeline: Optional[str] = "rrf_question_oriented",
        formatter_fn: Optional[Callable[[Dict[str, Any]], str]] = None
    ):
        """
        Args:
            client: OpenSearch client instance.
            embedding_model: Text embedding model instance.
            default_index_name: Target OpenSearch index name.
            default_search_pipeline: OpenSearch Hybrid Search Pipeline name.
            formatter_fn: Optional custom callable to transform `_source` dict into the target text string.
                          Defaults to a generic JSON sanitizer (stripping huge vector arrays).
        """
        super().__init__()

        self.service = OpenSearchService(client)
        self.embedding_model = embedding_model
        self.default_index_name = default_index_name

        # Generic defaults without hardcoded business schema paths
        self.default_search_pipeline = default_search_pipeline

        # Pluggable doc formatter function (defaults to generic JSON sanitizer)
        self.formatter_fn = formatter_fn or self._default_doc_formatter

    async def retrieve(
        self,
        query_items: List[HybridQueryItem],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        index_name: Optional[str] = None,
        routing: Optional[str] = None
    ) -> List[RetrievalResult]:
        """Executes multi-query hybrid vector + BM25 search and returns standardized RetrievalResults."""

        if not query_items:
            return []

        embeddings = await asyncio.gather(
            *(
                self.embedding_model.embed_query(item.query_text)
                for item in query_items
            )
        )

        for item, vector in zip(query_items, embeddings):
            item.query_vector = vector

        config = OpenSearchHybridConfig(
            query_items=query_items,
            top_k=top_k,
            filters=filters,
            search_pipeline=self.default_search_pipeline
        )
        dsl_body = OpenSearchDSLBuilder.build_hybrid_query(config)

        target_index = index_name or self.default_index_name

        hits = await self.service.search(
            index_name=target_index,
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
