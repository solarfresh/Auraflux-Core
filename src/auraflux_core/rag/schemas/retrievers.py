from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HybridSearchConfig(BaseModel):
    """
    Vendor-agnostic configuration model encapsulating essential parameters
    for executing a hybrid (lexical + dense vector) search query.
    """
    query_text: str = Field(
        ...,
        min_length=1,
        description="Lexical query text for keyword/BM25 matching."
    )
    query_vector: List[float] = Field(
        ...,
        description="Dense vector embedding of the query for semantic search."
    )
    top_k: int = Field(
        default=5,
        gt=0,
        description="Maximum number of top search hits to retrieve."
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Generic key-value criteria for payload/metadata filtering (e.g., {'project_id': 'proj_123'})."
    )
    search_pipeline: Optional[str] = Field(
        default=None,
        description="Optional re-ranking or search execution pipeline/strategy identifier."
    )


class OpenSearchHybridConfig(HybridSearchConfig):
    """OpenSearch-specific hybrid search parameters extending the base config."""
    text_fields: List[str] = Field(
        default_factory=lambda: ["text"],
        description="List of text fields for multi_match query, supporting boost notation (e.g., 'title^1.5')."
    )
    vector_fields: List[str] = Field(
        default_factory=lambda: ["vector"],
        description="List of vector field names to construct kNN streams."
    )


class HybridRetrieverInput(BaseModel):
    """Input parameters model for the generic agent HybridRetrieverTool."""
    query_text: str = Field(
        ...,
        description="The query statement or keywords used for semantic and text search."
    )
    top_k: int = Field(
        default=5,
        gt=0,
        description="Maximum number of context chunks to retrieve."
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Generic metadata filtering key-value criteria (e.g., {'project_id': 'proj_123'})."
    )


class RetrievalResult(BaseModel):
    """
    Vendor-agnostic data model representing a unified retrieval hit
    across heterogeneous vector stores and search engines.
    """
    id: str = Field(
        ...,
        description="Universal identifier for the retrieved chunk or document entity."
    )
    content: Any = Field(
        ...,
        description="The formatted document content payload (e.g., structured dict, domain object, or sanitized string) produced by the retriever's formatter function."
    )
    score: Optional[float] = Field(
        default=None,
        description="Normalized or engine-specific relevance score (e.g., RRF score, cosine similarity)."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata payload (e.g., project_id, file_id, chunk_id, location)."
    )
