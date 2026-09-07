from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HybridQueryItem(BaseModel):
    """Represents a single query target bound to its specific index fields."""
    query_text: str
    query_vector: Optional[List[float]]
    text_field: str
    vector_field: str


class HybridSearchConfig(BaseModel):
    """
    Vendor-agnostic configuration model encapsulating essential parameters
    for executing a hybrid (lexical + dense vector) search query.
    """
    query_items: List[HybridQueryItem] = Field(
        default_factory=list,
        description="Collection of multiple query items, each with 1-to-1 field mapping."
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


class OpenSearchHybridConfig(HybridSearchConfig):
    """OpenSearch-specific hybrid search configuration supporting multi-query fusion."""
    search_pipeline: Optional[str] = Field(
        default=None,
        description="Name of the OpenSearch search pipeline configured for score normalization or RRF."
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
