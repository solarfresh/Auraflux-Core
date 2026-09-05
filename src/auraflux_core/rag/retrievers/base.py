from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from auraflux_core.rag.schemas.retrievers import RetrievalResult


class BaseRetriever(ABC):
    """
    Abstract Base Class defining the universal execution interface
    for all vendor-specific retrieval engines.
    """

    @abstractmethod
    async def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[RetrievalResult]:
        """
        Asynchronously executes document retrieval and returns a unified list of RetrievalResult entities.

        Args:
            query_text (str): Lexical query string for search.
            top_k (int): Maximum number of top context hits to return.
            filters (Optional[Dict[str, Any]]): Generic key-value criteria for metadata filtering.
            **kwargs: Vendor-specific execution options (e.g., routing, index_name override).

        Returns:
            List[RetrievalResult]: Standardized retrieval results.
        """
        pass
