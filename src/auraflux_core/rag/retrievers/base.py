from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from auraflux_core.core.configs.logging_config import setup_logging
from auraflux_core.rag.schemas.retrievers import (HybridQueryItem,
                                                  RetrievalResult)


class BaseRetriever(ABC):
    """
    Abstract Base Class defining the universal execution interface
    for all vendor-specific retrieval engines.
    """

    def __init__(self):
        self.logger = setup_logging(name="[Retriever]")

    @abstractmethod
    async def retrieve(
        self,
        query_items: List[HybridQueryItem],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[RetrievalResult]:
        """
        Asynchronously executes document retrieval and returns a unified list of RetrievalResult entities.

        Args:
            query_items (List[HybridQueryItem]): Collection of query items, each with 1-to-1 field mapping.
            top_k (int): Maximum number of top context hits to return.
            filters (Optional[Dict[str, Any]]): Generic key-value criteria for metadata filtering.
            **kwargs: Vendor-specific execution options (e.g., routing, index_name override).

        Returns:
            List[RetrievalResult]: Standardized retrieval results.
        """
        pass
