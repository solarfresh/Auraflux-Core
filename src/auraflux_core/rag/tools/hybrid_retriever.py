from typing import Any, Dict, List, Optional

from auraflux_core.core.schemas.tools import ToolConfig
from auraflux_core.core.tools.base_tool import BaseTool
from auraflux_core.rag.schemas.retrievers import (HybridQueryItem,
                                                  HybridRetrieverInput)


class HybridRetrieverTool(BaseTool):
    """
    Generic vendor-agnostic tool providing hybrid search capabilities
    across vector and lexical search engines.
    """

    def __init__(
        self,
        retriever: Any,
        config: ToolConfig = ToolConfig()
    ) -> None:
        super().__init__(config=config)
        self.retriever = retriever

    async def run(
        self,
        query_items: List[Dict[str, Any]],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Executes hybrid retrieval through the injected retriever engine.
        Implements BaseTool.run() abstract method.
        """
        self.logger.info(f"Executing hybrid retrieval with {len(query_items)} query items, top_k={top_k}, filters={filters}")

        items: List[HybridQueryItem] = [
            HybridQueryItem(
                query_text=raw_item.get("query_text", ""),
                query_vector=None,
                text_field=raw_item.get("text_field", "text"),
                vector_field=raw_item.get("vector_field", "vector")
            )
            for raw_item in query_items
        ]

        if not items:
            self.logger.warning("No valid query items provided to hybrid_retriever.")
            return []

        results = await self.retriever.retrieve(
            query_items=items,
            top_k=top_k,
            filters=filters,
            **kwargs
        )
        return [result.model_dump() for result in results]

    def get_name(self) -> str:
        """Returns the unique tool identifier for LLM tool invocation."""
        return "hybrid_retriever"

    def get_description(self) -> str:
        """Returns the function description used by LLMs to determine tool routing."""
        return (
            "Retrieves relevant context chunks or evidence fragments "
            "using hybrid (dense vector + lexical keyword) search strategy."
        )

    def get_parameters(self) -> Dict[str, Any]:
        """
        Generates and returns the OpenAPI/JSON Schema parameter specs.
        Leverages Pydantic model's model_json_schema() from HybridRetrieverInput.
        """
        return HybridRetrieverInput.model_json_schema()
