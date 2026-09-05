from typing import Any, Dict, Optional

from auraflux_core.core.schemas.tools import ToolConfig
from auraflux_core.core.tools.base_tool import BaseTool
from auraflux_core.rag.schemas.retrievers import HybridRetrieverInput


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
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Executes hybrid retrieval through the injected retriever engine.
        Implements BaseTool.run() abstract method.
        """
        self.logger.info(f"Executing hybrid retrieval for query: '{query_text}' with top_k={top_k}")

        results = await self.retriever.retrieve(
            query_text=query_text,
            top_k=top_k,
            filters=filters
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
