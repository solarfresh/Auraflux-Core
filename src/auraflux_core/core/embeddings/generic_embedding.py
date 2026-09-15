from typing import List

from auraflux_core.core.embeddings.base_embedding import BaseEmbedding


class GenericEmbedding(BaseEmbedding):
    """
    A minimal, fully functional generic embedding implementation.
    Delegates text vectorization tasks directly to the ClientManager.
    """

    async def _embed_documents_impl(self, texts: List[str]) -> List[List[float]]:
        """
        Internal implementation to generate vector embeddings for a batch of text documents.
        Logging and error handling are managed by the BaseEmbedding parent class.
        """
        if not texts:
            return []

        embeddings = await self.client_manager.embed(
            provider=self.provider,
            model=self.model,
            input=texts,
            **self.config.parameters
        )
        return embeddings

    async def _embed_query_impl(self, text: str) -> List[float]:
        """
        Internal implementation to generate a vector embedding for a single query string.
        Logging and error handling are managed by the BaseEmbedding parent class.
        """
        embeddings = await self.client_manager.embed(
            provider=self.provider,
            model=self.model,
            input=[text],
            **self.config.parameters
        )
        return embeddings[0]
