from typing import List

from auraflux_core.core.embeddings.base_embedding import BaseEmbedding


class GenericEmbedding(BaseEmbedding):
    """
    A minimal, fully functional generic embedding implementation.
    Delegates text vectorization tasks directly to the ClientManager.
    """

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate vector embeddings for a batch of text documents.

        Args:
            texts (List[str]): List of document texts to embed.

        Returns:
            List[List[float]]: A list of float vectors representing document embeddings.
        """
        if not texts:
            return []

        self.logger.debug(f"Generating embeddings for {len(texts)} document(s).")
        try:
            embeddings = await self.client_manager.embed(
                provider=self.provider,
                model=self.model,
                input=texts,
                **self.config.parameters
            )
            return embeddings
        except NotImplementedError as e:
            self.logger.error(
                f"Embedding provider '{self.provider}' does not support vector embeddings."
            )
            raise e
        except Exception as e:
            self.logger.error(
                f"Failed to generate document embeddings using model '{self.name}': {e}",
                exc_info=True
            )
            raise e

    async def embed_query(self, text: str) -> List[float]:
        """
        Generate a vector embedding for a single query string.

        Args:
            text (str): Query string to embed.

        Returns:
            List[float]: A float vector representing the query embedding.
        """
        self.logger.debug("Generating embedding for single query string.")
        try:
            embeddings = await self.client_manager.embed(
                provider=self.provider,
                model=self.model,
                input=[text],
                **self.config.parameters
            )
            return embeddings[0]
        except NotImplementedError as e:
            self.logger.error(
                f"Embedding provider '{self.provider}' does not support vector embeddings."
            )
            raise e
        except Exception as e:
            self.logger.error(
                f"Failed to generate query embedding using model '{self.name}': {e}",
                exc_info=True
            )
            raise e
