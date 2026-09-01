from abc import ABC, abstractmethod
from typing import List

from auraflux_core.core.clients.client_manager import ClientManager
from auraflux_core.core.configs.logging_config import setup_logging
from auraflux_core.core.schemas.embeddings import EmbeddingConfig


class BaseEmbedding(ABC):
    """
    Base abstract class for all embedding models in the Auraflux system.

    Provides shared initialization, logging, and property accessors, ensuring a consistent
    interface for embedding generation across different providers and model families.
    """

    def __init__(self, config: EmbeddingConfig, client_manager: ClientManager):
        self.config = config
        self.client_manager = client_manager
        self.logger = setup_logging(name=f"[Embedding:{self.config.name}]")
        self.logger.info(f"Embedding model '{self.config.name}' initialized.")

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Abstract method to generate vector embeddings for a batch of text documents.
        """
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """
        Abstract method to generate a vector embedding for a single search query string.
        """
        pass
