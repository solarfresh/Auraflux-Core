import time
from abc import ABC, abstractmethod
from typing import List

from auraflux_core.core.clients.client_manager import ClientManager
from auraflux_core.core.configs.logging_config import get_logger
from auraflux_core.core.schemas.embeddings import EmbeddingConfig

logger = get_logger(__name__)


class BaseEmbedding(ABC):
    """
    Base abstract class for all embedding models in the Auraflux system.

    Provides shared initialization, logging, and property accessors, ensuring a consistent
    interface for embedding generation across different providers and model families.
    """

    def __init__(self, config: EmbeddingConfig, client_manager: ClientManager):
        self.config = config
        self.client_manager = client_manager

        logger.info(
            "embedding_model_initialized",
            embedding_name=self.name,
            provider=self.provider,
            model=self.model,
        )

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
    async def _embed_documents_impl(self, texts: List[str]) -> List[List[float]]:
        """
        Subclasses must implement the actual embedding API call for documents.
        """
        pass

    @abstractmethod
    async def _embed_query_impl(self, text: str) -> List[float]:
        """
        Subclasses must implement the actual embedding API call for a single query.
        """
        pass

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates vector embeddings for a batch of text documents with structured tracing.
        """
        start_time = time.perf_counter()
        input_count = len(texts)

        logger.debug(
            "embed_documents_started",
            embedding_name=self.name,
            provider=self.provider,
            model=self.model,
            input_count=input_count,
        )

        try:
            embeddings = await self._embed_documents_impl(texts)
            latency_sec = time.perf_counter() - start_time

            logger.info(
                "embed_documents_completed",
                embedding_name=self.name,
                provider=self.provider,
                model=self.model,
                input_count=input_count,
                latency_sec=round(latency_sec, 4),
                status="success",
            )
            return embeddings
        except Exception as e:
            latency_sec = time.perf_counter() - start_time
            logger.error(
                "embed_documents_failed",
                embedding_name=self.name,
                provider=self.provider,
                model=self.model,
                input_count=input_count,
                error_type=type(e).__name__,
                error_msg=str(e),
                latency_sec=round(latency_sec, 4),
                exc_info=True,
            )
            raise e

    async def embed_query(self, text: str) -> List[float]:
        """
        Generates a vector embedding for a single search query string with structured tracing.
        """
        start_time = time.perf_counter()

        logger.debug(
            "embed_query_started",
            embedding_name=self.name,
            provider=self.provider,
            model=self.model,
        )

        try:
            embedding = await self._embed_query_impl(text)
            latency_sec = time.perf_counter() - start_time

            logger.info(
                "embed_query_completed",
                embedding_name=self.name,
                provider=self.provider,
                model=self.model,
                latency_sec=round(latency_sec, 4),
                status="success",
            )
            return embedding
        except Exception as e:
            latency_sec = time.perf_counter() - start_time
            logger.error(
                "embed_query_failed",
                embedding_name=self.name,
                provider=self.provider,
                model=self.model,
                error_type=type(e).__name__,
                error_msg=str(e),
                latency_sec=round(latency_sec, 4),
                exc_info=True,
            )
            raise e