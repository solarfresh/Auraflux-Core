from abc import ABC, abstractmethod
from typing import Any

from auraflux_core.core.configs.logging_config import setup_logging
from auraflux_core.core.schemas.clients import (EmbeddingRequest,
                                                EmbeddingResponse, LLMRequest,
                                                LLMResponse, ProviderConfig)


class BaseHandler(ABC):
    """
    An abstract base class that defines the common interface for all
    language model handlers.
    """

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.logger = setup_logging(name=f"[{self.__class__.__name__}]")

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Asynchronously generates a response from a language model.
        """
        pass

    def generate_stream(self, request: LLMRequest) -> Any:
        """
        Generates a streaming response from a language model.
        """
        pass

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Optional method to process vector embedding requests.

        Concrete handlers that support embedding operations should override this method.
        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            f"Provider handler '{self.__class__.__name__}' does not support embedding operations."
        )

    def get_available_models(self):
        pass
