from abc import ABC, abstractmethod

from twinrad.core.configs.logging_config import setup_logging
from twinrad.core.schemas.clients import LLMRequest, LLMResponse, ModelConfig


class BaseHandler(ABC):
    """
    An abstract base class that defines the common interface for all
    language model handlers.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = setup_logging(name=f"[{self.__class__.__name__}]")

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Asynchronously generates a response from a language model.
        """
        pass