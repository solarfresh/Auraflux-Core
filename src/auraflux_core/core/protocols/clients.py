from typing import Any, Generator, List, Protocol, TypeVar

from auraflux_core.core.schemas.clients import (EmbeddingRequest,
                                                EmbeddingResponse, LLMRequest,
                                                LLMResponse)

T_LLM_Request = TypeVar("T_LLM_Request", bound=LLMRequest)
T_LLM_Response = TypeVar("T_LLM_Response", bound=LLMResponse)

class ClientManagerProtocol(Protocol):
    async def generate(self, request: LLMRequest) -> LLMResponse:
        ...

    async def embed(self, provider: str, model: str, input: List[str], **kwargs: Any) -> List[List[float]]:
        ...

    def generate_stream(self, request: LLMRequest) -> Generator[LLMResponse, Any, Any]:
        ...
