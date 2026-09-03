from typing import Optional

import openai
from openai import AsyncOpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.chat.chat_completion import ChatCompletion
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from auraflux_core.core.clients.handlers.base_handler import BaseHandler
from auraflux_core.core.configs.logging_config import setup_logging
from auraflux_core.core.schemas.clients import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMRequest,
    LLMResponse,
    ProviderConfig,
)


class OpenAIHandler(BaseHandler):
    """
    A concrete handler that adapts the OpenAI API to the BaseHandler interface.
    Provides text generation and text embedding capabilities asynchronously.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.config = config
        self.logger = setup_logging(name=f"[{self.__class__.__name__}:{self.config.id}]")
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.2, min=2, max=60),
        retry=retry_if_exception_type((
            openai.APIConnectionError,
            openai.InternalServerError,
            openai.RateLimitError,
        )),
        reraise=True,
    )
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Asynchronously generates a response from the OpenAI API.
        """
        try:
            messages_payload = []
            if request.system_message:
                messages_payload.append({"role": "system", "content": request.system_message})

            for msg in request.messages:
                messages_payload.append({"role": msg.role, "content": msg.content})

            response: ChatCompletion = await self.client.chat.completions.create(
                model=request.model,
                messages=messages_payload,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )

            if not response.choices:
                raise ValueError("Received an empty response from the OpenAI API.")

            response_text: Optional[str] = response.choices[0].message.content
            if response_text is None:
                raise ValueError("Response text was None.")

            total_tokens = response.usage.total_tokens if response.usage else 0

            return LLMResponse(
                text=response_text,
                token_usage=total_tokens,
            )

        except openai.APIError as e:
            self.logger.error(f"OpenAI API Error during generation: {e}", exc_info=True)
            raise RuntimeError(f"An OpenAI API error occurred: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in OpenAIHandler.generate: {e}", exc_info=True)
            raise RuntimeError(f"An unexpected error occurred in OpenAIHandler: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.2, min=2, max=60),
        retry=retry_if_exception_type((
            openai.APIConnectionError,
            openai.InternalServerError,
            openai.RateLimitError,
        )),
        reraise=True,
    )
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Asynchronously generates vector embeddings using the OpenAI API.
        """
        try:
            response: CreateEmbeddingResponse = await self.client.embeddings.create(
                model=request.model,
                input=request.input,
                **request.parameters,
            )

            if not response.data:
                raise ValueError("Received empty embedding data from the OpenAI API.")

            # Ensure vectors are ordered correctly by index
            sorted_data = sorted(response.data, key=lambda x: x.index)
            embeddings_list = [item.embedding for item in sorted_data]

            token_usage = None
            if response.usage:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return EmbeddingResponse(
                embeddings=embeddings_list,
                token_usage=token_usage,
            )

        except openai.APIError as e:
            self.logger.error(f"OpenAI API Error during embedding: {e}", exc_info=True)
            raise RuntimeError(f"An OpenAI API embedding error occurred: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in OpenAIHandler.embed: {e}", exc_info=True)
            raise RuntimeError(f"An unexpected error occurred in OpenAIHandler embedding: {e}")