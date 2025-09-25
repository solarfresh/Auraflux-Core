from typing import Optional

import openai
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

from twinrad.core.clients.handlers.base_handler import BaseHandler
from twinrad.core.schemas.clients import LLMRequest, LLMResponse, ModelConfig


class OpenAIHandler(BaseHandler):
    """
    A concrete handler that adapts the OpenAI API to the BaseHandler interface.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config=config)
        self.config = config
        # The OpenAI client automatically reads from the OPENAI_API_KEY env var
        # but we can explicitly pass it from the config.
        self.client = AsyncOpenAI(api_key=self.config.api_key, base_url=self.config.base_url)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Asynchronously generates a response from the OpenAI API.
        This is the recommended non-blocking method for modern applications.
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

            return LLMResponse(text=response_text)

        except openai.APIError as e:
            raise RuntimeError(f"An OpenAI API error occurred: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred in OpenAIHandler: {e}")