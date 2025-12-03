from typing import Any, Generator, List

from google import genai
from google.genai import types

from auraflux_core.core.clients.handlers.base_handler import BaseHandler
from auraflux_core.core.schemas.clients import (LLMRequest, LLMResponse,
                                                ModelConfig)


class GeminiHandler(BaseHandler):
    def __init__(self, config: ModelConfig):
        super().__init__(config=config)
        # Configure the Gemini API client
        self.client = genai.Client(api_key=self.config.api_key)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Asynchronously generates a response from the Gemini API.

        This method is non-blocking and is the recommended way to call the API
        in an asynchronous context.
        """
        try:
            # Prepare the request payload for Gemini
            messages_payload = [
                types.Part.from_text(text=msg.content)
                for msg in request.messages
            ]

            # Call the Gemini API asynchronously
            response = await self.client.aio.models.generate_content(
                model=request.model,
                contents=[types.UserContent(parts=messages_payload)],
                config=self._generate_content_config(request)
            )

            # Check for potential errors
            if not response or not response.text:
                raise ValueError("Received an empty or invalid response from the Gemini API.")

            response_text = response.text
            return LLMResponse(text=response_text)

        except Exception as e:
            raise RuntimeError(f"An error occurred while calling the Gemini API: {e}")

    def generate_stream(self, request: LLMRequest) -> Generator[LLMResponse, Any, Any]:
        """
        Generates a streaming response from the Gemini API.
        """
        chat_history: List[types.ContentOrDict] = [
            types.Content(
                role="user" if msg.role == "user" else "model",
                parts=[types.Part.from_text(text=msg.content)]
            )
            for msg in request.messages[:-1]
        ]
        last_message = request.messages[-1].content

        chat_session = self.client.chats.create(
            model=request.model,
            config=self._generate_content_config(request),
            history=chat_history
        )

        for chunk in chat_session.send_message_stream(last_message):
            response_text = chunk.text if chunk.text else ""
            yield LLMResponse(text=response_text)

    def _generate_content_config(self, request: LLMRequest) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=request.system_message,
            max_output_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
