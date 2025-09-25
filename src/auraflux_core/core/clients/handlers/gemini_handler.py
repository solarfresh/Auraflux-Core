from google import genai
from google.genai import types

from twinrad.core.clients.handlers.base_handler import BaseHandler
from twinrad.core.schemas.clients import LLMRequest, LLMResponse, ModelConfig


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

            # Extract generation configuration
            generation_config = types.GenerateContentConfig(
                system_instruction=request.system_message,
                max_output_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )

            # Call the Gemini API asynchronously
            response = await self.client.aio.models.generate_content(
                model=request.model,
                contents=[types.UserContent(parts=messages_payload)],
                config=generation_config
            )

            # Check for potential errors
            if not response or not response.text:
                raise ValueError("Received an empty or invalid response from the Gemini API.")

            response_text = response.text
            return LLMResponse(text=response_text)

        except Exception as e:
            raise RuntimeError(f"An error occurred while calling the Gemini API: {e}")