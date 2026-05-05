from typing import Any, Generator, List

from google import genai
from google.genai import types

from auraflux_core.core.clients.handlers.base_handler import BaseHandler
from auraflux_core.core.schemas.clients import (LLMRequest, LLMResponse,
                                                ProviderConfig)
from auraflux_core.core.tools.base_tool import ToolSpecConverter


class GeminiHandler(BaseHandler):
    def __init__(self, config: ProviderConfig):
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
            usage_metadata = response.usage_metadata
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts and len(candidate.content.parts) > 0:
                    part = candidate.content.parts[0]
                    if hasattr(part, 'function_call'):
                        function_call = part.function_call

            tool_calls = {'tool': function_call.name , 'args': function_call.args} if function_call is not None else None

            return LLMResponse(text=response_text, token_usage=getattr(usage_metadata, 'total_token_count', 0), tool_calls=tool_calls)

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

    def get_available_models(self):
        models = self.client.models.list(config={'page_size': 50})

        supported_models = []
        for m in models:
            supported_models.append({
                "name": m.name,
                "display_name": m.display_name,
                "description": m.description,
                "input_token_limit": m.input_token_limit,
                "output_token_limit": m.output_token_limit,
            })

        return {
            "status": "SUCCESS",
            "count": len(supported_models),
            "models": supported_models
        }

    def _generate_content_config(self, request: LLMRequest) -> types.GenerateContentConfig:
        tools = []
        if request.tools is not None:
            tools = [
                types.Tool(function_declarations=[
                    ToolSpecConverter.to_gemini(tool) for tool in request.tools
                ])
            ]

        thinking_config = None
        if request.thinking_level is not None:
            thinking_config = types.ThinkingConfig(thinking_level=getattr(types.ThinkingLevel, request.thinking_level.upper()) if request.thinking_level else None)

        return types.GenerateContentConfig(
            system_instruction=request.system_message,
            max_output_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            thinking_config=thinking_config,
            tools=tools,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.AUTO)
            ),
        )
