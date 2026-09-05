from typing import Any, Generator, List, Optional

from google import genai
from google.genai import errors, types
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
from auraflux_core.core.tools.base_tool import ToolSpecConverter


class GeminiHandler(BaseHandler):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.config = config
        self.logger = setup_logging(name=f"[{self.__class__.__name__}:{self.config.id}]")
        # Configure the Google GenAI SDK client
        self.client = genai.Client(api_key=self.config.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.2, min=30, max=300),
        retry=retry_if_exception_type(errors.ServerError),
        reraise=True,
    )
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Asynchronously generates a text response or tool calls using the Gemini API.
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
                config=self._generate_content_config(request),
            )

            # Check for potential empty responses
            if not response or not response.text:
                raise ValueError("Received an empty or invalid response from the Gemini API.")

            response_text = response.text
            usage_metadata = response.usage_metadata

            # Extract tool calls safely
            function_call = None
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts and len(candidate.content.parts) > 0:
                    part = candidate.content.parts[0]
                    if hasattr(part, 'function_call') and part.function_call:
                        function_call = part.function_call

            tool_calls = (
                {'tool': function_call.name, 'args': function_call.args}
                if function_call is not None
                else None
            )

            total_tokens = getattr(usage_metadata, 'total_token_count', 0) if usage_metadata else 0

            return LLMResponse(
                text=response_text,
                token_usage=total_tokens,
                tool_calls=tool_calls,
            )
        except errors.ServerError as e:
            self.logger.warning(f"Gemini Server Error encountered, retrying... Details: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"An error occurred while calling the Gemini API: {e}", exc_info=True)
            raise RuntimeError(f"An error occurred while calling the Gemini API: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.2, min=30, max=300),
        retry=retry_if_exception_type(errors.ServerError),
        reraise=True,
    )
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Asynchronously generates vector embeddings using the Gemini API.
        """
        try:
            # Call Gemini embedding API asynchronously
            res = await self.client.aio.models.embed_content(
                model=request.model,
                contents=request.input,
                config=types.EmbedContentConfig(**request.parameters)
                if request.parameters
                else None,
            )

            embeddings_list: List[List[float]] = []
            if hasattr(res, 'embeddings') and res.embeddings:
                embeddings_list = [e.values for e in res.embeddings if e.values is not None]
            else:
                raise ValueError("No embeddings found in the Gemini API response.")

            return EmbeddingResponse(
                embeddings=embeddings_list,
                token_usage=None,
            )
        except errors.ServerError as e:
            self.logger.warning(f"Gemini Server Error during embedding, retrying... Details: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"An error occurred while embedding with Gemini API: {e}", exc_info=True)
            raise RuntimeError(f"An error occurred while embedding with Gemini API: {e}")

    def generate_stream(self, request: LLMRequest) -> Generator[LLMResponse, Any, Any]:
        """
        Generates a streaming response from the Gemini API.
        """
        chat_history: List[types.ContentOrDict] = [
            types.Content(
                role="user" if msg.role == "user" else "model",
                parts=[types.Part.from_text(text=msg.content)],
            )
            for msg in request.messages[:-1]
        ]
        last_message = request.messages[-1].content

        chat_session = self.client.chats.create(
            model=request.model,
            config=self._generate_content_config(request),
            history=chat_history,
        )

        for chunk in chat_session.send_message_stream(last_message):
            response_text = chunk.text if chunk.text else ""
            yield LLMResponse(text=response_text)

    def get_available_models(self):
        """
        Retrieves the list of available models from Google Gemini API.
        """
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
            "models": supported_models,
        }

    def _generate_content_config(self, request: LLMRequest) -> types.GenerateContentConfig:
        """
        Helper method to map internal LLMRequest parameters to Gemini's GenerateContentConfig.
        """
        tools = None
        tool_config = None
        if request.tools is not None:
            tools = [
                types.Tool(
                    function_declarations=[
                        ToolSpecConverter.to_gemini(tool) for tool in request.tools
                    ]
                )
            ]
            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.AUTO
                )
            )

        thinking_config = None
        if request.thinking_level is not None:
            thinking_config = types.ThinkingConfig(
                thinking_level=getattr(types.ThinkingLevel, request.thinking_level.upper())
                if request.thinking_level
                else None
            )

        return types.GenerateContentConfig(
            system_instruction=request.system_message,
            max_output_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            thinking_config=thinking_config,
            tools=tools,
            tool_config=tool_config,
        )