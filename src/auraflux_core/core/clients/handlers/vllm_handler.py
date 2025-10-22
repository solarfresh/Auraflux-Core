from uuid import uuid4

from transformers import AutoTokenizer
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

from twinrad.core.clients.handlers.base_handler import BaseHandler
from twinrad.core.schemas.clients import LLMRequest, LLMResponse, ModelConfig


class VLLMHandler(BaseHandler):
    """
    A concrete handler that manages a single, centralized vLLM engine
    to prevent GPU resource contention.
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """
        TODO: if there are multiple models, this needs to be updated.
        Implements the Singleton pattern.
        """
        if cls._instance is None:
            cls._instance = super(VLLMHandler, cls).__new__(cls)

        return cls._instance

    def __init__(self, config: ModelConfig):
        """Initializes the VLLM engine only once."""
        super().__init__(config=config)
        if self._initialized:
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.name)
        self.engine = None
        self._initialized = True

    async def ainit(self):
        """
        Asynchronously loads the vLLM engine. This method should be awaited
        at application startup to prevent blocking.
        """
        if self.engine is None:
            engine_args = AsyncEngineArgs(
                model=self.config.name,
                max_model_len=self.config.max_model_len,
                tensor_parallel_size=self.config.tensor_parallel_size,
                device=self.config.device,
                dtype=self.config.dtype
            )
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generates a response using the vLLM engine with a concurrency lock.
        """
        if self.engine is None:
            raise RuntimeError("VLLM engine is not initialized. Call ainit() first.")

        try:
            prompt = self._apply_chat_template(request)

            sampling_params = SamplingParams(
                n=1,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
            )

            request_id = str(uuid4())
            self.logger.debug(f"vLLM request ID: {request_id}, Prompt: {prompt}")
            results_generator = self.engine.generate(
                prompt, sampling_params, request_id
            )
            self.logger.debug(f"vLLM results generator created for request ID: {request_id}")

            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            self.logger.debug(f"vLLM final output for request ID {request_id}: {final_output}")

            if not final_output:
                raise ValueError("Received no output from vLLM engine.")

            output_text = ''
            for output in final_output.outputs:
                self.logger.debug(f"vLLM output: {output}")
                if output.text:
                    output_text += output.text

            self.logger.debug(f"vLLM output text for request ID {request_id}: {output_text}")
            if not output_text:
                raise ValueError("Received empty output text from vLLM engine.")

            return LLMResponse(text=output_text.strip())

        except Exception as e:
            raise RuntimeError(f"An error occurred in VLLMHandler: {e}")

    def _apply_chat_template(self, request: LLMRequest) -> str:
        """
        Combines a list of messages into a single prompt string for vLLM.
        This method should be customized to match the specific model's chat template.
        """

        messages_payload = []
        if request.system_message:
            if self.config.restrict_user_assistant_alternate:
                role = 'user'
            else:
                role = 'system'

            messages_payload.append({"role": role, "content": request.system_message})

        messages_size = len(request.messages)
        last_message = request.messages[-1]
        for index, msg in enumerate(request.messages):
            messages_payload.append({"role": role, "content": msg.content})

        if self.config.restrict_user_assistant_alternate:
            if messages_payload[0]['role'] == 'assistant' or messages_payload[0]['role'] == messages_payload[1]['role']:
                messages_payload[1]['content'] = messages_payload[0]['content'] + '\n\n' + messages_payload[1]['content']
                messages_payload = messages_payload[1:]

        return self.tokenizer.apply_chat_template(messages_payload, tokenize=False, add_generation_prompt=True)
