import asyncio
from asyncio import Future, Queue, Task
from typing import Dict

from auraflux_core.core.clients.handlers.base_handler import BaseHandler
from auraflux_core.core.clients.handlers.gemini_handler import GeminiHandler
from auraflux_core.core.clients.handlers.openai_handler import OpenAIHandler
from auraflux_core.core.configs.logging_config import setup_logging
from auraflux_core.core.schemas.clients import (ClientConfig, LLMRequest,
                                                LLMResponse)

try:
    from auraflux_core.core.clients.handlers.vllm_handler import VLLMHandler
    using_vllm_handler = True
except ImportError:
    using_vllm_handler = False


class ClientManager:
    """
    Manages all LLM handlers and routes requests from agents.
    It acts as a single point of entry and enforces access permissions.
    """

    def __init__(self, config: ClientConfig):
        self.config = config
        self.logger = setup_logging(name=f"[{self.__class__.__name__}]")
        self.handlers: Dict[str, BaseHandler] = {}
        self.request_queue: Queue = Queue()
        self.dispatcher_task: Task | None = None

    async def initialize(self):
        """Asynchronously initializes handlers and the dispatcher."""
        await self._instantiate_handlers()
        self._start_dispatcher()

    async def _instantiate_handlers(self):
        """
        Instantiates all necessary LLM handlers based on the provided configuration.
        """
        for model_config in self.config.models:
            api_key = model_config.api_key
            if not api_key and model_config.mode not in ('vllm',):
                raise ValueError(f"API key for model '{model_config.name}' is not provided.")

            # vLLM is an optional dependency
            if using_vllm_handler and model_config.mode == "vllm":
                handler_instance = VLLMHandler(config=model_config)
                await handler_instance.ainit()

            if model_config.mode == "gemini":
                handler_instance = GeminiHandler(config=model_config)

            if model_config.mode == "openai":
                handler_instance = OpenAIHandler(config=model_config)
            # Add other handlers here as they are implemented

            if handler_instance:
                self.handlers[model_config.name] = handler_instance

    async def _dispatch_requests(self):
        """Dispatches requests from the queue to the correct handler."""
        while True:
            # Wait for a new request
            request, future = await self.request_queue.get()

            try:
                # Find the correct handler and process the request
                handler = self.handlers.get(request.model)
                if handler:
                    response = await handler.generate(request)
                    future.set_result(response)
                    self.logger.info(f"Dispatched request for model {request.model}.")
                else:
                    error_msg = f"Handler for model '{request.model}' not found. Check your configuration."
                    self.logger.error(error_msg)
                    future.set_exception(ValueError(error_msg))
            except Exception as e:
                self.logger.error(f"Error processing request: {e}")
                future.set_exception(RuntimeError(f"Error processing request for model {request.model}: {e}"))
            finally:
                self.request_queue.task_done()

    def _start_dispatcher(self):
        """Starts a single background task to dispatch requests."""
        if self.dispatcher_task is None or self.dispatcher_task.done():
            self.dispatcher_task = asyncio.create_task(self._dispatch_requests())
            self.logger.info("ClientManager dispatcher started.")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Submits a request to the queue and waits for the response.

        This method acts as a producer, adding the request to the queue and
        then awaiting a Future object which will be completed by the dispatcher.
        """
        # Create a Future to hold the result of the request
        future = Future()

        # Put the request and the future object into the queue
        await self.request_queue.put((request, future))

        # Wait for the result to be set by the dispatcher
        return await future

    async def shutdown(self):
        """
        Gracefully shuts down the client manager, waiting for pending requests to finish.
        """
        if self.dispatcher_task and not self.dispatcher_task.done():
            self.logger.warning("Stopping ClientManager dispatcher...")
            await self.request_queue.join()
            self.dispatcher_task.cancel()
            try:
                await self.dispatcher_task
            except asyncio.CancelledError:
                self.logger.info("ClientManager dispatcher stopped.")