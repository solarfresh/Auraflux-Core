import asyncio
import threading
import time
from asyncio import Future, Queue, Task
from concurrent.futures import Future as ThreadFuture
from typing import Any, Dict, Generator, List

from auraflux_core.core.clients.handlers.base_handler import BaseHandler
from auraflux_core.core.clients.handlers.gemini_handler import GeminiHandler
from auraflux_core.core.clients.handlers.openai_handler import OpenAIHandler
from auraflux_core.core.configs.logging_config import setup_logging
from auraflux_core.core.schemas.clients import (
    ClientConfig,
    EmbeddingRequest,
    EmbeddingResponse,
    LLMRequest,
    LLMResponse,
)


class ClientManager:
    """
    Manages all LLM and Embedding handlers, routing requests from agents and embedding instances.
    It acts as a single point of entry and enforces access permissions.
    """

    def __init__(self, config: ClientConfig):
        self.config = config
        self.logger = setup_logging(name=f"[{self.__class__.__name__}]")
        self.handlers: Dict[str, BaseHandler] = {}
        self.request_queue: Queue = Queue()
        self.dispatcher_task: Task | None = None
        self.initialize_mode = config.initialize_mode
        self.dispatcher_thread: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.dispatcher_task_future: ThreadFuture | None = None

    async def initialize(self):
        """Asynchronously initializes handlers and the dispatcher."""
        await self.instantiate_handlers()
        if self.initialize_mode == 'create_task':
            self._start_dispatcher()
        elif self.initialize_mode == 'run_forever':
            self.start_dispatcher_thread()
        else:
            raise ValueError(f"Invalid initialize_mode: {self.initialize_mode}")

    def get_available_models(self, provider_id: str):
        return self.handlers[provider_id].get_available_models()

    async def instantiate_handlers(self):
        """Instantiates all necessary LLM/Embedding handlers based on configuration."""
        for provider_config in self.config.providers:
            self.instantiate_handler_by_config(provider_config)

    def instantiate_handler_by_config(self, provider_config):
        api_key = provider_config.api_key
        if not api_key and provider_config.type not in ('vllm',):
            raise ValueError(f"API key for provider '{provider_config.id}' is not provided.")

        handler_instance = None
        if provider_config.type == "GOOGLE":
            handler_instance = GeminiHandler(config=provider_config)
        elif provider_config.type == "OPENAI":
            handler_instance = OpenAIHandler(config=provider_config)

        if handler_instance:
            self.handlers[provider_config.id] = handler_instance

    async def _dispatch_requests(self):
        """Dispatches both LLM and Embedding requests from the queue to the correct handler."""
        while True:
            try:
                (request, future) = await self.request_queue.get()
            except asyncio.CancelledError:
                self.logger.warning("Dispatcher received cancellation while waiting for queue item.")
                break
            except Exception as e:
                self.logger.critical(f"FATAL: Dispatcher queue retrieval failed: {e}", exc_info=True)
                await asyncio.sleep(1)
                continue

            response = None
            error_to_set = None

            try:
                handler = self.handlers.get(request.provider)

                if handler:
                    if isinstance(request, LLMRequest):
                        response = await handler.generate(request)
                    elif isinstance(request, EmbeddingRequest):
                        response = await handler.embed(request)
                    else:
                        raise TypeError(f"Unsupported request type: {type(request)}")

                    self.logger.debug(f"[{request.provider}] Handler response received.")
                else:
                    error_msg = f"Handler for provider '{request.provider}' not found. Check configuration."
                    self.logger.error(error_msg)
                    error_to_set = RuntimeError(error_msg)

            except NotImplementedError as e:
                self.logger.error(f"[{request.provider}] Feature not supported by handler: {e}")
                error_to_set = e

            except Exception as e:
                self.logger.error(f"[{request.provider}] Error processing request: {e}", exc_info=True)
                error_to_set = e

            # Safely set result or exception on Future
            try:
                if error_to_set:
                    if not future.done():
                        future.set_exception(error_to_set)
                elif response is not None:
                    if not future.done():
                        future.set_result(response)
                        self.logger.info(f"[{request.provider}] Dispatched request completed.")
            except asyncio.InvalidStateError:
                self.logger.warning(f"[{request.provider}] Future was already completed or cancelled.")
            except Exception as e:
                self.logger.critical(f"[{request.provider}] CRITICAL FAILURE setting Future result: {e}", exc_info=True)

            # Mark queue task done
            try:
                self.request_queue.task_done()
                self.logger.debug(f"[{request.provider}] Queue task_done() called.")
            except Exception as e:
                self.logger.critical(f"FATAL: Failed to call task_done() for {request.provider}: {e}")

    def _start_dispatcher(self):
        self.initialize_mode = 'create_task'
        if self.dispatcher_task is None or self.dispatcher_task.done():
            self.dispatcher_task = asyncio.create_task(self._dispatch_requests())
            self.logger.info("ClientManager dispatcher started.")

    def start_dispatcher_thread(self):
        self.initialize_mode = 'run_forever'
        if self.dispatcher_thread is None:
            self.loop = asyncio.new_event_loop()
            self.dispatcher_thread = threading.Thread(
                target=self._run_loop_forever, args=(self.loop,), daemon=True
            )
            self.dispatcher_thread.start()
            self.dispatcher_task_future = asyncio.run_coroutine_threadsafe(self._dispatch_requests(), self.loop)
            self.logger.info("ClientManager dispatcher thread started successfully.")

    def _run_loop_forever(self, loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def submit_to_queue(self, request, future):
        worker_tid = threading.get_ident()
        if self.loop is None:
            raise RuntimeError("Event loop for ClientManager is not initialized.")

        try:
            self.request_queue.put_nowait((request, future))
        except Exception as e:
            self.logger.error(f"[{request.provider}][TID:{worker_tid}] Failure inside submit_to_queue: {e}")
            self.loop.call_soon(future.set_exception, RuntimeError(f"Queue put failed internally: {e}"))

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Submits a LLM generation request to the queue and waits for response."""
        return await self._submit_and_await(request)

    async def embed(self, provider: str, model: str, input: List[str], **kwargs) -> List[List[float]]:
        """
        Submits an embedding request to the queue and returns vector outputs directly.
        """
        request = EmbeddingRequest(
            provider=provider,
            model=model,
            input=input,
            parameters=kwargs
        )
        response: EmbeddingResponse = await self._submit_and_await(request)
        return response.embeddings

    async def _submit_and_await(self, request: LLMRequest | EmbeddingRequest) -> Any:
        """Generic internal method to handle queue submission and synchronous/asynchronous polling."""
        worker_tid = threading.get_ident()
        self.logger.info(f"[{request.provider}] Request called. Worker TID: {worker_tid}")
        future = Future()

        if self.initialize_mode == 'create_task':
            self.logger.debug(f"Submitting request for provider {request.provider} to queue.")
            await self.request_queue.put((request, future))
        elif self.initialize_mode == 'run_forever':
            if self.loop is None:
                raise RuntimeError("Event loop for ClientManager is not initialized.")
            self.loop.call_soon_threadsafe(self.submit_to_queue, request, future)
            self.logger.info(f"Request for provider {request.provider} submitted to background loop.")
        else:
            raise RuntimeError("ClientManager is not properly initialized.")

        start_time = time.time()
        if self.loop is None and self.initialize_mode == 'run_forever':
            raise RuntimeError("Event loop for ClientManager is not initialized.")

        try:
            while not future.done():
                if (time.time() - start_time) > self.config.timeout_seconds:
                    if self.loop:
                        self.loop.call_soon_threadsafe(future.cancel)
                    else:
                        future.cancel()

                    raise TimeoutError(
                        f"Async request timed out after {self.config.timeout_seconds} seconds."
                    )
                time.sleep(self.config.sleep_interval_seconds)

            response = await future
            return response
        except Exception as e:
            self.logger.error(f"[{request.provider}] Error awaiting response: {e}", exc_info=True)
            raise e

    def generate_stream(self, request: LLMRequest) -> Generator[LLMResponse, Any, Any]:
        """Generates a streaming response from the appropriate handler."""
        handler = self.handlers.get(request.provider)
        if handler and hasattr(handler, 'generate_stream'):
            stream_generator = handler.generate_stream(request)
            for response in stream_generator:
                yield response
        else:
            error_msg = f"Streaming not supported for provider '{request.provider}' or handler not found."
            self.logger.error(error_msg)
            raise NotImplementedError(error_msg)

    async def shutdown(self):
        """Gracefully shuts down the client manager."""
        if self.dispatcher_task and not self.dispatcher_task.done():
            self.logger.warning("Stopping ClientManager dispatcher...")
            await self.request_queue.join()
            self.dispatcher_task.cancel()
            try:
                await self.dispatcher_task
            except asyncio.CancelledError:
                self.logger.info("ClientManager dispatcher stopped.")

        if self.loop and self.loop.is_running():
            self.logger.warning("Initiating graceful shutdown...")
            await self.request_queue.join()

            if self.dispatcher_task_future and not self.dispatcher_task_future.done():
                self.logger.warning("Stopping background event loop.")
                self.loop.call_soon_threadsafe(self.loop.stop)

            if self.dispatcher_thread and self.dispatcher_thread.is_alive():
                self.dispatcher_thread.join(timeout=5)
