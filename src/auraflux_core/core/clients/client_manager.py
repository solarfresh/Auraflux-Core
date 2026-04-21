import asyncio
import threading
from asyncio import Future, Queue, Task
from concurrent.futures import Future as ThreadFuture
from typing import Dict, Generator, Any
import time

from auraflux_core.core.clients.handlers.base_handler import BaseHandler
from auraflux_core.core.clients.handlers.gemini_handler import GeminiHandler
from auraflux_core.core.clients.handlers.openai_handler import OpenAIHandler
from auraflux_core.core.configs.logging_config import setup_logging
from auraflux_core.core.schemas.clients import (ClientConfig, LLMRequest,
                                                LLMResponse)

# try:
#     from auraflux_core.core.clients.handlers.vllm_handler import VLLMHandler
#     using_vllm_handler = True
# except ImportError:
#     using_vllm_handler = False


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

    def get_available_models(self, provider_id):
        return self.handlers[provider_id].get_available_models()

    async def instantiate_handlers(self):
        """
        Instantiates all necessary LLM handlers based on the provided configuration.
        """
        for provider_config in self.config.providers:
            api_key = provider_config.api_key
            if not api_key and provider_config.type not in ('vllm',):
                raise ValueError(f"API key for provider '{provider_config.id}' is not provided.")

            # vLLM is an optional dependency
            # if using_vllm_handler and model_config.provider_type == "VLLM":
            #     handler_instance = VLLMHandler(config=model_config)
            #     await handler_instance.ainit()

            if provider_config.type == "GOOGLE":
                handler_instance = GeminiHandler(config=provider_config)

            if provider_config.type == "OPENAI":
                handler_instance = OpenAIHandler(config=provider_config)
            # Add other handlers here as they are implemented

            if handler_instance:
                self.handlers[provider_config.id] = handler_instance

    async def _dispatch_requests(self):
        """Dispatches requests from the queue to the correct handler."""
        while True:
            # 1. Wait for a new request (this is outside the inner try/except block)
            try:
                (request, future) = await self.request_queue.get()
            except asyncio.CancelledError:
                self.logger.warning("Dispatcher received cancellation while waiting for queue item.")
                break # Exit loop cleanly
            except Exception as e:
                self.logger.critical(f"FATAL: Dispatcher queue retrieval failed: {e}", exc_info=True)
                await asyncio.sleep(1) # Prevent busy loop on catastrophic failure
                continue

            # 2. Process the request inside an inner try block
            response = None
            error_to_set = None

            try:
                handler = self.handlers.get(request.provider)

                if handler:
                    response = await handler.generate(request)
                    self.logger.debug(f"[{request.provider}] Handler response: {response}")
                else:
                    error_msg = f"Handler for provider '{request.provider}' not found. Check configuration."
                    self.logger.error(error_msg)
                    error_to_set = LLMResponse(text=error_msg)

            except Exception as e:
                self.logger.error(f"[{request.provider}] Error processing request: {e}", exc_info=True)
                error_to_set = LLMResponse(text=f"Error processing request for provider {request.provider}: {e}")

            # 3. CRITICAL: Safely set the result or exception on the Future
            try:
                if error_to_set:
                    # CRITICAL: Always check if the future is already done before setting an exception
                    if not future.done():
                        future.set_exception(error_to_set)
                elif response is not None:
                    if not future.done():
                        future.set_result(response)
                        self.logger.info(f"[{request.provider}] Dispatched request completed.")
            except asyncio.InvalidStateError:
                self.logger.warning(f"[{request.provider}] Future was already completed (possibly cancelled by generator).")
            except Exception as e:
                # If setting the result/exception fails, log but proceed to task_done
                self.logger.critical(f"[{request.provider}] CRITICAL FAILURE setting Future result/exception: {e}", exc_info=True)

            # 4. FINAL STEP: Mark task as done in the queue
            # This must be the last thing, and is the reason for the inner structure.
            try:
                self.request_queue.task_done()
                self.logger.debug(f"[{request.provider}] Queue task_done() called.") # FIX: Add confirmation log
            except Exception as e:
                self.logger.critical(f"FATAL: Failed to call task_done() for {request.provider}: {e}")

    def _start_dispatcher(self):
        """Starts a single background task to dispatch requests."""
        self.initialize_mode = 'create_task'
        if self.dispatcher_task is None or self.dispatcher_task.done():
            self.dispatcher_task = asyncio.create_task(self._dispatch_requests())
            self.logger.info("ClientManager dispatcher started.")

    def start_dispatcher_thread(self):
        """Starts a dedicated thread to run the asyncio loop and the dispatcher ('run_forever' mode)."""
        self.initialize_mode = 'run_forever'
        if self.dispatcher_thread is None:
            self.loop = asyncio.new_event_loop()
            self.dispatcher_thread = threading.Thread(
                target=self._run_loop_forever, args=(self.loop,), daemon=True
            )
            self.dispatcher_thread.start()

            # Start the dispatcher task in the new event loop
            # This returns a concurrent.futures.Future
            self.dispatcher_task_future = asyncio.run_coroutine_threadsafe(self._dispatch_requests(), self.loop)
            self.logger.info("ClientManager dispatcher thread started successfully.")

    def _run_loop_forever(self, loop):
        """The target function for the background thread."""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def submit_to_queue(self, request, future):
        # add the request to the queue in the background loop
        worker_tid = threading.get_ident()
        if self.loop is None:
            raise RuntimeError("Event loop for ClientManager is not initialized.")

        try:
            # asyncio.run_coroutine_threadsafe(
            #     self.request_queue.put((request, future)),
            #     self.loop
            # )
            self.request_queue.put_nowait((request, future))
        except Exception as e:
            # If coroutine_threadsafe fails (e.g. loop closed), set exception on future from this thread
            self.logger.error(f"[{request.provider}][TID:{worker_tid}] Failure inside submit_to_queue: {e}")
            # Use loop.call_soon to set the exception back in the dispatcher's loop context
            self.loop.call_soon(future.set_exception, LLMResponse(text=f"Queue put failed internally: {e}"))

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Submits a request to the queue and waits for the response.

        This method acts as a producer, adding the request to the queue and
        then awaiting a Future object which will be completed by the dispatcher.
        """
        worker_tid = threading.get_ident()
        self.logger.info(f"[{request.provider}] Generate called. Worker TID: {worker_tid}")
        future = Future()

        if self.initialize_mode == 'create_task':
            # Put the request and the future object into the queue
            self.logger.debug(f"Submitting request for provider {request.provider} to the queue.")
            await self.request_queue.put((request, future))
            self.logger.debug(f"Request for model {request.provider} added to the queue.")

        elif self.initialize_mode == 'run_forever':
            # Submit the request to the queue in the background thread's event loop
            if self.loop is None:
                raise RuntimeError("Event loop for ClientManager is not initialized.")

            # Schedule the submission in the background thread's event loop
            self.loop.call_soon_threadsafe(self.submit_to_queue, request, future)
            self.logger.info(f"Request for provider {request.provider} submitted to the queue in background thread.")
        else:
            raise RuntimeError("ClientManager is not properly initialized.")

        start_time = time.time()
        if self.loop is None:
            raise RuntimeError("Event loop for ClientManager is not initialized.")

        try:
            self.logger.info(f"[{request.provider}] Awaiting response from dispatcher (using controlled polling)...")

            # The spinning loop with sleep to keep Celery worker responsive
            # This is a workaround for Celery's known issue with async tasks
            # Reference issue:
            # https://github.com/celery/celery/issues/6603
            while not future.done():
                if (time.time() - start_time) > self.config.timeout_seconds:
                    self.loop.call_soon_threadsafe(future.cancel)

                    raise TimeoutError(
                        f"Async generation timed out after {self.config.timeout_seconds} seconds due to synchronization failure in Celery Worker."
                    )

                time.sleep(self.config.sleep_interval_seconds)

            self.logger.info(f"[{request.provider}] Awaiting response from dispatcher...")
            response = await future
        except Exception as e:
            self.logger.error(f"[{request.provider}] Error awaiting response: {e}", exc_info=True)
            raise e

        return response

    def generate_stream(self, request: LLMRequest) -> Generator[LLMResponse, Any, Any]:
        """
        Generates a streaming response from the appropriate handler.
        """
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

        if self.loop and self.loop.is_running():
            self.logger.warning("Initiating graceful shutdown...")

            # 1. Wait for current queue items to be processed
            await self.request_queue.join()

            # 2. Stop the dispatcher task
            if self.dispatcher_task_future and not self.dispatcher_task_future.done():
                # run_coroutine_threadsafe returns a Future, we need to call cancel on the task it wraps.
                # Since we don't have the Task object, the best we can do is stop the loop.
                self.logger.warning("Stopping the background event loop.")
                self.loop.call_soon_threadsafe(self.loop.stop)

            # 3. Wait for the thread to join (optional, but good practice)
            if self.dispatcher_thread and self.dispatcher_thread.is_alive():
                self.dispatcher_thread.join(timeout=5)
