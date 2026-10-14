import asyncio
import json
from typing import Any, AsyncGenerator, Dict, Optional

from auraflux_core.core.orchestrators.ports import OrchestrationCallbackPort
from auraflux_core.core.orchestrators.state import (ExecutionStep,
                                                    OrchestratorState,
                                                    OrchestratorStatus)


class CallbackTransportAdapter(OrchestrationCallbackPort):
    """
    Adapter implementing OrchestrationCallbackPort to bridge core orchestration
    status and step events with an asynchronous queue for real-time SSE streaming.
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

    async def on_status_changed(self, state: OrchestratorState, new_status: OrchestratorStatus) -> None:
        await self.queue.put({
            "event": "status_changed",
            "session_id": state.session_id,
            "status": new_status.value
        })

    async def on_step_start(self, state: OrchestratorState, agent_name: str, method_name: str, step_kwargs: Dict[str, Any]) -> None:
        await self.queue.put({
            "event": "step_start",
            "session_id": state.session_id,
            "agent_name": agent_name,
            "method_name": method_name,
            "params": step_kwargs
        })

    async def on_step_complete(self, state: OrchestratorState, step: ExecutionStep) -> None:
        await self.queue.put({
            "event": "step_complete",
            "session_id": state.session_id,
            "status": state.status.value,
            "data": step.model_dump(mode="json")
        })

    async def on_orchestration_complete(self, state: OrchestratorState) -> None:
        await self.queue.put({
            "event": "completed",
            "session_id": state.session_id,
            "status": state.status.value,
            "output": state.output
        })
        await self.queue.put(None)  # Sentinel for stream termination

    async def on_orchestration_failed(self, state: OrchestratorState, error_message: str) -> None:
        await self.queue.put({
            "event": "failed",
            "session_id": state.session_id,
            "status": state.status.value,
            "error": error_message
        })
        await self.queue.put(None)  # Sentinel for stream termination

    async def stream_events(self) -> AsyncGenerator[str, None]:
        """Consumes the async queue and yields formatted Server-Sent Events (SSE)."""
        while True:
            item = await self.queue.get()
            if item is None:
                break
            yield f"event: {item['event']}\ndata: {json.dumps(item)}\n\n"
