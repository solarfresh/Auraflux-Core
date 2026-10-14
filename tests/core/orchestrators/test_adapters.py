import json
from typing import Any, Dict, cast

import pytest

from auraflux_core.core.orchestrators.adapters import CallbackTransportAdapter
from auraflux_core.core.orchestrators.state import (ExecutionStep,
                                                    OrchestratorState,
                                                    OrchestratorStatus)


@pytest.mark.asyncio
async def test_callback_adapter_status_changed() -> None:
    """
    Validate that calling on_status_changed correctly pushes the status change event into the queue.
    """
    adapter = CallbackTransportAdapter()
    state = OrchestratorState(session_id="session_test_123")

    await adapter.on_status_changed(state, OrchestratorStatus.PROCESSING)

    raw_item = await adapter.queue.get()
    assert raw_item is not None
    item = cast(Dict[str, Any], raw_item)

    assert item["event"] == "status_changed"
    assert item["session_id"] == "session_test_123"
    assert item["status"] == "PROCESSING"


@pytest.mark.asyncio
async def test_callback_adapter_step_lifecycle() -> None:
    """
    Validate the correctness of event payloads for step start (on_step_start) and step complete (on_step_complete).
    """
    adapter = CallbackTransportAdapter()
    state = OrchestratorState(session_id="session_test_123")

    # 1. Test step start
    await adapter.on_step_start(state, "agent_a", "analyze_data", {"query": "test"})
    raw_start_item = await adapter.queue.get()
    assert raw_start_item is not None
    start_item = cast(Dict[str, Any], raw_start_item)

    assert start_item["event"] == "step_start"
    assert start_item["agent_name"] == "agent_a"
    assert start_item["method_name"] == "analyze_data"
    assert start_item["params"] == {"query": "test"}

    # 2. Test step complete
    step = ExecutionStep(
        actor_name="agent_a",
        method_name="analyze_data",
        input_params={"query": "test"},
        output_data={"result": "success"},
        token_usage=15,
        duration_ms=120.5
    )
    await adapter.on_step_complete(state, step)
    raw_complete_item = await adapter.queue.get()
    assert raw_complete_item is not None
    complete_item = cast(Dict[str, Any], raw_complete_item)

    assert complete_item["event"] == "step_complete"
    assert complete_item["data"]["actor_name"] == "agent_a"
    assert complete_item["data"]["method_name"] == "analyze_data"
    assert complete_item["data"]["token_usage"] == 15


@pytest.mark.asyncio
async def test_callback_adapter_orchestration_complete_and_failed() -> None:
    """
    Validate that orchestration completion and failure events are correctly sent and include the termination sentinel (None).
    """
    # Test complete workflow
    adapter_complete = CallbackTransportAdapter()
    state = OrchestratorState(session_id="session_test_123")
    state.output = {"final": "result"}

    await adapter_complete.on_orchestration_complete(state)

    raw_item = await adapter_complete.queue.get()
    assert raw_item is not None
    item = cast(Dict[str, Any], raw_item)

    assert item["event"] == "completed"
    assert item["output"] == {"final": "result"}

    # Verify termination sentinel is sent
    sentinel = await adapter_complete.queue.get()
    assert sentinel is None

    # Test failed workflow
    adapter_failed = CallbackTransportAdapter()
    await adapter_failed.on_orchestration_failed(state, "Database connection error")

    raw_fail_item = await adapter_failed.queue.get()
    assert raw_fail_item is not None
    fail_item = cast(Dict[str, Any], raw_fail_item)

    assert fail_item["event"] == "failed"
    assert fail_item["error"] == "Database connection error"

    fail_sentinel = await adapter_failed.queue.get()
    assert fail_sentinel is None


@pytest.mark.asyncio
async def test_stream_events_generator() -> None:
    """
    Validate that the stream_events() async generator correctly translates queue events
    into standard SSE format and terminates safely upon receiving the None sentinel.
    """
    adapter = CallbackTransportAdapter()
    state = OrchestratorState(session_id="session_stream_999")

    # Simulate sequential events dispatch: status change followed by completion
    await adapter.on_status_changed(state, OrchestratorStatus.INGESTING)
    await adapter.on_orchestration_complete(state)

    events = []
    async for sse_chunk in adapter.stream_events():
        events.append(sse_chunk)

    # Exactly two SSE chunks should be received
    assert len(events) == 2

    # Check if the first chunk matches the SSE format (event: ...\ndata: ...\n\n)
    assert events[0].startswith("event: status_changed\ndata: ")
    assert events[0].endswith("\n\n")

    # Check the second chunk
    assert events[1].startswith("event: completed\ndata: ")
    assert events[1].endswith("\n\n")

    # Verify data content can be parsed as JSON
    data_line = events[0].split("\n")[1].replace("data: ", "")
    parsed_data = cast(Dict[str, Any], json.loads(data_line))
    assert parsed_data["session_id"] == "session_stream_999"
    assert parsed_data["status"] == "INGESTING"
