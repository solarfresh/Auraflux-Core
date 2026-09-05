from typing import Any, Dict

import pytest

from auraflux_core.core.agents.base_agent import BaseAgent
from auraflux_core.core.agents.pipelines.base import (BaseAgentPipeline,
                                                      PipelineRegistry)


class DummyPipeline(BaseAgentPipeline):
    """Dummy pipeline class for testing purposes."""

    async def execute(self, agent: "BaseAgent", payload: Dict[str, Any]) -> Any:
        return "dummy_result"


@pytest.fixture(autouse=True)
def reset_pipeline_registry():
    """Ensure registry state is clean before and after each test execution."""
    original_registry = PipelineRegistry._registry.copy()
    yield
    PipelineRegistry._registry = original_registry


def test_register_pipeline_success():
    """Verify that a custom pipeline can be successfully registered into PipelineRegistry."""
    pipeline_name = "test_dummy_pipeline"

    # Register using decorator
    PipelineRegistry.register(pipeline_name)(DummyPipeline)

    assert pipeline_name in PipelineRegistry.list_registered()
    instance = PipelineRegistry.get(pipeline_name)
    assert isinstance(instance, DummyPipeline)


def test_register_pipeline_duplicate_conflict():
    """Verify that registering a pipeline with a duplicate name raises a ValueError."""
    pipeline_name = "test_duplicate_pipeline"

    PipelineRegistry.register(pipeline_name)(DummyPipeline)

    with pytest.raises(ValueError) as exc_info:
        PipelineRegistry.register(pipeline_name)(DummyPipeline)

    assert f"Pipeline registration conflict: Name '{pipeline_name}' is already registered" in str(exc_info.value)


def test_get_registered_pipeline():
    """Verify that retrieving a registered pipeline returns an instantiated BaseAgentPipeline."""
    pipeline_name = "test_instantiation_pipeline"

    PipelineRegistry.register(pipeline_name)(DummyPipeline)
    instance = PipelineRegistry.get(pipeline_name)

    assert isinstance(instance, BaseAgentPipeline)
    assert isinstance(instance, DummyPipeline)


def test_get_unregistered_pipeline_raises_key_error():
    """Verify that attempting to retrieve an unregistered pipeline raises KeyError with available pipelines in message."""
    unregistered_name = "non_existent_pipeline"

    with pytest.raises(KeyError) as exc_info:
        PipelineRegistry.get(unregistered_name)

    assert f"Pipeline strategy '{unregistered_name}' is not registered" in str(exc_info.value)


def test_list_registered_returns_copy():
    """Verify that list_registered returns a copy of internal registry to prevent unintended mutations."""
    registered = PipelineRegistry.list_registered()
    registered["fake_key"] = DummyPipeline

    assert "fake_key" not in PipelineRegistry._registry
