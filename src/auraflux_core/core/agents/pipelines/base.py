import time
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Type

import structlog

from auraflux_core.core.configs.logging_config import get_logger

if TYPE_CHECKING:
    from auraflux_core.core.agents.base_agent import BaseAgent

logger = get_logger(__name__)


class BaseAgentPipeline(ABC):
    """
    Abstract Base Class for all Agent execution pipelines.
    Encapsulates lifecycle steps, branching logic, and tool coordination.
    """

    async def execute(self, agent: "BaseAgent", payload: Dict[str, Any]) -> Any:
        """
        Executes the specialized pipeline logic for the given agent.
        Automatically binds global Context variables for tracing and logs execution boundaries.

        Args:
            agent: The parent BaseAgent instance providing LLM, tools, and output parser.
            payload: Input payload required for pipeline processing.
        """
        trace_id = payload.get("trace_id") or str(uuid.uuid4())
        pipeline_name = self.__class__.__name__

        # 自動綁定全域 Pipeline Context，底層的所有 Log (如 BaseAgent/Handlers) 皆會自動帶入這些 key
        structlog.contextvars.bind_contextvars(
            trace_id=trace_id,
            agent_name=agent.name,
            pipeline_name=pipeline_name,
        )

        start_time = time.perf_counter()
        logger.info(
            "pipeline_execution_started",
            payload_keys=list(payload.keys()),
        )

        try:
            result = await self._run_pipeline(agent, payload)
            latency_sec = time.perf_counter() - start_time

            logger.info(
                "pipeline_execution_completed",
                latency_sec=round(latency_sec, 4),
                status="success",
            )
            return result
        except Exception as e:
            latency_sec = time.perf_counter() - start_time
            logger.error(
                "pipeline_execution_failed",
                error_type=type(e).__name__,
                error_msg=str(e),
                latency_sec=round(latency_sec, 4),
                exc_info=True,
            )
            raise e
        finally:
            # 清理 ContextVars，防止 Context 污染後續請求
            structlog.contextvars.unbind_contextvars(
                "trace_id", "agent_name", "pipeline_name", "stage_name"
            )

    @abstractmethod
    async def _run_pipeline(self, agent: "BaseAgent", payload: Dict[str, Any]) -> Any:
        """
        Internal implementation of specialized pipeline execution.
        Subclasses must implement this method instead of overriding execute().
        """
        pass


class PipelineRegistry:
    """
    Centralized registry for managing and dynamically instantiating Agent Pipelines.
    Thread-safe class-level storage for pipeline classes.
    """

    _registry: Dict[str, Type["BaseAgentPipeline"]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a BasePipeline subclass under a unique name.

        Usage:
            @PipelineRegistry.register("plan_and_execute")
            class PlanAndExecutePipeline(BaseAgentPipeline):
                ...
        """
        def decorator(pipeline_cls: Type["BaseAgentPipeline"]):
            if name in cls._registry:
                logger.error(
                    "pipeline_registration_conflict",
                    registered_name=name,
                    existing_class=cls._registry[name].__name__,
                    attempted_class=pipeline_cls.__name__,
                )
                raise ValueError(
                    f"Pipeline registration conflict: Name '{name}' is already registered to "
                    f"{cls._registry[name].__name__}."
                )
            cls._registry[name] = pipeline_cls
            logger.info(
                "pipeline_registered",
                registered_name=name,
                pipeline_class=pipeline_cls.__name__,
            )
            return pipeline_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> "BaseAgentPipeline":
        """
        Retrieves and instantiates a registered pipeline instance by name.

        Args:
            name: The registered key name of the pipeline strategy.

        Returns:
            BaseAgentPipeline: An instantiated stateless pipeline strategy.
        """
        pipeline_cls = cls._registry.get(name)
        if not pipeline_cls:
            registered_names = list(cls._registry.keys())
            logger.error(
                "pipeline_not_found",
                requested_name=name,
                available_pipelines=registered_names,
            )
            raise KeyError(
                f"Pipeline strategy '{name}' is not registered. Available pipelines: {registered_names}"
            )
        return pipeline_cls()

    @classmethod
    def list_registered(cls) -> Dict[str, Type["BaseAgentPipeline"]]:
        """Returns a copy of all registered pipelines for debugging or introspection."""
        return cls._registry.copy()
