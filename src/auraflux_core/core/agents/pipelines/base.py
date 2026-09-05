from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Type

if TYPE_CHECKING:
    from auraflux_core.core.agents.base_agent import BaseAgent


class BaseAgentPipeline(ABC):
    """
    Abstract Base Class for all Agent execution pipelines.
    Encapsulates lifecycle steps, branching logic, and tool coordination.
    """

    @abstractmethod
    async def execute(self, agent: "BaseAgent", payload: Dict[str, Any]) -> Any:
        """
        Executes the specialized pipeline logic for the given agent.

        Args:
            agent: The parent BaseAgent instance providing LLM, tools, and output parser.
            payload: Input payload required for pipeline processing.
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
            class PlanAndExecutePipeline(BasePipeline):
                ...
        """
        def decorator(pipeline_cls: Type["BaseAgentPipeline"]):
            if name in cls._registry:
                raise ValueError(
                    f"Pipeline registration conflict: Name '{name}' is already registered to "
                    f"{cls._registry[name].__name__}."
                )
            cls._registry[name] = pipeline_cls
            return pipeline_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> "BaseAgentPipeline":
        """
        Retrieves and instantiates a registered pipeline instance by name.

        Args:
            name: The registered key name of the pipeline strategy.

        Returns:
            BasePipeline: An instantiated stateless pipeline strategy.
        """
        pipeline_cls = cls._registry.get(name)
        if not pipeline_cls:
            registered_names = list(cls._registry.keys())
            raise KeyError(
                f"Pipeline strategy '{name}' is not registered. Available pipelines: {registered_names}"
            )
        return pipeline_cls()

    @classmethod
    def list_registered(cls) -> Dict[str, Type["BaseAgentPipeline"]]:
        """Returns a copy of all registered pipelines for debugging or introspection."""
        return cls._registry.copy()
