from typing import Any, Dict, List, Optional, Type, cast

from auraflux_core.core.agents.pipelines import PipelineRegistry
from auraflux_core.core.agents.pipelines.base import BaseAgentPipeline
from auraflux_core.core.configs.logging_config import get_logger
from auraflux_core.core.orchestrators.adapters import OrchestrationCallbackPort
from auraflux_core.core.orchestrators.base import OrchestrationStrategy
from auraflux_core.core.orchestrators.router import BaseAgentRouter
from auraflux_core.core.orchestrators.state import (OrchestratorState,
                                                    OrchestratorStatus)
from auraflux_core.core.orchestrators.strategies.base import \
    OrchestrationStrategy

logger = get_logger(__name__)


class DualLayerAgenticStrategy(OrchestrationStrategy):
    """
    A lean dual-layer orchestration strategy that robustly resolves the agent's
    pipeline (via direct attribute or PipelineRegistry fallback) to drive fine-grained steps
    via strategy.dispatch(), guaranteeing complete telemetry tracking and decoupled event broadcasting.
    """

    def __init__(
        self,
        router: BaseAgentRouter,
        callback_port: Optional[OrchestrationCallbackPort] = None
    ) -> None:
        self.router = router
        self.callback_port = callback_port

    async def _change_status(self, state: OrchestratorState, new_status: OrchestratorStatus) -> None:
        """
        Updates the global orchestrator state and broadcasts the status change event if a callback port is present.
        """
        state.update_status(new_status)
        if self.callback_port:
            await self.callback_port.on_status_changed(state, new_status)

    async def execute(
        self,
        input_data: Any,
        tools: Dict[str, Any],
        agents: Dict[str, Any],
        state: OrchestratorState
    ) -> OrchestratorState:
        """
        Executes the dual-layer orchestration loop:
        1. Registers tools across all available agents.
        2. Drives macro-level routing via the router.
        3. Resolves micro-level tactical steps through agent pipelines.
        4. Dispatches each step sequentially under telemetry and lifecycle protection.
        """
        logger.info("dual_layer_strategy_execution_started")
        await self._change_status(state, OrchestratorStatus.INGESTING)

        # Step 1: Register tools across agents
        self.register_tools_to_agents(tools, agents)

        last_result = None

        try:
            await self._change_status(state, OrchestratorStatus.PROCESSING)

            # Step 2: Multi-agent coordination loop (Macro level)
            while True:
                agent_name = self.router.get_next_agent(state, agents, last_result)
                if not agent_name:
                    logger.info("dual_layer_strategy_routing_ended")
                    break

                target_agent = agents.get(agent_name)
                if not target_agent:
                    logger.error("routed_agent_not_found", agent_name=agent_name)
                    raise ValueError(f"Agent '{agent_name}' specified by router not found in actors registry.")

                # Step 3: Robust pipeline resolution aligned with BaseAgent & conftest standards
                agent_pipeline = getattr(target_agent, "pipeline", None)

                if not agent_pipeline:
                    # Fallback: Check if agent config specifies a pipeline_name to load from PipelineRegistry
                    agent_config = getattr(target_agent, "config", None)
                    pipeline_name = getattr(agent_config, "pipeline_name", "direct")

                    try:
                        pipeline_cls = cast(Type[BaseAgentPipeline], PipelineRegistry.get(pipeline_name))
                        if pipeline_cls:
                            agent_pipeline = pipeline_cls()
                    except Exception as e:
                        logger.warning("pipeline_registry_lookup_failed", agent_name=agent_name, error=str(e))

                if not agent_pipeline:
                    logger.error("agent_missing_pipeline", agent_name=agent_name)
                    raise AttributeError(f"Agent '{agent_name}' does not have a valid bound pipeline or registered pipeline name.")

                pipeline_getter = getattr(agent_pipeline, "get_run_pipeline", None)
                if not callable(pipeline_getter):
                    logger.error("pipeline_missing_getter", agent_name=agent_name, pipeline_class=agent_pipeline.__class__.__name__)
                    raise AttributeError(f"Pipeline for agent '{agent_name}' must implement a callable 'get_run_pipeline' method.")

                # Retrieve and cast tactical execution steps (Micro level)
                raw_steps = pipeline_getter(payload=input_data)
                execution_steps = cast(List[Dict[str, Any]], raw_steps)
                logger.info("pipeline_steps_retrieved", agent_name=agent_name, step_count=len(execution_steps))

                # Step 4: Sequentially dispatch each step under orchestrator telemetry and lifecycle protection
                for step in execution_steps:
                    method_name = step.get("method_name")

                    if not isinstance(method_name, str):
                        logger.error("invalid_pipeline_step_method_name", agent_name=agent_name, method_name=method_name)
                        raise TypeError(f"Pipeline step 'method_name' must be a string, got {type(method_name).__name__}.")

                    step_kwargs = step.get("kwargs", {})
                    if not isinstance(step_kwargs, dict):
                        step_kwargs = {}

                    step_kwargs.setdefault("payload", input_data)

                    # Trigger pre-step lifecycle hook
                    if self.callback_port:
                        await self.callback_port.on_step_start(state, agent_name, method_name, step_kwargs)

                    logger.info("dispatching_pipeline_step", agent_name=agent_name, method=method_name)

                    # Execute the atomic step via uniform dispatch mechanism
                    last_result = await self.dispatch(
                        state=state,
                        actor_name=agent_name,
                        actors=agents,
                        method_name=method_name,
                        **step_kwargs
                    )

                    # Trigger post-step telemetry completion hook
                    if self.callback_port and state.history:
                        await self.callback_port.on_step_complete(state, state.history[-1])

                # Step 5: Evaluate completion criteria via router
                if self.router.evaluate_completion(state, last_result):
                    logger.info("dual_layer_strategy_completion_reached")
                    break

            # Finalize orchestration success state
            await self._change_status(state, OrchestratorStatus.COMPLETED)
            if self.callback_port:
                await self.callback_port.on_orchestration_complete(state)

        except Exception as e:
            # Handle orchestration failure lifecycle
            await self._change_status(state, OrchestratorStatus.FAILED)
            if self.callback_port:
                await self.callback_port.on_orchestration_failed(state, str(e))
            raise

        logger.info("dual_layer_strategy_execution_completed")
        return state
