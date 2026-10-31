from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple, Type, cast

from auraflux_core.core.agents.pipelines import PipelineRegistry
from auraflux_core.core.agents.pipelines.base import BaseAgentPipeline
from auraflux_core.core.configs.logging_config import get_logger
from auraflux_core.core.orchestrators.base import OrchestrationStrategy
from auraflux_core.core.orchestrators.constants import (
    DEFAULT_PIPELINE_NAME, MAX_ORCHESTRATION_ITERATIONS, MAX_RESUME_ATTEMPTS)
from auraflux_core.core.orchestrators.exceptions import (
    AgentPipelineError, CheckpointCorruptedError, CheckpointError,
    ConcurrencyConflictError, InvalidOrchestrationStateError,
    MaxResumeAttemptsExceededError, OrchestrationLimitExceededError)
from auraflux_core.core.orchestrators.ports import (CheckpointPort,
                                                    OrchestrationCallbackPort)
from auraflux_core.core.orchestrators.router import BaseAgentRouter
from auraflux_core.core.orchestrators.state import (ExecutionCursor,
                                                    OrchestratorState,
                                                    OrchestratorStatus)

logger = get_logger(__name__)


class DualLayerAgenticStrategy(OrchestrationStrategy):
    """
    A robust dual-layer orchestration strategy featuring checkpoint session resilience,
    exact-step resume tracking, circuit breaker loop protection, and safe error propagation.
    Refactored with single responsibility principle for clean execution flow.
    """

    def __init__(
        self,
        router: BaseAgentRouter,
        callback_port: Optional[OrchestrationCallbackPort] = None,
        checkpoint_port: Optional[CheckpointPort] = None
    ) -> None:
        self.router = router
        self.callback_port = callback_port
        self.checkpoint_port = checkpoint_port
        self._lock = asyncio.Lock()
        self._active_sessions: set = set()

    async def _change_status(
        self,
        state: OrchestratorState,
        new_status: OrchestratorStatus,
        error_message: Optional[str] = None
    ) -> None:
        """Updates orchestrator state and broadcasts status change via callback port."""
        state.update_status(new_status, error_message=error_message)
        if self.callback_port:
            try:
                await self.callback_port.on_status_changed(state, new_status)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error("callback_on_status_changed_failed", session_id=state.session_id, exc_info=True)

    async def _save_checkpoint(self, state: OrchestratorState) -> None:
        """
        Safely persists the orchestration state checkpoint.
        Delegates serialization defense and metadata sanitization to the state layer.
        """
        if self.checkpoint_port:
            try:
                # Serialize and dump state checkpoint (metadata sanitization is handled within state encapsulation)
                state_json = state.dump_checkpoint()
                await self.checkpoint_port.save(state.session_id, state_json)
                logger.debug(
                    "checkpoint_saved",
                    session_id=state.session_id,
                    status=state.status.value
                )
            except Exception as e:
                logger.error(
                    "checkpoint_save_failed",
                    session_id=state.session_id,
                    error=str(e),
                    exc_info=True
                )
                raise

    def _prepare_runtime_environment(
        self,
        tools: Optional[Dict[str, Any]],
        agents: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Ensures tools are registered and agents registry is valid before execution or resume.
        Returns the resolved and normalized tools dictionary.
        """
        if not agents:
            raise AgentPipelineError("Agents registry cannot be empty.")

        resolved_tools = tools or {}

        try:
            self.register_tools_to_agents(resolved_tools, agents)
        except Exception as e:
            logger.error("register_tools_to_agents_failed", error=str(e), exc_info=True)
            raise AgentPipelineError(f"Failed to register tools to agents: {e}") from e

        return resolved_tools

    async def _prepare_execution_environment(
        self,
        tools: Optional[Dict[str, Any]],
        agents: Optional[Dict[str, Any]],
        state: OrchestratorState
    ) -> ExecutionCursor:
        """
        Prepares the runtime environment, validates state transition gates,
        and enforces retry limits and attempt persistence for session resilience.
        """
        if state.status == OrchestratorStatus.COMPLETED:
            raise InvalidOrchestrationStateError(
                f"Cannot execute or resume session {state.session_id} "
                "because it is already COMPLETED."
            )

        # Bind and register tools to the agents registry
        self._prepare_runtime_environment(tools, agents)

        if state.status == OrchestratorStatus.IDLE:
            await self._change_status(state, OrchestratorStatus.INGESTING)
            await self._change_status(state, OrchestratorStatus.PROCESSING)
            await self._save_checkpoint(state)
        elif state.status == OrchestratorStatus.PROCESSING:
            await self._change_status(state, OrchestratorStatus.PROCESSING)
        elif state.status == OrchestratorStatus.FAILED:
            # Enforce hard circuit breaker limit on retry/resume attempts using typed state attribute
            if state.attempt_count >= MAX_RESUME_ATTEMPTS:
                raise MaxResumeAttemptsExceededError(
                    f"Session {state.session_id} exceeded maximum resume attempts ({MAX_RESUME_ATTEMPTS})."
                )

            # Increment attempt count and immediately persist to prevent uncounted poison steps
            state.bump_attempt()
            await self._change_status(state, OrchestratorStatus.PROCESSING)
            await self._save_checkpoint(state)
        else:
            logger.error(
                "execute_called_on_invalid_status",
                status=state.status.value,
                session_id=state.session_id
            )
            raise InvalidOrchestrationStateError(
                f"Cannot execute orchestration session in invalid status: {state.status.value}"
            )

        # Return the validated structured execution cursor
        if not isinstance(state.cursor, ExecutionCursor):
            state.cursor = ExecutionCursor()

        return state.cursor

    async def _resolve_agent_and_pipeline(
        self,
        state: OrchestratorState,
        agents: Dict[str, Any],
        payload: Any,
        last_result: Any,
        cursor: ExecutionCursor
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Resolves the next agent via router or cursor resumption and loads its pipeline steps
        with robust boundary checks, avoiding premature cursor mutation.
        """
        # If cursor has an active agent binding with pending steps to skip, use it for resume
        if cursor.agent_name and cursor.completed_steps_count > 0:
            agent_name = cursor.agent_name
        else:
            agent_name = self.router.get_next_agent(state, agents, last_result)

        if not agent_name:
            logger.info("dual_layer_strategy_routing_ended", session_id=state.session_id)
            return None, []

        target_agent = agents.get(agent_name)
        if not target_agent:
            logger.error("routed_agent_not_found", agent_name=agent_name, session_id=state.session_id)
            raise AgentPipelineError(f"Agent '{agent_name}' specified by router not found in actors registry.")

        agent_pipeline = getattr(target_agent, "pipeline", None)
        if agent_pipeline is None:
            agent_config = getattr(target_agent, "config", None)
            pipeline_name = getattr(agent_config, "pipeline_name", DEFAULT_PIPELINE_NAME)
            try:
                pipeline_cls = cast(Type[BaseAgentPipeline], PipelineRegistry.get(pipeline_name))
                if pipeline_cls:
                    agent_pipeline = pipeline_cls()
            except Exception as e:
                logger.warning("pipeline_registry_lookup_failed", agent_name=agent_name, error=str(e))

        if agent_pipeline is None:
            raise AgentPipelineError(f"Agent '{agent_name}' does not have a valid bound pipeline.")

        pipeline_getter = getattr(agent_pipeline, "get_execute_pipeline", None)
        if not callable(pipeline_getter):
            raise AgentPipelineError(f"Pipeline for agent '{agent_name}' must implement callable 'get_execute_pipeline'.")

        raw_steps = pipeline_getter(payload=payload)
        execution_steps = cast(List[Dict[str, Any]], raw_steps)

        if cursor.completed_steps_count < 0 or cursor.completed_steps_count > len(execution_steps):
            raise CheckpointCorruptedError(
                f"Corrupted checkpoint: completed_steps_count ({cursor.completed_steps_count}) "
                f"is out of bounds for pipeline step count ({len(execution_steps)})."
            )

        # Successfully validated and resolved; anchor the cursor agent name securely
        cursor.agent_name = agent_name

        return agent_name, execution_steps

    async def _process_agent_steps(
        self,
        state: OrchestratorState,
        agents: Dict[str, Any],
        agent_name: str,
        execution_steps: List[Dict[str, Any]],
        payload: Any,
        remaining_to_skip: int
    ) -> Tuple[Any, bool]:
        """
        Processes micro-steps for the current agent, handles exact-step skipping,
        dispatches executions, and persists atomic checkpoints while ensuring proper cursor isolation.
        """
        last_result = state.metadata.get("last_result")

        for step in execution_steps:
            # Skip already completed steps for resilient resume handling
            if remaining_to_skip > 0:
                remaining_to_skip -= 1
                continue

            method_name = step.get("method_name")
            if not isinstance(method_name, str):
                raise TypeError(f"Pipeline step 'method_name' must be a string, got {type(method_name).__name__}.")

            step_kwargs_raw = step.get("kwargs", {})
            if not isinstance(step_kwargs_raw, dict):
                raise TypeError(f"Pipeline step 'kwargs' must be a dictionary, got {type(step_kwargs_raw).__name__}.")

            step_kwargs = {**step_kwargs_raw}
            step_kwargs.setdefault("payload", payload)

            if self.callback_port:
                try:
                    await self.callback_port.on_step_start(state, agent_name, method_name, step_kwargs)
                except Exception:
                    logger.error("callback_on_step_start_failed", session_id=state.session_id, exc_info=True)

            # Dispatch execution to the target actor/agent method
            last_result = await self.dispatch(
                state=state,
                actor_name=agent_name,
                actors=agents,
                method_name=method_name,
                **step_kwargs
            )

            # Safely set last result utilizing state-level encapsulation and serialization defense
            state.set_last_result(last_result)

            # Update the structured execution cursor and persist state atomically
            state.cursor.agent_name = agent_name
            state.cursor.completed_steps_count += 1

            if self.callback_port and state.history:
                try:
                    await self.callback_port.on_step_complete(state, state.history[-1])
                except Exception:
                    logger.error("callback_on_step_complete_failed", session_id=state.session_id, exc_info=True)

            await self._save_checkpoint(state)

        # Advance macro iteration and reset completed step count for the upcoming agent round
        state.cursor.macro_iteration += 1
        state.cursor.completed_steps_count = 0

        # Evaluate whether the orchestration goal has been achieved
        is_completed = self.router.evaluate_completion(state, last_result)
        return last_result, is_completed

    async def _handle_execution_failure(self, state: OrchestratorState, e: Exception) -> None:
        """Handles orchestration execution failures by isolating errors, updating status, and writing fallback checkpoints."""
        error_msg = str(e)
        logger.error("orchestration_execution_failed", session_id=state.session_id, error=error_msg, exc_info=True)

        try:
            await self._change_status(state, OrchestratorStatus.FAILED, error_message=error_msg)
        except Exception:
            logger.error("failed_status_transition_error", session_id=state.session_id, exc_info=True)

        if self.callback_port:
            try:
                await self.callback_port.on_orchestration_failed(state, error_msg)
            except Exception:
                logger.error("callback_on_orchestration_failed_error", session_id=state.session_id, exc_info=True)

        try:
            await self._save_checkpoint(state)
        except Exception:
            logger.error("failed_checkpoint_save_error", session_id=state.session_id, exc_info=True)

    async def execute(
        self,
        payload: Any,
        tools: Optional[Dict[str, Any]],
        agents: Dict[str, Any],
        state: OrchestratorState
    ) -> OrchestratorState:
        """
        Executes or continues the dual-layer orchestration loop with environment prep,
        structured cursor tracking, and state-first persistence order.
        """
        async with self._lock:
            if state.session_id in self._active_sessions:
                raise ConcurrencyConflictError(
                    f"Session {state.session_id} is already actively executing concurrently."
                )
            self._active_sessions.add(state.session_id)

        try:
            cursor = await self._prepare_execution_environment(tools, agents, state)
            last_result = state.metadata.get("last_result")

            try:
                while cursor.macro_iteration < MAX_ORCHESTRATION_ITERATIONS:
                    agent_name, execution_steps = await self._resolve_agent_and_pipeline(
                        state, agents, payload, last_result, cursor
                    )
                    if not agent_name:
                        break

                    last_result, finished = await self._process_agent_steps(
                        state, agents, agent_name, execution_steps, payload, cursor.completed_steps_count
                    )
                    if finished:
                        break
                else:
                    raise OrchestrationLimitExceededError(
                        f"Orchestration exceeded maximum iterations limit ({MAX_ORCHESTRATION_ITERATIONS})."
                    )

                # Transition to completed state and persist checkpoint before triggering callbacks
                await self._change_status(state, OrchestratorStatus.COMPLETED)
                await self._save_checkpoint(state)

                if self.callback_port:
                    try:
                        await self.callback_port.on_orchestration_complete(state)
                    except Exception:
                        logger.error("callback_on_orchestration_complete_failed", session_id=state.session_id, exc_info=True)

            except Exception as e:
                await self._handle_execution_failure(state, e)
                raise
        finally:
            async with self._lock:
                self._active_sessions.discard(state.session_id)

        return state

    @classmethod
    async def resume(
        cls,
        session_id: str,
        checkpoint_port: CheckpointPort,
        router: BaseAgentRouter,
        agents: Dict[str, Any],
        callback_port: Optional[OrchestrationCallbackPort] = None,
        payload: Any = None,
        tools: Optional[Dict[str, Any]] = None
    ) -> Tuple['DualLayerAgenticStrategy', OrchestratorState]:
        """
        Resumes an orchestration session from a persisted checkpoint
        with structured cursor tracking and runtime environment alignment.
        """
        try:
            state_json = await checkpoint_port.load(session_id)
        except Exception as e:
            logger.error("checkpoint_load_failed", session_id=session_id, error=str(e), exc_info=True)
            raise CheckpointError(f"Failed to load checkpoint for session {session_id!r}: {e}") from e

        if not state_json:
            raise CheckpointCorruptedError(f"No checkpoint found or empty content for session_id: {session_id!r}")

        try:
            state = OrchestratorState.load_checkpoint(state_json)
        except Exception as e:
            logger.error("checkpoint_deserialization_failed", session_id=session_id, error=str(e), exc_info=True)
            raise CheckpointCorruptedError(f"Failed to deserialize checkpoint for session {session_id!r}: {e}") from e

        # Gate check: Prevent resuming an already completed session
        if state.status == OrchestratorStatus.COMPLETED:
            raise InvalidOrchestrationStateError(f"Cannot resume session {session_id} because it is already COMPLETED.")

        strategy = cls(router=router, callback_port=callback_port, checkpoint_port=checkpoint_port)

        logger.info(
            "session_resumed_from_checkpoint",
            session_id=session_id,
            status=state.status.value,
            cursor=state.cursor
        )

        # Proceed execution only for valid resumable statuses
        if state.status in (OrchestratorStatus.PROCESSING, OrchestratorStatus.FAILED):
            await strategy.execute(
                payload=payload,
                tools=tools or {},
                agents=agents,
                state=state
            )
        else:
            raise InvalidOrchestrationStateError(f"Cannot resume session {session_id} in current status: {state.status.value}")

        return strategy, state
