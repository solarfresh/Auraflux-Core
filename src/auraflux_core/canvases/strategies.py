import logging
import time
from typing import Any, Dict

from auraflux_core.core.orchestrators.state import (ExecutionStep,
                                                    OrchestratorState,
                                                    OrchestratorStatus)
from auraflux_core.core.orchestrators.strategies.base import \
    OrchestrationStrategy
from auraflux_core.core.tools.base_tool import BaseTool


class AgenticStrategy(OrchestrationStrategy):
    """
    An Agentic (Reflective) Strategy.
    Implements a Feedback Loop: Actor -> Auditor -> Actor (Refinement).
    """

    def __init__(
        self,
        actor_name: str,
        auditor_name: str,
        ingestion_tool_name: str = "file_reader",
        max_retries: int = 2
    ):
        self.actor_name = actor_name
        self.auditor_name = auditor_name
        self.ingestion_tool = ingestion_tool_name
        self.max_retries = max_retries
        self.logger = logging.getLogger(f"[{self.__class__.__name__}]")

    async def execute(
        self,
        input_data: Any,
        tools: Dict[str, Any],
        agents: Dict[str, Any],
        state: OrchestratorState
    ) -> OrchestratorState:

        state.source_identifier = str(input_data)

        # 1. Ingestion
        ingest_res = await self._wrapped_call(
            state, self.ingestion_tool, tools[self.ingestion_tool].run, file_path=input_data
        )
        units = ingest_res.get("chunks") or ingest_res.get("units") or []

        # 2. Identification of Actors
        actor = agents.get(self.actor_name) or tools.get(self.actor_name)
        auditor = agents.get(self.auditor_name) or tools.get(self.auditor_name)

        if not actor or not auditor:
            raise ValueError(f"Actor '{self.actor_name}' or Auditor '{self.auditor_name}' missing.")

        state.update_status(OrchestratorStatus.PROCESSING)

        processed_results = []

        # 3. Reflective Loop per Unit
        for unit in units:
            state.current_unit_id = unit.get("source_id") or unit.get("id")
            feedback = None
            final_unit_output = None

            for attempt in range(self.max_retries + 1):
                # A. Execution/Refinement Phase
                if attempt > 0:
                    state.update_status(OrchestratorStatus.REFINING)
                    state.retry_count += 1

                actor_method = actor.process if hasattr(actor, "process") else actor.run
                output = await self._wrapped_call(
                    state,
                    self.actor_name,
                    actor_method,
                    content=unit.get("content"),
                    source_id=state.current_unit_id,
                    feedback=feedback # Injecting previous critique
                )

                # B. Validation Phase
                state.update_status(OrchestratorStatus.VALIDATING)
                auditor_method = auditor.audit if hasattr(auditor, "audit") else auditor.run

                audit_report = await self._wrapped_call(
                    state,
                    self.auditor_name,
                    auditor_method,
                    data=output # Auditor inspects the actor's output
                )

                if audit_report.get("is_valid"):
                    final_unit_output = output
                    break

                # C. Feedback Loop Preparation
                feedback = audit_report.get("critique")
                self.logger.warning(f"Refinement required for {state.current_unit_id}: {feedback}")

                if attempt == self.max_retries:
                    self.logger.error(f"Max retries reached for unit {state.current_unit_id}")
                    final_unit_output = output # Fallback to last known state

            processed_results.append(final_unit_output)

        # 4. State Update
        state.output["raw_results"] = processed_results
        return state

    async def _wrapped_call(self, state: OrchestratorState, name: str, func: Any, **kwargs) -> Any:
        """Standardized wrapper for telemetry and error handling."""
        start_time = time.perf_counter()
        try:
            result = await func(**kwargs)
            duration = (time.perf_counter() - start_time) * 1000

            step = ExecutionStep(
                tool_name=name,
                input_params=kwargs,
                output_data=result,
                token_usage=result.get("usage", 0) if isinstance(result, dict) else 0,
                duration_ms=duration
            )
            state.add_step(step)
            return result
        except Exception as e:
            state.add_step(ExecutionStep(tool_name=name, input_params=kwargs, error=str(e)))
            raise e


class SequentialStrategy(OrchestrationStrategy):
    """
    A domain-agnostic Sequential Strategy.
    It follows a strict linear path: Ingest -> Process Units -> Finalize.
    """

    def __init__(self, actor_name: str, ingestion_tool_name: str = "file_reader"):
        self.actor_name = actor_name
        self.ingestion_tool = ingestion_tool_name
        self.logger = logging.getLogger(f"[{self.__class__.__name__}]")

    async def execute(
        self,
        input_data: Any,
        tools: Dict[str, Any],
        agents: Dict[str, Any],
        state: OrchestratorState
    ) -> OrchestratorState:

        state.source_identifier = str(input_data)

        # 1. Ingestion
        ingest_res = await self._wrapped_call(
            state, self.ingestion_tool, tools[self.ingestion_tool].run, file_path=input_data
        )
        units = ingest_res.get("chunks") or ingest_res.get("units") or []

        # 2. Identification of the Actor (Agent or Tool)
        actor = agents.get(self.actor_name) or tools.get(self.actor_name)
        if not actor:
            raise ValueError(f"Actor '{self.actor_name}' not found.")

        state.update_status(OrchestratorStatus.PROCESSING)

        # 3. Sequential Loop
        results = []
        for unit in units:
            state.current_unit_id = unit.get("source_id") or unit.get("id")

            # Determine calling convention (Agent.process vs Tool.run)
            call_method = actor.process if hasattr(actor, "process") else actor.run

            output = await self._wrapped_call(
                state,
                self.actor_name,
                call_method,
                content=unit.get("content"),
                source_id=state.current_unit_id
            )
            results.append(output)

        # 4. State Update using domain-agnostic 'output'
        state.output["raw_results"] = results
        return state

    async def _wrapped_call(self, state: OrchestratorState, name: str, func: Any, **kwargs) -> Any:
        """Helper to ensure every action is recorded via state.add_step"""
        start_time = time.perf_counter()
        try:
            result = await func(**kwargs)
            duration = (time.perf_counter() - start_time) * 1000

            step = ExecutionStep(
                tool_name=name,
                input_params=kwargs,
                output_data=result,
                token_usage=result.get("usage", 0) if isinstance(result, dict) else 0,
                duration_ms=duration
            )
            state.add_step(step)
            return result
        except Exception as e:
            state.add_step(ExecutionStep(tool_name=name, input_params=kwargs, error=str(e)))
            raise e
