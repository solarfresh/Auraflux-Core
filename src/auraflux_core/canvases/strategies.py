import json
from typing import Any, Dict

from auraflux_core.core.orchestrators.state import (OrchestratorState,
                                                    OrchestratorStatus)
from auraflux_core.core.orchestrators.strategies.base import \
    OrchestrationStrategy
from auraflux_core.core.schemas.messages import Message


class AgenticStrategy(OrchestrationStrategy):
    """
    Agentic (Reflective) Strategy.
    Uses 'name' in Message to distinguish between Architect and Auditor turns.
    """

    def __init__(self, actor_name: str, auditor_name: str, validator_tool_name: str, max_retries: int = 2):
        super().__init__()
        self.actor_name = actor_name
        self.auditor_name = auditor_name
        self.validator_tool_name = validator_tool_name
        self.max_retries = max_retries

    async def execute(
        self, input_data: Any, tools: Dict[str, Any], agents: Dict[str, Any], state: OrchestratorState
    ) -> OrchestratorState:

        # 1. Ingestion
        ingest_res = await self.dispatch(state, "file_reader", tools, file_path=input_data)
        units = ingest_res.get("chunks") or []

        state.update_status(OrchestratorStatus.PROCESSING)
        processed_results = []

        for unit in units:
            # Initial User request with name 'User'
            messages = [
                Message(role="user", content=f"Extract: {unit['content']}", name="User")
            ]

            for attempt in range(self.max_retries + 1):
                # dispatch calls architect.generate(messages)
                output = await self.dispatch(
                    state,
                    self.actor_name,
                    agents,
                    messages=messages,
                )
                output_json = json.loads(output.content)
                self.logger.info('====== output ======')
                self.logger.info(output)

                valid_res = await self.dispatch(
                    state,
                    self.validator_tool_name,
                    tools,
                    nodes=output_json.get("nodes", []),
                    edges=output_json.get("edges", [])
                )

                # Auditor message named by the agent name
                audit_msgs = [
                    Message(role="user", content=f"Audit this: {output}\nValidation Results: {valid_res}", name="User")
                ]
                audit_res = await self.dispatch(state, self.auditor_name, agents, messages=audit_msgs)
                audit_res_json = json.loads(audit_res.content)

                if valid_res.get("is_valid") and audit_res_json.get("is_valid"):
                    state.output["final_graph"] = output.content
                    break

                # Prepare Refinement
                if attempt < self.max_retries:
                    state.update_status(OrchestratorStatus.REFINING)

                    # We record the architect's failed attempt with its name
                    messages.append(
                        Message(role="assistant", content=str(output), name=self.actor_name)
                    )
                    # We record the auditor's critique with its name
                    critique_combined = f"Tool Errors: {audit_res_json.get('errors')}\nAuditor Critique: {audit_res_json.get('critique')}"
                    messages.append(
                        Message(role="user", content=critique_combined, name=self.auditor_name)
                    )
                else:
                    processed_results.append(output)

        state.output["raw_results"] = processed_results
        return state


class SequentialStrategy(OrchestrationStrategy):
    """
    A domain-agnostic Sequential Strategy.
    Follows a linear execution path: Ingest -> Atomic Process -> Collect.
    Used as the baseline for comparing against reflective (Agentic) modes.
    """

    def __init__(self, actor_name: str, ingestion_tool_name: str = "file_reader"):
        super().__init__()
        self.actor_name = actor_name
        self.ingestion_tool = ingestion_tool_name

    async def execute(
        self,
        input_data: Any,
        tools: Dict[str, Any],
        agents: Dict[str, Any],
        state: OrchestratorState
    ) -> OrchestratorState:

        # 1. Ingestion Phase
        # Falls back to 'run' method automatically via dispatch logic
        ingest_res = await self.dispatch(
            state,
            self.ingestion_tool,
            tools,
            file_path=input_data
        )
        units = ingest_res.get("chunks") or ingest_res.get("units") or []

        state.update_status(OrchestratorStatus.PROCESSING)
        results = []

        # 2. Linear Processing Loop
        for unit in units:
            unit_id = unit.get("source_id") or unit.get("id")
            state.current_unit_id = unit_id

            self.logger.info(f"Processing Unit [{unit_id}]: Initiating knowledge extraction.")

            # Construct a single-turn message for the actor
            messages = [
                Message(
                    role="user",
                    content=f"Extract knowledge from the following content:\n{unit.get('content')}",
                    name="User"
                )
            ]

            # Dispatch to actor's generate() method (default)
            # This captures duration, tokens, and output into the state history
            output = await self.dispatch(
                state,
                self.actor_name,
                agents,
                messages=messages
            )

            results.append(output)

        # 3. Final State Update
        state.output["raw_results"] = results
        return state
