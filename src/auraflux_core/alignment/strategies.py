import asyncio
from typing import Any, Dict, List, Optional

from auraflux_core.alignment.agents import BaseAlignmentAgent
from auraflux_core.alignment.objective_claim.schemas import \
    ObjectiveClaimVerdict
from auraflux_core.core.orchestrators.state import OrchestratorState
from auraflux_core.core.orchestrators.strategies.base import \
    OrchestrationStrategy

# Default routing dictionary: maps claim.type to agent key in agents dict
DEFAULT_AGENT_MAPPING = {
    "objective_claim": "ObjectiveClaimAgent",
    # "subjective_claim": "SubjectiveClaimAgent",
    # "logical_claim": "LogicalConsistencyAgent",
}


class AlignmentOrchestrationStrategy(OrchestrationStrategy):
    """
    Alignment Orchestration Strategy for multi-agent proposition verification.

    Dynamically routes atomized claims to their corresponding alignment agents based on claim types,
    registers cross-checking tools across all active agents, and aggregates verification verdicts.
    """

    def __init__(self, agent_mapping: Optional[Dict[str, str]] = None):
        super().__init__()
        # Flexible routing map: claim_type -> agent_key
        self.agent_mapping = agent_mapping or DEFAULT_AGENT_MAPPING

    async def execute(
        self,
        input_data: List[Dict[str, Any]],
        tools: Dict[str, Any],
        agents: Dict[str, BaseAlignmentAgent],
        state: OrchestratorState
    ) -> OrchestratorState:
        verdicts: List[Any] = []
        tasks: List[asyncio.Task[Any]] = []

        # 1. Register tools across all candidate alignment agents in the mapping
        if tools:
            for agent_key in set(self.agent_mapping.values()):
                agent = agents.get(agent_key)
                if agent and hasattr(agent, "register_tools"):
                    agent.register_tools(tools)

        # 2. Dynamic routing & task dispatching based on claim_type
        for claim in input_data:
            claim_type = claim.get("type", "")
            claim_id = claim.get("id", "")
            claim_text = claim.get("text", "")

            # Resolve the target agent using the routing mapping
            agent_key = self.agent_mapping.get(claim_type)
            agent = agents.get(agent_key) if agent_key else None

            if not agent:
                self.logger.warning(
                    f"No agent configured/found for claim_type '{claim_type}' (ID: {claim_id}). Skipping."
                )
                continue

            # Dispatch task based on agent capabilities
            if hasattr(agent, "diagnose_and_verify"):
                task = asyncio.create_task(
                    agent.diagnose_and_verify(
                        proposition_id=claim_id,
                        claim_text=claim_text
                    )
                )
                tasks.append(task)
            else:
                self.logger.error(
                    f"Agent '{agent_key}' does not implement 'diagnose_and_verify'."
                )

        # 3. Guard clause: Return locked state immediately if no tasks were dispatched
        if not tasks:
            state.metadata["verdicts"] = []
            state.metadata["is_locked"] = True
            return state

        # 4. Parallel collection of diagnostic results (Map-Reduce)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        has_block_condition = False
        for result in results:
            if isinstance(result, ObjectiveClaimVerdict):
                verdicts.append(result)
                # Trigger BLOCK state if an UNSUPPORTED claim is detected
                if result.status == "UNSUPPORTED":
                    has_block_condition = True
            elif isinstance(result, BaseException):
                self.logger.error(f"Error during claim verification: {str(result)}")
                state.metadata.setdefault("errors", []).append(str(result))

        # 5. Update OrchestratorState with results and lock status
        state.metadata["verdicts"] = [
            v.model_dump() if hasattr(v, "model_dump") else v for v in verdicts
        ]
        state.metadata["is_locked"] = not has_block_condition

        return state
