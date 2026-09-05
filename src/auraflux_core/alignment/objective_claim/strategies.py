import asyncio
from typing import Any, Dict, List

from auraflux_core.alignment.objective_claim.agents import ObjectiveClaimAgent
from auraflux_core.alignment.objective_claim.schemas import \
    ObjectiveClaimVerdict
from auraflux_core.core.orchestrators.state import OrchestratorState
from auraflux_core.core.orchestrators.strategies.base import \
    OrchestrationStrategy


class AlignmentOrchestrationStrategy(OrchestrationStrategy):
    """
    Parallel Map-Reduce strategy for alignment proposition verification.
    """

    async def execute(
        self,
        input_data: List[Dict[str, Any]],
        tools: Dict[str, Any],
        agents: Dict[str, Any],
        state: OrchestratorState
    ) -> OrchestratorState:
        verdicts: List[ObjectiveClaimVerdict] = []
        tasks: List[asyncio.Task[ObjectiveClaimVerdict]] = []

        for claim in input_data:
            claim_type = claim.get("type")
            claim_id = claim.get("id", "")
            claim_text = claim.get("text", "")

            if claim_type == "objective_claim":
                agent: ObjectiveClaimAgent | None = agents.get("ObjectiveClaimAgent", None)
                if agent:
                    task = asyncio.create_task(
                        agent.diagnose_and_verify(claim_id, claim_text)
                    )
                    tasks.append(task)

        if not tasks:
            state.metadata["verdicts"] = []
            state.metadata["is_locked"] = True
            return state

        results = await asyncio.gather(*tasks, return_exceptions=True)

        has_block_condition = False
        for result in results:
            if isinstance(result, ObjectiveClaimVerdict):
                verdicts.append(result)
                if result.status == "UNSUPPORTED":
                    has_block_condition = True
            elif isinstance(result, BaseException):
                state.metadata.setdefault("errors", []).append(str(result))

        state.metadata["verdicts"] = [v.model_dump() for v in verdicts]
        state.metadata["is_locked"] = not has_block_condition

        return state