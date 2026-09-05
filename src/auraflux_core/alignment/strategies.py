import asyncio
from typing import Any, Dict, List, Optional, Tuple

from auraflux_core.alignment.agents import BaseAlignmentAgent
from auraflux_core.alignment.objective_claim.schemas import \
    ObjectiveClaimVerdict
from auraflux_core.core.orchestrators.state import OrchestratorState
from auraflux_core.core.orchestrators.strategies.base import \
    OrchestrationStrategy

DEFAULT_AGENT_MAPPING = {
    "objective_claim": "ObjectiveClaimAgent",
}


class AlignmentOrchestrationStrategy(OrchestrationStrategy):
    """
    Alignment Orchestration Strategy for multi-agent proposition verification.
    """

    def __init__(self, agent_mapping: Optional[Dict[str, str]] = None):
        super().__init__()
        self.agent_mapping = agent_mapping or DEFAULT_AGENT_MAPPING

    async def execute(
        self,
        input_data: List[Dict[str, Any]],
        tools: Dict[str, Any],
        agents: Dict[str, BaseAlignmentAgent],
        state: OrchestratorState
    ) -> OrchestratorState:
        self.register_tools_to_agents(
            tools=tools,
            agents=agents,
            target_agent_keys=set(self.agent_mapping.values())
        )

        tasks = self._dispatch_claim_tasks(input_data, agents)
        if not tasks:
            state.metadata["verdicts"] = []
            state.metadata["is_locked"] = True
            return state

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        verdicts, errors, has_block = self._aggregate_verdicts(raw_results)

        state.metadata["verdicts"] = [
            v.model_dump() if hasattr(v, "model_dump") else v for v in verdicts
        ]
        if errors:
            state.metadata.setdefault("errors", []).extend(errors)

        state.metadata["is_locked"] = not has_block
        return state

    # ----------------------------------------------------------------------
    # Helper Methods
    # ----------------------------------------------------------------------

    def _dispatch_claim_tasks(
        self, input_data: List[Dict[str, Any]], agents: Dict[str, BaseAlignmentAgent]
    ) -> List[asyncio.Task[Any]]:
        """Routes claims to agents and dispatches async tasks."""
        tasks: List[asyncio.Task[Any]] = []

        for claim in input_data:
            claim_type = claim.get("type", "")
            claim_id = claim.get("id", "")
            claim_text = claim.get("text", "")

            agent_key = self.agent_mapping.get(claim_type)
            agent = agents.get(agent_key) if agent_key else None

            if not agent:
                self.logger.warning(
                    f"No agent configured/found for claim_type '{claim_type}' (ID: {claim_id}). Skipping."
                )
                continue

            if hasattr(agent, "diagnose_and_verify"):
                task = asyncio.create_task(
                    agent.diagnose_and_verify(
                        proposition_id=claim_id, claim_text=claim_text
                    )
                )
                tasks.append(task)
            else:
                self.logger.error(
                    f"Agent '{agent_key}' does not implement 'diagnose_and_verify'."
                )

        return tasks

    def _aggregate_verdicts(
        self, raw_results: List[Any]
    ) -> Tuple[List[ObjectiveClaimVerdict], List[str], bool]:
        """Processes gather results, extracting valid verdicts, errors, and block conditions."""
        verdicts: List[ObjectiveClaimVerdict] = []
        errors: List[str] = []
        has_block_condition = False

        for result in raw_results:
            if isinstance(result, ObjectiveClaimVerdict):
                verdicts.append(result)
                if result.status == "UNSUPPORTED":
                    has_block_condition = True
            elif isinstance(result, BaseException):
                err_msg = str(result)
                self.logger.error(f"Error during claim verification: {err_msg}")
                errors.append(err_msg)

        return verdicts, errors, has_block_condition
