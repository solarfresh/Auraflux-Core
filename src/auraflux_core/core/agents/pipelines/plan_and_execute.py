from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from auraflux_core.core.agents.pipelines.base import (BaseAgentPipeline,
                                                      PipelineRegistry)
from auraflux_core.core.schemas.messages import Message

if TYPE_CHECKING:
    from auraflux_core.core.agents.base_agent import BaseAgent


class PlanAndExecuteHandler(ABC):
    """
    Separate Handler interface specifically for PlanAndExecutePipeline.
    Decouples Pipeline-specific hook contracts from BaseAgent.
    """

    @abstractmethod
    def build_plan_messages(self, payload: Dict[str, Any]) -> List[Message]:
        pass

    @abstractmethod
    async def execute_tool_workflow(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any]
    ) -> List[Message]:
        """Stage 2: Deterministic code-controlled tool execution workflow."""
        pass

    @abstractmethod
    def build_synthesis_messages(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any], tool_results: List[Message]
    ) -> List[Message]:
        pass

    @abstractmethod
    def parse_final_output(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any], raw_llm_output: str
    ) -> Any:
        pass


@PipelineRegistry.register("plan_and_execute")
class PlanAndExecutePipeline(BaseAgentPipeline):
    """
    Deterministic Plan-and-Execute Pipeline.
    Stage 1: Intent Planning (LLM)
    Stage 2: Code-controlled Tool Execution (Deterministic Workflow)
    Stage 3: Synthesis & Verdict Generation (LLM)
    """

    async def execute(self, agent: "BaseAgent", payload: Dict[str, Any]) -> Any:
        if not isinstance(agent, PlanAndExecuteHandler):
            raise TypeError(f"Agent '{agent.name}' must implement PlanAndExecuteHandler interface.")

        plan_messages = agent.build_plan_messages(payload)
        plan_response = await agent.generate(plan_messages)
        plan_output = agent.output_parser.parse_json(plan_response.content)

        # Stage 2: Tool Workflow
        tool_results: List[Message] = await agent.execute_tool_workflow(payload, plan_output)

        synth_messages = agent.build_synthesis_messages(payload, plan_output, tool_results)
        synth_response = await agent.generate(synth_messages)

        return agent.parse_final_output(payload, plan_output, synth_response.content)
