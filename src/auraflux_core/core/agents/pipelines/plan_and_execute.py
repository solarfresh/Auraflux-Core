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
    def extract_tool_call_spec(self, plan_output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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

    async def execute(self, agent: "BaseAgent", payload: Dict[str, Any]) -> Any:
        # Ensure agent implements the required handler contract
        if not isinstance(agent, PlanAndExecuteHandler):
            raise TypeError(f"Agent '{agent.name}' must implement PlanAndExecuteHandler interface.")

        # Stage 1: Plan
        plan_messages = agent.build_plan_messages(payload)
        plan_response = await agent.generate(plan_messages)
        plan_output = agent.output_parser.parse_json(plan_response.content)

        # Stage 2: Tool
        tool_spec = agent.extract_tool_call_spec(plan_output)
        tool_results = []
        if tool_spec and agent.tool_executor:
            tool_name = tool_spec.get("tool_name")
            tool_args = tool_spec.get("tool_args", {})

            if tool_name and tool_name in agent.tool_executor.tool_registry:
                tool_msg = await agent.tool_executor.run(tool_name=tool_name, tool_args=tool_args)
                tool_results.append(tool_msg)

        # Stage 3: Synthesis
        synth_messages = agent.build_synthesis_messages(payload, plan_output, tool_results)
        synth_response = await agent.generate(synth_messages)

        return agent.parse_final_output(payload, plan_output, synth_response.content)
