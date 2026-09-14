from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from auraflux_core.core.agents.pipelines.base import (BaseAgentPipeline,
                                                      PipelineRegistry)
from auraflux_core.core.schemas.messages import Message

if TYPE_CHECKING:
    from auraflux_core.core.agents.base_agent import BaseAgent


class PlanAndExecuteHandler(ABC):
    """
    Handler interface for PlanAndExecutePipeline.
    Decouples stage-specific contracts and validation logic from BaseAgent.
    """

    @abstractmethod
    def build_plan_messages(self, payload: Dict[str, Any]) -> List[Message]:
        """
        Stage 1: Constructs initial messages for intent planning or extraction.
        """
        pass

    async def inspect_plan_output(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any]
    ) -> Any:
        """
        Stage 1.5 Inspection Hook.
        Executes deterministic rules (e.g., Rule Checker Tool) to inspect the plan_output.
        Must return an object with an `is_valid` boolean attribute and a `reasons` list.

        Default: always valid. Handlers that don't need inspection incur only one
        cheap await per pipeline run, no LLM call.
        """
        class DefaultValidationResult:
            is_valid = True
            reasons = []
        return DefaultValidationResult()

    def build_plan_refinement_messages(
        self, payload: Dict[str, Any], flawed_output: Dict[str, Any], validation_result: Any
    ) -> Optional[List[Message]]:
        """
        Stage 1.5 Reflection Prompt Hook.
        Constructs refinement messages based on validation failure reasons.

        Returning None signals that this handler has no repair capability for the
        current failure (or chooses not to repair it). The pipeline records the
        validation reasons on the plan output and stops the loop immediately for
        this attempt, instead of spinning through remaining retries.
        """
        return None

    @abstractmethod
    async def execute_tool_workflow(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any]
    ) -> List[Message]:
        """
        Stage 2: Executes deterministic code-controlled tool workflows (e.g., RAG, Calculators).
        """
        pass

    @abstractmethod
    def build_synthesis_messages(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any], tool_results: List[Message]
    ) -> List[Message]:
        """
        Stage 3: Constructs final synthesis and report generation messages.
        """
        pass

    async def validate_synthesis_output(
        self, payload: Dict[str, Any], synth_output: str, tool_results: List[Message]
    ) -> Optional[Any]:
        """
        Stage 3.5 Post-Synthesis Boundary Verification Hook.
        Validates synthesized responses against domain constraints.

        Returning None signals the pipeline to bypass Stage 3.5 validation.
        """
        return None

    @abstractmethod
    def parse_final_output(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any], raw_llm_output: str
    ) -> Any:
        """
        Parses and formats the final pipeline result for the client.
        """
        pass


@PipelineRegistry.register("plan_and_execute")
class PlanAndExecutePipeline(BaseAgentPipeline):
    """
    Deterministic Plan-and-Execute Pipeline structured into purpose-driven stage functions.
    Supports dynamic hook detection for automatic stage bypassing.
    """

    async def execute(self, agent: "BaseAgent", payload: Dict[str, Any]) -> Any:
        if not isinstance(agent, PlanAndExecuteHandler):
            raise TypeError(f"Agent '{agent.name}' must implement PlanAndExecuteHandler interface.")

        # Stage 1: Initial Planning & Extraction
        plan_output = await self._execute_planning_stage(agent, payload)

        # Stage 1.5: Pre-Execution Validation & Reflection Loop
        refined_plan_output = await self._run_pre_execution_validation_loop(agent, payload, plan_output)

        # Stage 2: Code-Controlled Tool Execution Workflow
        tool_results = await self._execute_tool_workflow_stage(agent, payload, refined_plan_output)

        # Stage 3: Response Synthesis
        raw_synthesis_output = await self._execute_synthesis_stage(
            agent, payload, refined_plan_output, tool_results
        )

        # Stage 3.5: Post-Synthesis Verification
        verified_synthesis_output = await self._verify_post_synthesis_boundary(
            agent, payload, raw_synthesis_output, tool_results
        )

        # Output Formatting
        return agent.parse_final_output(payload, refined_plan_output, verified_synthesis_output)

    async def _execute_planning_stage(
        self, agent: "BaseAgent", payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates and parses the initial plan or extracted structured output.
        """
        handler = cast(PlanAndExecuteHandler, agent)
        plan_messages = handler.build_plan_messages(payload)
        plan_response = await agent.generate(plan_messages)
        return agent.output_parser.parse_json(plan_response.content)

    async def _run_pre_execution_validation_loop(
        self, agent: "BaseAgent", payload: Dict[str, Any], initial_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes an iterative inspection and reflection loop on the plan output.

        No upfront probing call is made to detect whether refinement is implemented.
        The loop's own control flow already covers every handler shape:
          - Handler overrides nothing: inspect_plan_output defaults to is_valid=True,
            loop exits on the first iteration with zero LLM calls.
          - Handler overrides only inspect_plan_output (validate-but-not-repair):
            build_plan_refinement_messages stays at its default (returns None), so a
            failing validation records `validation_warnings` and stops immediately
            instead of spinning through the remaining retries.
          - Handler overrides both: full check -> reflect -> recheck loop, bounded by
            max_refine_retries.
        """
        handler = cast(PlanAndExecuteHandler, agent)
        max_retries = payload.get("max_refine_retries", 1)
        current_plan = initial_plan

        for attempt in range(max_retries + 1):
            validation_result = await handler.inspect_plan_output(payload, current_plan)

            # Stop Criteria 1: Passed deterministic validation
            if getattr(validation_result, "is_valid", True):
                break

            refinement_messages = handler.build_plan_refinement_messages(
                payload, current_plan, validation_result
            )

            # Stop Criteria 2: Handler has no repair path for this failure. Record and
            # stop now rather than re-validating an unchanged plan for the remaining
            # retries.
            if refinement_messages is None:
                current_plan["validation_warnings"] = getattr(validation_result, "reasons", [])
                break

            # Stop Criteria 3: Maximum retries reached; log warnings and fallback
            if attempt >= max_retries:
                current_plan["validation_warnings"] = getattr(validation_result, "reasons", [])
                break

            # Generate Pass 2 reflection messages and execute repair
            refinement_response = await agent.generate(refinement_messages)
            current_plan = agent.output_parser.parse_json(refinement_response.content)

        return current_plan

    async def _execute_tool_workflow_stage(
        self, agent: "BaseAgent", payload: Dict[str, Any], plan_output: Dict[str, Any]
    ) -> List[Message]:
        """
        Executes code-controlled tools (e.g., database queries, calculators) deterministically.
        """
        handler = cast(PlanAndExecuteHandler, agent)
        return await handler.execute_tool_workflow(payload, plan_output)

    async def _execute_synthesis_stage(
        self,
        agent: "BaseAgent",
        payload: Dict[str, Any],
        plan_output: Dict[str, Any],
        tool_results: List[Message],
    ) -> str:
        """
        Synthesizes tool execution results and plan output into a raw LLM response.
        """
        handler = cast(PlanAndExecuteHandler, agent)
        synth_messages = handler.build_synthesis_messages(payload, plan_output, tool_results)
        synth_response = await agent.generate(synth_messages)
        return synth_response.content

    async def _verify_post_synthesis_boundary(
        self,
        agent: "BaseAgent",
        payload: Dict[str, Any],
        raw_synthesis_output: str,
        tool_results: List[Message],
    ) -> str:
        """
        Runs post-synthesis verification checks against mathematical or policy constraints.
        Bypasses automatically if validate_synthesis_output is not implemented.
        """
        handler = cast(PlanAndExecuteHandler, agent)
        validated_synth = await handler.validate_synthesis_output(
            payload, raw_synthesis_output, tool_results
        )
        return raw_synthesis_output if validated_synth is None else validated_synth
