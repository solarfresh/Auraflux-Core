# Plan and Execute Agent Tutorial (`plan-execute-agent-tutorial.md`)

This tutorial demonstrates how to build an advanced agent utilizing the [`PlanAndExecutePipeline`](src/auraflux_core/core/agents/pipelines/plan_and_execute.py:200) and implementing the [`PlanAndExecuteHandler`](src/auraflux_core/core/agents/pipelines/plan_and_execute.py:19) interface within the **Auraflux-Core** framework, including detailed Hook payload management, qualitative validation rules, and lifecycle pre-conditions.

---

## 1. Introduction & Pipeline Lifecycle

The **Plan-and-Execute Pipeline** decomposes complex workflows into deterministic stages:
1. **Planning (`plan`)**: Intent extraction and initial plan JSON generation.
2. **Pre-Execution Validation & Reflection (`pre_execution_validation`)**: Iterative qualitative inspection and repair loop.
3. **Tool Execution Workflow (`tool_workflow`)**: Deterministic code-controlled execution of tools based on plan steps.
4. **Synthesis (`synthesis`)**: Aggregating plan output and tool execution results into a final response.
5. **Post-Synthesis Verification (`post_synthesis_verification`)**: Boundary and policy checks on the final response.

---

## 2. Prerequisites & Pre-Conditions

Before running a `plan_and_execute` agent, ensure the following pre-conditions are met:
- **Client Manager Initialized**: Must call `await client_manager.initialize()` before executing the agent.
- **Prompt Templates Registered**: The `PromptConfig` must define templates for each stage key:
  - `"plan"`: Prompt for generating initial steps.
  - `"plan_refinement"`: Prompt for reflecting and repairing invalid plans.
  - `"synthesis"`: Prompt for compiling tool execution results.
- **Interface Compliance**: The agent class **must** inherit from both [`BaseAgent`](src/auraflux_core/core/agents/base.py:22) and [`PlanAndExecuteHandler`](src/auraflux_core/core/agents/pipelines/plan_and_execute.py:19).

---

## 3. Hook Contracts & Payload Flow

Each stage communicates via well-defined hook methods. Below is how payloads and contexts flow through the pipeline:

- `get_plan_context(payload)` -> Extracts variables for the planning prompt.
- `inspect_plan_output(payload, plan_output)` -> Returns a `ValidationResult` (`is_valid: bool`, `reasons: List[str]`).
- `get_plan_refinement_context(payload, plan_output, validation_result)` -> Prepares variables for repair prompts if validation fails.
- `merge_refined_plan(payload, current_plan, refined_plan, validation_result)` -> Merges repaired plans back into the current execution state.
- `execute_tool_workflow(payload, plan_output)` -> Executes code-controlled tools, returning a list of `Message` objects.
- `get_synthesis_context(payload, plan_output, tool_results)` -> Structures variables (`payload`, `plan_output`, `tool_results`) for synthesis.
- `parse_final_output(payload, plan_output, raw_llm_output)` -> Formats the ultimate return object for the caller.

---

## 4. Comprehensive Implementation Code

```python
import asyncio
from typing import List, Dict, Any, Optional
from auraflux_core.core.agents.base import BaseAgent
from auraflux_core.core.agents.pipelines.plan_and_execute import PlanAndExecuteHandler
from auraflux_core.core.clients.client_manager import ClientManager
from auraflux_core.core.schemas.agents import AgentConfig
from auraflux_core.core.schemas.messages import PromptConfig, SystemPromptSpec, Message
from auraflux_core.core.schemas.pipelines import ValidationResult

class AdvancedPlanExecuteAgent(BaseAgent, PlanAndExecuteHandler):
    """An advanced agent implementing PlanAndExecuteHandler with qualitative validation and hooks."""

    def get_plan_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Hook: Customize variables passed to the planning prompt template."""
        return {
            "user_query": payload.get("prompt"),
            "domain": payload.get("domain", "general")
        }

    async def inspect_plan_output(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any]
    ) -> ValidationResult:
        """
        Stage 1.5 Hook: Qualitative Step Validation.
        Checks if the generated plan contains actionable steps and valid structure.
        """
        class CustomValidationResult(ValidationResult):
            def __init__(self, is_valid: bool, reasons: List[str]):
                self.is_valid = is_valid
                self.reasons = reasons

        steps = plan_output.get("steps", [])
        reasons = []

        if not isinstance(steps, list) or len(steps) == 0:
            reasons.append("Plan must contain a non-empty 'steps' list.")

        for idx, step in enumerate(steps):
            if not step.get("description"):
                reasons.append(f"Step {idx} is missing a valid 'description'.")

        is_valid = len(reasons) == 0
        return CustomValidationResult(is_valid=is_valid, reasons=reasons)

    def get_plan_refinement_context(
        self,
        payload: Dict[str, Any],
        plan_output: Dict[str, Any],
        validation_result: ValidationResult,
    ) -> Optional[Dict[str, Any]]:
        """Hook: Prepares context for plan reflection if qualitative inspection fails."""
        return {
            "user_query": payload.get("prompt"),
            "current_plan": plan_output,
            "validation_reasons": validation_result.reasons
        }

    async def execute_tool_workflow(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any]
    ) -> List[Message]:
        """Stage 2: Deterministic execution of tool workflows based on the validated plan."""
        steps = plan_output.get("steps", [])
        tool_results: List[Message] = []

        for idx, step in enumerate(steps):
            desc = step.get("description")
            # Simulate tool execution (e.g., retriever or calculator)
            tool_output = f"Result for step {idx}: Executed [{desc}] successfully."
            tool_results.append(Message(role="tool", content=tool_output))

        return tool_results

    def get_synthesis_context(
        self,
        payload: Dict[str, Any],
        plan_output: Dict[str, Any],
        tool_results: List[Message]
    ) -> Dict[str, Any]:
        """Hook: Structures context variables for the synthesis prompt template."""
        return {
            "payload": payload,
            "plan_output": plan_output,
            "tool_results": [msg.content for msg in tool_results]
        }

    def parse_final_output(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any], raw_llm_output: str
    ) -> Any:
        """Formats the final response returned to the client."""
        return {
            "status": "success",
            "query": payload.get("prompt"),
            "final_plan": plan_output,
            "report": raw_llm_output
        }

async def main():
    # 1. Initialize Client Manager
    client_manager = ClientManager()
    await client_manager.initialize()

    # 2. Define Multi-Stage Prompt Configuration
    prompt_config = PromptConfig(
        system=SystemPromptSpec(
            templates={
                "default": "You are an expert autonomous planning agent.",
                "plan": "Create a JSON plan with 'steps' for query: {user_query} in domain: {domain}.",
                "plan_refinement": "Your previous plan failed validation: {validation_reasons}. Fix the plan for query: {user_query}.",
                "synthesis": "Synthesize the following tool execution results into a final report: {tool_results}"
            }
        )
    )

    config = AgentConfig(
        name="AdvancedPlanExecuteAgent",
        description="An advanced agent with validation hooks and multi-stage prompts.",
        provider="gemini",
        model="gemini-flash-lite-latest",
        temperature=0.3,
        prompt_config=prompt_config,
        pipeline_name="plan_and_execute",
        max_refine_retries=2
    )

    # 3. Instantiate Agent
    agent = AdvancedPlanExecuteAgent(
        config=config,
        client_manager=client_manager
    )

    # 4. Execute Agent with payload
    payload = {"prompt": "Perform security audit for microservices architecture.", "domain": "security"}
    result = await agent.execute(payload=payload)

    print("Execution Result:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 5. Summary & Best Practices
- **Isolation**: Keep business logic in `execute_tool_workflow` separate from LLM prompt formatting.
- **Resilience**: Implement `inspect_plan_output` and `get_plan_refinement_context` to allow self-correction before tool execution.
