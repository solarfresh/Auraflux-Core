import json
from typing import Any, Dict, List, Optional, Tuple

from auraflux_core.core.agents.base_agent import BaseAgent
from auraflux_core.core.agents.pipelines.plan_and_execute import \
    PlanAndExecuteHandler
from auraflux_core.core.configs.logging_config import get_logger
from auraflux_core.core.schemas.messages import Message
from auraflux_core.core.schemas.pipelines import ValidationResult

logger = get_logger(__name__)


class ExtractKeywordsAgent(BaseAgent, PlanAndExecuteHandler):
    """
    Agent responsible for extracting keywords and semantic triples from input payload,
    executing Stage 1.5 deterministic post-processing for both tags and triples via ToolExecutor.
    """

    def get_plan_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts context variables needed for Stage 1 planning template."""
        return {
            "excerpt_text": payload.get("excerpt_text", "")
        }

    def build_plan_messages(
        self,
        payload: Dict[str, Any],
        agent: "BaseAgent"
    ) -> List[Message]:
        """Stage 1: Constructs user messages for LLM keyword & triple extraction.

        Attempts dynamic lookup via `prompt_formatter.format_messages`. If absent or
        not callable, falls back to the default Stage 1 extraction user message.
        """
        return super().build_plan_messages(payload, agent)

    async def inspect_plan_output(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any]
    ) -> ValidationResult:
        """
        Stage 1.5 Inspection Hook:
        Delegates incremental triple processing and cleans tags.
        Returns ValidationResult aligned with TripleRuleCheckerTool inspection outputs.
        """
        try:
            flagged_reasons: List[str] = []
            check_data: Dict[str, Any] = {}

            # Step 1: Execute incremental triple processing & rule inspection
            if "triples" in plan_output and isinstance(plan_output["triples"], list) and self.tool_executor:
                has_flagged, flagged_reasons, check_data = await self._process_incremental_triples(
                    payload, plan_output
                )

            # Step 2: Clean tags
            if "tags" in plan_output and isinstance(plan_output["tags"], list):
                clean_func = self._get_cleaner_function()
                plan_output["tags"] = [
                    cleaned
                    for tag in plan_output["tags"]
                    if (cleaned := clean_func(tag))
                ]

            # Step 3: Return consistent ValidationResult to control Stage 1.5 Reflection Pass
            if flagged_reasons:
                return ValidationResult(
                    is_valid=False,
                    reasons=flagged_reasons,
                    metadata={
                        "check_data": check_data,
                        "flagged_triples": check_data.get("flagged_triples", []),
                        "clean_triples": check_data.get("clean_triples", []),
                    }
                )

            return ValidationResult(
                is_valid=True,
                reasons=[],
                metadata={
                    "check_data": check_data,
                    "clean_triples": check_data.get("clean_triples", []),
                }
            )

        except Exception as e:
            logger.error(
                "inspect_plan_output_failed",
                error_type=type(e).__name__,
                error_msg=str(e),
                exc_info=True,
            )
            return ValidationResult(
                is_valid=False,
                reasons=[f"Failed to post-process tags or triples: {str(e)}"]
            )

    def get_plan_refinement_context(
        self,
        payload: Dict[str, Any],
        plan_output: Dict[str, Any],
        validation_result: ValidationResult,
    ) -> Optional[Dict[str, Any]]:
        """
        Extracts context variables targeting flagged triples for Stage 1.5 Pass 2 repair.
        Returns None if no flagged triples are present, skipping Pass 2.
        """
        flagged_triples = []
        if hasattr(validation_result, "metadata") and isinstance(validation_result.metadata, dict):
            flagged_triples = validation_result.metadata.get("flagged_triples", [])

        if not flagged_triples:
            return None

        return {
            "excerpt_text": payload.get("excerpt_text", ""),
            "flagged_triples": json.dumps(flagged_triples, ensure_ascii=False, indent=2),
        }

    def build_plan_refinement_messages(
        self,
        payload: Dict[str, Any],
        plan_output: Dict[str, Any],
        validation_result: Any,
        agent: "BaseAgent",
    ) -> Optional[List[Message]]:
        """
        Stage 1.5 Reflection Prompt Hook:
        Constructs refinement messages specifically targeting flagged triples for Pass 2 repair.
        If no flagged triples exist, returns None to bypass Pass 2 execution.
        """
        return super().build_plan_refinement_messages(
            payload, plan_output, validation_result, agent
        )

    def merge_refined_plan(
        self,
        payload: Dict[str, Any],
        current_plan: Dict[str, Any],
        refined_plan: Dict[str, Any],
        validation_result: ValidationResult,
    ) -> Dict[str, Any]:
        merged_plan = dict(current_plan)

        accumulated_clean: List[Dict[str, Any]] = list(
            merged_plan.get("_accumulated_clean_triples", [])
        )
        clean_key_to_index = {
            (t.get("subject"), t.get("predicate"), t.get("object")): idx
            for idx, t in enumerate(accumulated_clean)
        }
        verified_clean = []
        if hasattr(validation_result, "metadata") and isinstance(validation_result.metadata, dict):
            verified_clean = validation_result.metadata.get("clean_triples", [])

        for c_item in verified_clean:
            c_key = (c_item.get("subject"), c_item.get("predicate"), c_item.get("object"))

            if c_key in clean_key_to_index:
                idx = clean_key_to_index[c_key]
                accumulated_clean[idx] = c_item
            else:
                clean_key_to_index[c_key] = len(accumulated_clean)
                accumulated_clean.append(c_item)

        incoming_triples = refined_plan.get("triples", [])
        candidate_triples = list(accumulated_clean)
        candidate_key_to_index = {
            (t.get("subject"), t.get("predicate"), t.get("object")): idx
            for idx, t in enumerate(candidate_triples)
        }
        for item in incoming_triples:
            key = (item.get("subject"), item.get("predicate"), item.get("object"))

            if key in candidate_key_to_index:
                idx = candidate_key_to_index[key]
                candidate_triples[idx] = item
            else:
                candidate_key_to_index[key] = len(candidate_triples)
                candidate_triples.append(item)

        merged_plan["_accumulated_clean_triples"] = accumulated_clean
        merged_plan["triples"] = candidate_triples

        return merged_plan

    async def _process_incremental_triples(
        self,
        payload: Dict[str, Any],
        plan_output: Dict[str, Any]
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Helper method dedicated to handling incremental triple processing and revision.
        Cleans incoming triples, inspects rule violations via TripleRuleCheckerTool,
        updates resolved fields by key, and safely appends newly passed clean triples.

        Returns:
            Tuple[bool, List[str]]: (has_flagged, combined_flag_reasons)
        """
        excerpt_text = payload.get("excerpt_text")
        raw_incoming_triples = plan_output.get("triples", [])

        # Step 1: Clean and split enumerated objects via triple_processor
        processed_res = await self.tool_executor.run(
            tool_name="triple_processor",
            tool_args={
                "triples": raw_incoming_triples,
            }
        )

        raw_processed = getattr(processed_res, "content", processed_res)
        if isinstance(raw_processed, str):
            try:
                processed_triples = json.loads(raw_processed)
            except json.JSONDecodeError:
                logger.error("failed_to_parse_triple_processor_output", raw_output=raw_processed)
                processed_triples = raw_incoming_triples
        elif isinstance(raw_processed, (list, dict)):
            processed_triples = raw_processed
        else:
            processed_triples = raw_incoming_triples

        # Step 2: Inspect candidate triples via triple_rule_checker
        check_res = await self.tool_executor.run(
            tool_name="triple_rule_checker",
            tool_args={
                "triples": processed_triples,
                "excerpt_text": excerpt_text
            }
        )

        raw_content = getattr(check_res, "content", check_res)
        if isinstance(raw_content, str):
            try:
                check_data = json.loads(raw_content)
            except json.JSONDecodeError:
                logger.error("failed_to_parse_tool_response_json", raw_content=raw_content)
                check_data = {}
        elif isinstance(raw_content, dict):
            check_data = raw_content
        else:
            check_data = {}

        has_flagged = False
        all_reasons: List[str] = []

        if isinstance(check_data, dict):
            flagged = check_data.get("flagged_triples", [])
            has_flagged = check_data.get("has_flagged", False)

            # Aggregate all _flag_reasons from flagged triples for validation output
            for f_item in flagged:
                reasons = f_item.get("_flag_reasons", [])
                if isinstance(reasons, list):
                    all_reasons.extend(reasons)
                elif isinstance(reasons, str):
                    all_reasons.append(reasons)

            if has_flagged:
                logger.warning(
                    "incremental_triples_inspection_flagged",
                    flagged_count=len(flagged),
                )

        return has_flagged, all_reasons, check_data

    def _get_cleaner_function(self):
        """Helper to safely resolve a clean_extracted_text callable from the tool registry or fallback."""
        if self.tool_executor:
            processor_tool = self.tool_executor.tool_registry.get("triple_processor")
            clean_extractor = getattr(processor_tool, "clean_extracted_text", None)
            if callable(clean_extractor):
                return clean_extractor
        return self.clean_extracted_text

    @staticmethod
    def clean_extracted_text(text: str) -> str:
        """Fallback method to remove leading and trailing invalid punctuation marks."""
        STRIP_CHARS = '《》""''「」『』【】（）()〈〉'
        if not text:
            return text
        return text.strip().strip(STRIP_CHARS).strip()

    async def execute_tool_workflow(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any]
    ) -> List[Message]:
        return []

    def build_synthesis_messages(
        self,
        payload: Dict[str, Any],
        plan_output: Dict[str, Any],
        tool_results: List[Message],
        agent: "BaseAgent",
    ) -> List[Message]:
        return []

    def parse_final_output(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any], raw_llm_output: str
    ) -> Dict[str, Any]:
        return plan_output
