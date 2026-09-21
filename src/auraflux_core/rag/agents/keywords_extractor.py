import json
from typing import Any, Dict, List, Optional, Tuple, cast

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

    def get_system_message_map(self) -> Dict[str, str]:
        """
        Provides default system messages aligned with the original keywords_extractor_2 prompt.
        """
        full_system_prompt = (
            "You are an expert Information Extraction Agent specializing in Knowledge Graph construction, Metric Normalization, and Coreference Resolution.\n"
            "Your task is to analyze raw document text and extract structured key information: Bound Semantic Triples (with embedded Metric Normalization & Coreference Resolution) and Exact Keywords/Entities.\n\n"
            "### EXTRACTION & REFINEMENT RULES\n\n"
            "1. **Semantic Triples (`triples`)**:\n"
            "   - Extract bound facts expressed as closed-world triples: `(subject, predicate, object)`.\n"
            "   - **EXACT TEXT MATCH REQUIRED**: `subject` and `object` MUST strictly use exact phrasing and terms present in the source text. Do NOT summarize or substitute words.\n"
            "   - `subject`: Core entity, concrete object, actor, parameter name, or descriptive phrase directly quoted from text.\n"
            "   - `subject_resolved`: (Coreference & Entity Resolution Rule)\n"
            "     - Ask: \"Does the verbatim `subject` string explicitly contain the concrete product, system, proper noun, or entity name?\"\n"
            "     - If NO (e.g., pronouns like 'Neither', 'It', or generic descriptive phrases like 'The most recent figure', 'The official benchmark', 'The company'), you MUST resolve and write the explicit entity name(s) here (e.g., 'Bun').\n"
            "     - If YES (e.g., 'Bun v1.0', 'Rust compiler'), set to null.\n"
            "   - `predicate`: The canonical action, relation, logical condition, or operator connecting subject and object.\n"
            "     - **PREDICATE CLEANLINESS & NO PRONOUNS (STRICT RULE)**:\n"
            "       * NEVER include second-person or conversational pronouns (e.g., 'your', 'our', 'you', 'me', 'us') inside the predicate.\n"
            "       * Keep predicates lean, standard active verbs, or relation phrases (e.g., use 'provides', 'ships to', 'supports').\n"
            "       * Do NOT embed target audiences or receivers inside the predicate. For example, if text states 'gives your agents X', simplify the predicate to 'provides' or 'gives' and attach target context to object (e.g., object: 'TypeScript SDK for AI agents').\n"
            "   - `object`: Target entity, metric, quantitative limit, or constrained value directly quoted from text.\n"
            "   - **FACTUAL RELATIONS ONLY**: Extract only explicit, concrete, operational, or logical relationships. Do NOT extract poetic metaphors, analogies, or rhetorical comparisons.\n"
            "   - **Length Limit**: Keep `subject`, `predicate`, and `object` concise (under 10 words each).\n"
            "   - **Quantitative Metrics Binding & Location (`data_target`)**:\n"
            "     - Whenever `normalized_value` is extracted, you MUST set `data_target`.\n"
            "     - Set `data_target: 'object'` if the raw numeric expression/quantity resides inside the `object` field.\n"
            "     - Set `data_target: 'subject'` if the raw numeric expression/quantity resides inside the `subject` field.\n"
            "     - Set all metric fields (`data_target`, `metric_name`, `normalized_value`, `unit`, `operator`) to null if no quantitative scale is present.\n"
            "     - **`operator` Strict Restriction**: If an operator or boundary condition applies, `operator` MUST strictly be one of the following comparison symbols: `\"<=\"`, `\">=\"`, `\"==\"`, `\"<\"`, `\">\"`, or `null`. **NEVER** use natural language words (e.g., do NOT use 'decrease', 'increase', 'between').\n"
            "   - **Multiple Quantitative Dimensions**: Extract SEPARATE triples for distinct measurable quantities.\n\n"
            "2. **Metric Normalization Principles (SI & Standard Base Units)**:\n"
            "   - Convert raw quantitative expressions into SI standard base units and full numeric scale (e.g., \"500萬\" -> 5000000.0).\n"
            "   - Currency: ISO 4217 (TWD, USD, EUR).\n"
            "   - Time: Convert long durations to 'day', short SLA times to 'hour' or 'second'.\n"
            "   - Percentage: Express as standard 0-100 values (e.g., \"85折\" -> 85.0).\n"
            "   - Standard Base Units: 'unit' MUST be a standardized metric unit (SI, ISO currency, '%', 'bytes', 'ms') or 'count'. Domain-specific entity nouns (e.g., \"crates\", \"users\") are forbidden as units; incorporate entities into 'metric_name' instead (e.g., metric_name: \"crate_count\", unit: \"count\").\n\n"
            "3. **Exact Keywords & Entities (`tags`)**:\n"
            "   - Extract ALL distinct key terms, proper nouns, domain concepts, and technical entities verbatim.\n"
            "   - **NO QUANTITY LIMIT**: Extract every valid entity present.\n\n"
            "4. **Empty / Low-Information Chunks**:\n"
            "   - Return empty lists for both triples and tags if text lacks substantive facts.\n\n"
            "### OUTPUT STRUCTURE SPECIFICATION\n"
            "Output JSON object containing:\n"
            "* `triples`: List of objects with keys: subject, subject_resolved, predicate, object, data_target, metric_name, normalized_value, unit, operator.\n"
            "* `tags`: List of exact strings."
        )

        return {
            "default": full_system_prompt,
            "en": full_system_prompt
        }

    def build_plan_messages(self, payload: Dict[str, Any]) -> List[Message]:
        """Stage 1: Constructs user messages for LLM keyword & triple extraction.

        Attempts dynamic lookup via `prompt_formatter.format_messages`. If absent or
        not callable, falls back to the default Stage 1 extraction user message.
        """
        formatter = getattr(self, "prompt_formatter", None)
        format_messages = getattr(formatter, "format_messages", None)
        if callable(format_messages):
            try:
                return cast(List[Message], format_messages(payload))
            except Exception as e:
                logger.warning(
                    "formatter_execution_failed",
                    formatter_class=formatter.__class__.__name__,
                    error_msg=str(e),
                )

        # --- Safe Fallback: User Prompt Only ---
        excerpt_text = payload.get("excerpt_text")

        user_content = (
            "[TASK: INITIAL EXTRACTION]\n\n"
            "Please analyze the following source text chunk and extract concise domain keywords (tags) "
            "and structured semantic triples (triples) according to the system extraction rules.\n\n"
            "### SOURCE TEXT CHUNK\n"
            f"{excerpt_text}\n\n"
            "### REQUIRED OUTPUT FORMAT (JSON)\n"
            "CRITICAL: Your output MUST be a SINGLE JSON OBJECT containing two root keys: 'triples' and 'tags'. "
            "DO NOT return a top-level JSON Array/List [].\n\n"
            "{\n"
            '  "triples": [\n'
            "    {\n"
            '      \"subject\": \"<exact text from chunk>\",\n'
            '      \"subject_resolved\": \"<resolved entity name or null>\",\n'
            '      \"predicate\": \"<action or relation verb>\",\n'
            '      \"object\": \"<target entity or value>\",\n'
            '      \"data_target\": \"<object / subject / null>\",\n'
            '      \"metric_name\": \"<metric name or null>\",\n'
            '      \"normalized_value\": <numeric value or null>,\n'
            '      \"unit\": \"<unit string or null>\",\n'
            '      \"operator\": \"<operator string or null>\"\n'
            "    }\n"
            "  ],\n"
            '  "tags": ["<keyword_1>", "<keyword_2>"]\n'
            "}"
        )

        return [
            Message(role="user", content=user_content, name=self.name)
        ]

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

    def build_plan_refinement_messages(
        self, payload: Dict[str, Any], flawed_output: Dict[str, Any], validation_result: Any
    ) -> Optional[List[Message]]:
        """
        Stage 1.5 Reflection Prompt Hook:
        Constructs refinement messages specifically targeting flagged triples for Pass 2 repair.
        If no flagged triples exist, returns None to bypass Pass 2 execution.
        """
        flagged_triples = flawed_output.get("flagged_triples", [])
        if not flagged_triples:
            return None

        # Extract raw input text chunk from payload
        excerpt_text = payload.get("excerpt_text")

        user_content = (
            "[TASK: REFLECTION & REFINEMENT]\n\n"
            "The previous extraction generated triple(s) that failed our Knowledge Graph quality inspection rules. "
            "Review the localized context text below and fix ONLY the flagged triple(s).\n\n"
            "### LOCALIZED CONTEXT TEXT\n"
            f"{excerpt_text}\n\n"
            "### FLAGGED TRIPLES TO REPAIR\n"
            "The following triples failed validation rules. Repair each item based on its violation reasons (`_flag_reasons`):\n"
            f"{flagged_triples}\n\n"
            "### REFINEMENT INSTRUCTIONS\n"
            "1. Fix each triple according to its specific `_flag_reasons`.\n"
            "2. Ensure `predicate` contains NO conversational pronouns (e.g., 'your', 'our') and NO hidden targets. Move target audiences/receivers to the `object` field.\n"
            "3. Maintain strict adherence to system extraction rules.\n\n"
            "### REQUIRED OUTPUT FORMAT (JSON)\n"
            "CRITICAL: Your output MUST be a SINGLE JSON OBJECT containing two root keys: 'triples' and 'tags'. "
            "DO NOT return a top-level JSON Array/List [].\n\n"
            "{\n"
            '  "triples": [\n'
            "    {\n"
            '      \"subject\": \"<exact text from chunk>\",\n'
            '      \"subject_resolved\": \"<resolved entity name or null>\",\n'
            '      \"predicate\": \"<action or relation verb>\",\n'
            '      \"object\": \"<target entity or value>\",\n'
            '      \"data_target\": \"<object / subject / null>\",\n'
            '      \"metric_name\": \"<metric name or null>\",\n'
            '      \"normalized_value\": <numeric value or null>,\n'
            '      \"unit\": \"<unit string or null>\",\n'
            '      \"operator\": \"<operator string or null>\"\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        return [
            Message(role="user", content=user_content, name=self.name),
        ]

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
                "excerpt_text": excerpt_text
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
            tool_args={"triples": processed_triples}
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
        self, payload: Dict[str, Any], plan_output: Dict[str, Any], tool_results: List[Message]
    ) -> List[Message]:
        return []

    def parse_final_output(
        self, payload: Dict[str, Any], plan_output: Dict[str, Any], raw_llm_output: str
    ) -> Dict[str, Any]:
        return plan_output
