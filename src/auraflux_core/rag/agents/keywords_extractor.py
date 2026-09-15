from typing import Any, Dict, List, Optional, Tuple, cast

from auraflux_core.core.agents.base_agent import BaseAgent
from auraflux_core.core.agents.pipelines.plan_and_execute import \
    PlanAndExecuteHandler
from auraflux_core.core.schemas.messages import Message
from auraflux_core.core.schemas.pipelines import ValidationResult


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
            "   - **Multiple Quantitative Dimensions**: Extract SEPARATE triples for distinct measurable quantities.\n\n"
            "2. **Metric Normalization Principles (SI & Standard Base Units)**:\n"
            "   - Convert raw quantitative expressions into SI standard base units and full numeric scale (e.g., \"500萬\" -> 5000000.0).\n"
            "   - Currency: ISO 4217 (TWD, USD, EUR).\n"
            "   - Time: Convert long durations to 'day', short SLA times to 'hour' or 'second'.\n"
            "   - Percentage: Express as standard 0-100 values (e.g., \"85折\" -> 85.0).\n\n"
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
            return cast(List[Message], format_messages(payload))

        # --- Safe Fallback: User Prompt Only ---
        chunk_text = payload.get("chunk_text") or payload.get("text", "")

        user_content = (
            "[TASK: INITIAL EXTRACTION]\n\n"
            "Please analyze the following source text chunk and extract concise domain keywords (tags) "
            "and structured semantic triples (triples) according to the system extraction rules.\n\n"
            "### SOURCE TEXT CHUNK\n"
            f"{chunk_text}"
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

            # Step 1: Execute incremental triple processing & rule inspection
            if "triples" in plan_output and isinstance(plan_output["triples"], list) and self.tool_executor:
                has_flagged, flagged_reasons = await self._process_incremental_triples(plan_output)

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
                    reasons=flagged_reasons
                )

            return ValidationResult(is_valid=True, reasons=[])

        except Exception as e:
            self.logger.error(f"Error in inspect_plan_output: {str(e)}")
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
        # Fixed Key: Updated from "_flagged_triples" to "flagged_triples" to match TripleRuleCheckerOutput schema
        flagged_triples = flawed_output.get("flagged_triples", [])
        if not flagged_triples:
            return None

        # Extract raw input text chunk from payload
        chunk_text = payload.get("chunk_text") or payload.get("text", "")

        user_content = (
            "[TASK: REFLECTION & REFINEMENT]\n\n"
            "The previous extraction generated triple(s) that failed our Knowledge Graph quality inspection rules. "
            "Review the localized context text below and fix ONLY the flagged triple(s).\n\n"
            "### LOCALIZED CONTEXT TEXT\n"
            f"{chunk_text}\n\n"
            "### FLAGGED TRIPLES TO REPAIR\n"
            "The following triples failed validation rules. Repair each item based on its violation reasons (`_flag_reasons`):\n"
            f"{flagged_triples}\n\n"
            "### REFINEMENT INSTRUCTIONS\n"
            "1. Fix each triple according to its specific `_flag_reasons`.\n"
            "2. Ensure `predicate` contains NO conversational pronouns (e.g., 'your', 'our') and NO hidden targets. Move target audiences/receivers to the `object` field.\n"
            "3. Maintain strict adherence to system extraction rules.\n"
            "4. Output a valid JSON object containing ONLY the corrected `triples` list."
        )

        return [
            Message(role="user", content=user_content, name=self.name),
        ]

    async def _process_incremental_triples(self, plan_output: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Helper method dedicated to handling incremental triple processing and revision.
        Cleans incoming triples, inspects rule violations via TripleRuleCheckerTool,
        updates resolved fields by key, and safely appends newly passed clean triples.

        Returns:
            Tuple[bool, List[str]]: (has_flagged, combined_flag_reasons)
        """
        raw_incoming_triples = plan_output.get("triples", [])

        # Step 1: Clean and split enumerated objects via triple_processor
        processed_res = await self.tool_executor.run(
            tool_name="triple_processor",
            tool_args={"triples": raw_incoming_triples}
        )
        processed_triples = (
            processed_res.content
            if hasattr(processed_res, "content")
            else processed_res
        )

        # Step 2: Inspect candidate triples via triple_rule_checker
        check_res = await self.tool_executor.run(
            tool_name="triple_rule_checker",
            tool_args={"triples": processed_triples}
        )
        check_data = (
            check_res.content
            if hasattr(check_res, "content")
            else check_res
        )

        has_flagged = False
        all_reasons: List[str] = []

        if isinstance(check_data, dict):
            new_clean = check_data.get("clean_triples", [])
            flagged = check_data.get("flagged_triples", [])
            has_flagged = check_data.get("has_flagged", False)

            # Step 3: Deduplicate and update state using pure (subject, predicate, object) identity keys
            accumulated_clean: List[Dict[str, Any]] = plan_output.get("_accumulated_clean_triples", [])

            # Map triple key to list index for in-place updates
            key_to_index = {
                (t.get("subject"), t.get("predicate"), t.get("object")): idx
                for idx, t in enumerate(accumulated_clean)
            }

            for item in new_clean:
                key = (item.get("subject"), item.get("predicate"), item.get("object"))
                if key in key_to_index:
                    # If key exists, overwrite with the latest refined version
                    idx = key_to_index[key]
                    accumulated_clean[idx] = item
                else:
                    # Append new unique triple and record index
                    key_to_index[key] = len(accumulated_clean)
                    accumulated_clean.append(item)

            # Aggregate all _flag_reasons from flagged triples for validation output
            for f_item in flagged:
                reasons = f_item.get("_flag_reasons", [])
                if isinstance(reasons, list):
                    all_reasons.extend(reasons)

            # Update internal accumulation state and write back to plan output
            plan_output["_accumulated_clean_triples"] = accumulated_clean
            plan_output["triples"] = accumulated_clean
            plan_output["_flagged_triples"] = flagged

            if has_flagged:
                self.logger.warning(
                    f"Incremental Inspection: Flagged {len(flagged)} flawed triples. "
                    f"Total deduplicated clean triples: {len(accumulated_clean)}."
                )

        return has_flagged, all_reasons

    def _get_cleaner_function(self):
        """Helper to safely resolve a clean_extracted_text callable from the tool registry or fallback.

        Some registry entries may be BaseTool objects/classes without a typed
        clean_extracted_text attribute. In that case, we must not try to access the
        attribute directly, otherwise static type-checkers / language servers report
        that the attribute is unknown on BaseTool.
        """
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
