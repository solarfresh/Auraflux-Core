from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import ValidationError

from auraflux_core.core.schemas.tools import ToolConfig
from auraflux_core.core.tools.base_tool import BaseTool
from auraflux_core.rag.config.regex_patterns import (
    DEFAULT_ANAPHORA_PATTERNS, DEFAULT_KEYWORD_PATTERNS,
    DEFAULT_TRIPLE_CHECKER_PATTERNS, AnaphoraPatternCollection,
    KeywordPatternCollection, TripleCheckerPatternCollection)
from auraflux_core.rag.schemas.chunker import TripleItem
from auraflux_core.rag.schemas.triple_processors import (
    FlaggedTripleItem, TripleRuleCheckerOutput)


class TripleProcessorTool(BaseTool):
    """
    Tool for cleaning, object-splitting, and metric normalizing structured semantic triples.
    """

    def __init__(
        self,
        patterns: KeywordPatternCollection = DEFAULT_KEYWORD_PATTERNS,
        config: ToolConfig = ToolConfig()
    ) -> None:
        super().__init__(config=config)
        self.patterns = patterns

    async def run(
        self,
        triples: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Executes triple cleaning, enumeration splitting, and metric formatting.
        Implements BaseTool.run() abstract method.
        """
        self.logger.info(f"Processing {len(triples)} semantic triples.")
        return self.process_triples(triples, patterns=self.patterns)

    def process_triples(
        self,
        triples: List[Dict[str, Any]],
        patterns: KeywordPatternCollection = DEFAULT_KEYWORD_PATTERNS
    ) -> List[Dict[str, Any]]:
        """
        Orchestrates triple processing: cleaning, object splitting, and metric formatting.
        """
        processed_triples: List[Dict[str, Any]] = []

        for item in triples:
            subj = self.clean_extracted_text(item.get("subject", ""))
            pred = self.clean_extracted_text(item.get("predicate", ""))
            obj_raw = self.clean_extracted_text(item.get("object", ""))

            if not (subj and pred and obj_raw):
                continue

            norm_val = self._normalize_metric_value(item.get("normalized_value"))
            split_objects = self._split_enumerated_object(obj_raw, patterns)

            for single_obj in split_objects:
                triple_dict = self._build_triple_dict(
                    subj=subj,
                    pred=pred,
                    obj=single_obj,
                    item=item,
                    norm_val=norm_val
                )
                processed_triples.append(triple_dict)

        return processed_triples

    def clean_extracted_text(self, text: str) -> str:
        """Removes leading and trailing invalid punctuation marks from a single string."""
        STRIP_CHARS = '《》""''「」『』【】（）()〈〉'
        if not text:
            return text
        return text.strip().strip(STRIP_CHARS).strip()

    def _split_enumerated_object(
        self,
        obj_raw: str,
        patterns: KeywordPatternCollection
    ) -> List[str]:
        primary_delimiter_split_chunks = patterns.explicit_delimiter_splitter.split(obj_raw)

        primary_split_objects = [
            cleaned
            for o in primary_delimiter_split_chunks
            if (cleaned := self.clean_extracted_text(o))
        ]

        final_split_objects: List[str] = []
        is_multi_item_enumeration = len(primary_split_objects) > 1

        for chunk in primary_split_objects:
            cleaned_chunk = self.clean_extracted_text(chunk)

            if is_multi_item_enumeration and patterns.contextual_enumeration_splitter.search(cleaned_chunk):
                secondary_chunks = patterns.contextual_enumeration_splitter.split(cleaned_chunk)
                final_split_objects.extend([
                    cleaned
                    for sub in secondary_chunks
                    if (cleaned := self.clean_extracted_text(sub))
                ])
            else:
                final_split_objects.append(cleaned_chunk)

        return final_split_objects

    def _normalize_metric_value(self, raw_val: Any) -> Optional[float]:
        if raw_val is None:
            return None
        try:
            return float(raw_val)
        except (ValueError, TypeError):
            return None

    def _build_triple_dict(
        self,
        subj: str,
        pred: str,
        obj: str,
        item: Dict[str, Any],
        norm_val: Optional[float]
    ) -> Dict[str, Any]:
        triple_dict: Dict[str, Any] = {
            "subject": subj,
            "subject_resolved": item.get("subject_resolved"),
            "predicate": pred,
            "object": obj,
            "data_target": item.get("data_target")
        }

        metric_name = item.get("metric_name")
        unit = item.get("unit")
        operator = item.get("operator", "<=")

        if metric_name and norm_val is not None:
            triple_dict.update({
                "metric_name": str(metric_name),
                "normalized_value": norm_val,
                "unit": str(unit) if unit else "",
                "operator": str(operator)
            })

        return triple_dict

    def get_name(self) -> str:
        """Returns the unique tool identifier for LLM tool invocation."""
        return "triple_processor"

    def get_description(self) -> str:
        """Returns the function description used by LLMs to determine tool routing."""
        return (
            "Cleans, splits enumerated objects, and formats metric metadata "
            "for extracted knowledge graph semantic triples."
        )

    def get_parameters(self) -> Dict[str, Any]:
        """Generates JSON Schema parameter specs for LLM function calling."""
        return {
            "type": "object",
            "properties": {
                "triples": {
                    "type": "array",
                    "description": "A list of raw triple dictionaries containing subject, predicate, and object.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                            "metric_name": {"type": "string"},
                            "normalized_value": {"type": "number"},
                            "unit": {"type": "string"},
                            "operator": {"type": "string"}
                        },
                        "required": ["subject", "predicate", "object"]
                    }
                }
            },
            "required": ["triples"]
        }


class TripleRuleCheckerTool(BaseTool):
    """
    Tool for inspecting extracted knowledge graph semantic triples using lightweight regex rules,
    filtering clean triples for ingestion and flagging flawed triples for reflection repair.
    """

    PREDICATE_MAX_WORDS = 6

    def __init__(
        self,
        checker_patterns: TripleCheckerPatternCollection = DEFAULT_TRIPLE_CHECKER_PATTERNS,
        anaphora_patterns: AnaphoraPatternCollection = DEFAULT_ANAPHORA_PATTERNS,
        config: ToolConfig = ToolConfig()
    ) -> None:
        super().__init__(config=config)
        self.checker_patterns = checker_patterns
        self.anaphora_patterns = anaphora_patterns

    async def run(
        self,
        triples: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes rule inspection on triples.
        Implements BaseTool.run() abstract method and converts the internal
        TripleRuleCheckerOutput Pydantic model into a JSON-serializable dictionary.
        """
        self.logger.info(f"Inspecting {len(triples)} semantic triples for rule violations.")
        inspection_result: TripleRuleCheckerOutput = self.inspect_triples(triples)
        return inspection_result.model_dump(by_alias=True)

    def inspect_triples(
        self, triples: List[Dict[str, Any]]
    ) -> TripleRuleCheckerOutput:
        """
        Orchestrates triple validation by delegating to filter_triples and
        returning a strongly-typed TripleRuleCheckerOutput model.

        Args:
            triples: A list of raw dictionary triples or TripleItem instances.

        Returns:
            TripleRuleCheckerOutput: Strongly-typed inspection result containing
            clean triples, flagged triples, and execution metrics.
        """
        clean_triples, flagged_triples = self.filter_triples(triples)

        return TripleRuleCheckerOutput(
            clean_triples=clean_triples,
            flagged_triples=flagged_triples,
            has_flagged=len(flagged_triples) > 0,
            total_count=len(triples),
            flagged_count=len(flagged_triples)
        )

    def inspect_single_triple(self, triple: Dict[str, Any]) -> List[str]:
        """
        Inspects a single triple using a clean, modular validation pipeline.
        """
        reasons: List[str] = []

        # 1. Schema & Pydantic Parsing Gate
        item = self._parse_and_validate_schema(triple, reasons)
        if not item:
            return reasons  # Stop early if schema/type validation fails

        # 2. Execution Pipeline of Semantic Rules
        reasons.extend(self._check_empty_nodes(item))
        reasons.extend(self._check_predicate_rules(item))
        reasons.extend(self._check_anaphora_rules(item))
        reasons.extend(self._check_operator_rules(item))
        reasons.extend(self._check_metric_rules(item))

        return reasons

    # =========================================================================
    # Modular Private Helper Validators
    # =========================================================================

    def _parse_and_validate_schema(self, triple: Dict[str, Any], reasons: List[str]) -> Optional[TripleItem]:
        """Parses raw dict into TripleItem, capturing any Pydantic validation errors."""
        try:
            return TripleItem.model_validate(triple)
        except ValidationError as e:
            for error in e.errors():
                loc = " -> ".join(str(x) for x in error["loc"])
                reasons.append(f"Schema validation error at [{loc}]: {error['msg']}")
            return None

    def _check_empty_nodes(self, item: TripleItem) -> List[str]:
        """Validates that subject and object are not empty."""
        subject = str(item.subject or "").strip()
        obj = str(item.object or "").strip()
        if not subject or not obj:
            return ["Subject or Object is empty."]
        return []

    def _check_predicate_rules(self, item: TripleItem) -> List[str]:
        """Validates predicate constraints (pronouns, weak verbs, length limit)."""
        reasons = []
        predicate = str(item.predicate or "").strip()

        if self.checker_patterns.pronoun_pattern.search(predicate):
            reasons.append(f"Predicate contains personal/possessive pronoun: '{predicate}'")

        if self.checker_patterns.weak_predicate_pattern.search(predicate):
            reasons.append(f"Predicate uses a weak verb phrase with embedded object: '{predicate}'")

        word_count = len(predicate.split())
        if word_count > self.PREDICATE_MAX_WORDS:
            reasons.append(f"Predicate is too long ({word_count} words): '{predicate}'")

        return reasons

    def _check_anaphora_rules(self, item: TripleItem) -> List[str]:
        """
        Validates anaphora resolution rules:
        - Ensures anaphoric pronouns MUST have 'subject_resolved'.
        - Flags unnecessary 'subject_resolved' provided for standard non-anaphoric subjects.
        """
        subject = str(item.subject or "").strip()
        subject_resolved = item.subject_resolved

        is_anaphoric = bool(self.anaphora_patterns.anaphora_reference_pattern.search(subject))
        has_resolved = subject_resolved is not None and str(subject_resolved).strip() != ""

        # Case 1: Anaphoric pronoun missing required resolution
        if is_anaphoric and not has_resolved:
            return [f"Subject contains unresolved anaphoric pronoun without subject_resolved: '{subject}'"]

        # Case 2: Standard non-anaphoric subject provided with subject_resolved
        if not is_anaphoric and has_resolved:
            return [
                f"Redundant subject_resolved provided for non-anaphoric subject: '{subject}'. "
                f"If '{subject}' is a standard noun, set 'subject_resolved' to null."
            ]

        return []

    def _check_operator_rules(self, item: TripleItem) -> List[str]:
        """Validates operator whitelist restrictions."""
        if item.operator is not None:
            operator_str = str(item.operator).strip()
            valid_operators = {"<=", ">=", "==", "<", ">"}
            if operator_str and operator_str not in valid_operators:
                return [f"Operator contains unauthorized value or invalid format: '{operator_str}'"]
        return []

    def _check_metric_rules(self, item: TripleItem) -> List[str]:
        """
        Validates metric consistency:
        1. Core Metric (metric_name, normalized_value) MUST be All-or-None.
        2. 'operator' is mandatory if metric data is present (forces prediction).
        3. 'unit' is optional, but if present, follows validation.
        """
        reasons: List[str] = []

        metric_name = item.metric_name
        normalized_value = item.normalized_value
        unit = item.unit
        operator = item.operator

        has_metric_name = metric_name is not None and str(metric_name).strip() != ""
        has_norm_val = normalized_value is not None
        has_unit = unit is not None and str(unit).strip() != ""
        has_operator = operator is not None and str(operator).strip() != ""

        # 1. Core Metric All-or-None Check (metric_name & normalized_value)
        core_states = [has_metric_name, has_norm_val]
        all_present = all(core_states)
        all_absent = not any(core_states)

        if not (all_present or all_absent):
            reasons.append(
                f"Inconsistent core metric: 'metric_name' and 'normalized_value' must "
                f"both be specified or both be null."
            )
            return reasons

        # 2. Orphan check: if core is absent, operator/unit should not be orphans
        if all_absent:
            if has_operator or has_unit:
                reasons.append(
                    f"Inconsistent metric data: 'operator' or 'unit' is specified, "
                    f"but mandatory core metric fields (metric_name, normalized_value) are null."
                )
            return reasons

        # 3. Strict Operator Enforcement (Forces LLM to Predict Operator when metric is present)
        if not has_operator:
            reasons.append(
                f"Missing operator: 'operator' cannot be null/blank when metric data is present. "
                f"You MUST predict an operator (e.g., '==' for exact point values)."
            )
            return reasons

        # 4. Operator Whitelist Check
        valid_operators = {"<=", ">=", "==", "<", ">"}
        operator_str = str(operator).strip()
        if operator_str not in valid_operators:
            reasons.append(
                f"Invalid operator value: '{operator_str}'. Must be one of {valid_operators}."
            )

        return reasons

    def filter_triples(
        self, triples: List[Dict[str, Any]]
    ) -> Tuple[List[TripleItem], List[FlaggedTripleItem]]:
        """
        Internal helper method to inspect and partition input triples into clean
        TripleItem instances and rule-violating FlaggedTripleItem instances safely.
        """
        clean_triples: List[TripleItem] = []
        flagged_triples: List[FlaggedTripleItem] = []

        for item in triples:
            reasons = self.inspect_single_triple(item)

            if reasons:
                flagged_data = {**item, "_flag_reasons": reasons}
                try:
                    flagged_triples.append(FlaggedTripleItem.model_validate(flagged_data))
                except ValidationError:
                    # Fallback for severely corrupted data (e.g., object is None) to prevent tool crash
                    safe_data = {
                        "subject": item.get("subject") if isinstance(item.get("subject"), str) else str(item.get("subject") or ""),
                        "predicate": item.get("predicate") if isinstance(item.get("predicate"), str) else str(item.get("predicate") or ""),
                        "object": item.get("object") if isinstance(item.get("object"), str) else str(item.get("object") or ""),
                        "subject_resolved": item.get("subject_resolved"),
                        "data_target": item.get("data_target"),
                        "metric_name": item.get("metric_name"),
                        "normalized_value": item.get("normalized_value"),
                        "unit": item.get("unit"),
                        "operator": item.get("operator"),
                        "_flag_reasons": reasons
                    }
                    flagged_triples.append(FlaggedTripleItem.model_construct(**safe_data))
            else:
                try:
                    clean_triples.append(TripleItem.model_validate(item))
                except ValidationError as e:
                    # Catch any unexpected pydantic validation error on clean path
                    schema_reasons = [f"Schema validation error at [{ ' -> '.join(str(x) for x in err['loc']) }]: {err['msg']}" for err in e.errors()]
                    flagged_data = {**item, "_flag_reasons": schema_reasons}
                    try:
                        flagged_triples.append(FlaggedTripleItem.model_validate(flagged_data))
                    except ValidationError:
                        safe_data = {
                            "subject": str(item.get("subject") or ""),
                            "predicate": str(item.get("predicate") or ""),
                            "object": str(item.get("object") or ""),
                            "_flag_reasons": schema_reasons
                        }
                        flagged_triples.append(FlaggedTripleItem.model_construct(**safe_data))

        return clean_triples, flagged_triples

    def get_name(self) -> str:
        """Returns the unique tool identifier for LLM tool invocation."""
        return "triple_rule_checker"

    def get_description(self) -> str:
        """Returns the function description used by LLMs to determine tool routing."""
        return (
            "Validates semantic triples against deterministic CJK and English syntax rules, "
            "flagging triples containing pronouns, weak verb phrases, excessive length, or unresolved anaphora."
        )

    def get_parameters(self) -> Dict[str, Any]:
        """Generates JSON Schema parameter specs fully aligned with TripleItem definition."""
        return {
            "type": "object",
            "properties": {
                "triples": {
                    "type": "array",
                    "description": "A list of bound semantic triple dictionaries containing entities, predicates, and metric metadata.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {
                                "type": "string",
                                "description": "Subject entity as verbatim from text."
                            },
                            "subject_resolved": {
                                "type": ["string", "null"],
                                "description": "Resolved actual entity name if subject is a pronoun/generic term."
                            },
                            "predicate": {
                                "type": "string",
                                "description": "Relation / Predicate / Operator verbatim from text."
                            },
                            "object": {
                                "type": "string",
                                "description": "Metric / Constraint / Object entity verbatim from text."
                            },
                            "data_target": {
                                "type": ["string", "null"],
                                "enum": ["subject", "object", None],
                                "description": "Field that contains the raw quantitative text span."
                            },
                            "metric_name": {
                                "type": ["string", "null"],
                                "description": "Standardized metric identifier (e.g., 'budget', 'duration', 'sla_time')."
                            },
                            "normalized_value": {
                                "type": ["number", "null"],
                                "description": "Pure numeric value converted strictly to the standard base unit."
                            },
                            "unit": {
                                "type": ["string", "null"],
                                "description": "Standard unit symbol (e.g., 'TWD', 'USD', 'day', 'hour', 'GB', '%')."
                            },
                            "operator": {
                                "type": ["string", "null"],
                                "enum": ["<=", ">=", "==", "<", ">", None],
                                "description": "Boundary condition operator."
                            }
                        },
                        "required": ["subject", "predicate", "object"]
                    }
                }
            },
            "required": ["triples"]
        }