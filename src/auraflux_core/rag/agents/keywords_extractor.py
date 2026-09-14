import json
from typing import Any, Dict, List, Optional

from auraflux_core.core.agents.base_agent import BaseAgent
from auraflux_core.rag.config.regex_patterns import (DEFAULT_KEYWORD_PATTERNS,
                                                     KeywordPatternCollection)


class ExtractKeywordsAgent(BaseAgent):

    def get_system_message_map(self) -> Dict[str, str]:
        return {
            "default": (
                "You are an expert Information Extraction Agent specializing in Knowledge Graph construction, Metric Normalization, and Document Analysis.\n"
                "Your task is to analyze the provided raw document chunk text and extract structured key information: Bound Semantic Triples (with embedded Metric Normalization & Coreference Resolution) and Exact Keywords/Entities.\n\n"
                "### EXTRACTION RULES\n\n"
                "1. **Semantic Triples (`triples`)**:\n"
                "   - Extract bound facts expressed as closed-world triples: `(subject, predicate, object)`.\n"
                "   - **EXACT TEXT MATCH REQUIRED**: `subject`, `predicate`, and `object` MUST strictly use the exact phrasing, exact terms, and wording present in the source text. Do NOT paraphrase, summarize, or translate any of the words into synonyms.\n"
                "   - `subject`: Core entity, concrete object, actor, parameter name, or descriptive phrase directly quoted from text.\n"
                "   - `subject_resolved`: (Coreference & Entity Resolution Rule)\n"
                "     - Ask: \"Does the verbatim `subject` string explicitly contain the concrete product, system, proper noun, or entity name?\"\n"
                "     - If NO (e.g., pronouns like 'Neither', 'It', or generic descriptive phrases like 'The most recent figure', 'The official benchmark', 'The company'), you MUST resolve and write the explicit entity name(s) here (e.g., 'Bun').\n"
                "     - If YES (e.g., 'Bun v1.0', 'Rust compiler'), set to null.\n"
                "   - `predicate`: The action, relation, logical condition, or operator connecting subject and object as verbatim from text.\n"
                "   - `object`: Target entity, metric, quantitative limit, or constrained value directly quoted from text.\n"
                "   - **FACTUAL RELATIONS ONLY**: Extract only explicit, concrete, operational, or logical relationships. Do NOT extract poetic metaphors, analogies, or rhetorical comparisons (e.g., skip statements like 'X is like Y learning to drive').\n"
                "   - **Length Limit**: Keep `subject`, `predicate`, and `object` concise (under 10 words each). Do NOT insert entire sentences into a triple field.\n"
                "   - **Quantitative Metrics Binding & Location (`data_target`)**:\n"
                "     - Whenever `normalized_value` is extracted, you MUST set `data_target`.\n"
                "     - **Location Matching Rule**: Indicate which field contains the raw quantitative text span from the source text:\n"
                "       - Set `data_target: 'object'` if the raw numeric expression/quantity resides inside the `object` field.\n"
                "       - Set `data_target: 'subject'` if the raw numeric expression/quantity resides inside the `subject` field.\n"
                "   - **Multiple Quantitative Dimensions**: If a single statement expresses more than one distinct measurable quantity (e.g., a count AND a duration, a price AND a deadline), you MUST extract a SEPARATE triple for each quantity. Each triple's `object` must be the exact text span describing that single quantity only, and its `metric_name`/`normalized_value`/`unit`/`operator` must correspond to that same quantity — never let one triple's metric fields absorb a number that belongs to a different quantity in the same sentence.\n"
                "     Example: \"Bun bundles 10,000 React components in 269ms.\" must produce TWO triples:\n"
                "       1. (subject: \"Bun\", subject_resolved: null, predicate: \"bundles\", object: \"10,000 React components\", data_target: \"object\", metric_name: \"component_count\", normalized_value: 10000.0, unit: \"count\", operator: \"==\")\n"
                "       2. (subject: \"Bun\", subject_resolved: null, predicate: \"bundles ... in\", object: \"269ms\", data_target: \"object\", metric_name: \"bundling_duration\", normalized_value: 0.269, unit: \"second\", operator: \"<=\")\n\n"
                "2. **Metric Normalization Principles (SI & Standard Base Units)**:\n"
                "   When extracting optional normalized metric fields (`normalized_value` and `unit`), convert raw quantitative expressions into SI standard base units and full numeric scale:\n"
                "   - **EXPLICIT NUMERIC VALUES ONLY**: Only extract `normalized_value` for concrete numeric expressions (e.g., \"2 million\" -> 2000000.0, \"500萬\" -> 5000000.0, \"$2.5K\" -> 2500.0). Do NOT normalize fuzzy or indefinite quantities (e.g., \"millions of\", \"thousands of\", \"several\", \"many\"); set `normalized_value`, `unit`, `operator`, and `data_target` to null for these fuzzy descriptions.\n"
                "   - **Multipliers**: Expand all scale prefixes ('萬', '億', 'k', 'm', 'b') into full floating-point numbers (e.g., \"500萬\" -> 5000000.0; \"$2.5K\" -> 2500.0).\n"
                "   - **Currency**: Normalize unit to standard ISO 4217 currency codes (e.g., 'TWD', 'USD', 'EUR', 'JPY').\n"
                "   - **Time & Duration**: Convert long durations (weeks/months/years) strictly to **'day'** (1 week = 7, 1 month = 30, 1 year = 365). Convert short response/SLA times (minutes/hours) to **'hour'** or **'second'**.\n"
                "   - **Length & Area**: Convert local or imperial units (坪, sq ft, cm, mm) to SI base units ('m', 'm2').\n"
                "   - **Mass & Volume**: Convert to SI base units ('kg', 'L', 'm3').\n"
                "   - **Data & Network**: Convert TB/KB to 'GB'; Gbps/Kbps to 'Mbps'.\n"
                "   - **Percentage & Ratio**: Express directly as standard percentage values 0-100 (e.g., \"85折\" -> value: 85.0, unit: \"%\"; \"50 bp\" -> value: 0.5, unit: \"%\").\n\n"
                "3. **Exact Keywords & Entities (`tags`)**:\n"
                "   - Extract ALL distinct key terms, proper nouns, domain concepts, and technical entities directly quoted from the source text.\n"
                "   - **NO QUANTITY LIMIT**: Extract every valid entity present; do not cap or artificially limit the count.\n"
                "   - **EXACT TEXT MATCH REQUIRED**: Every item MUST be a verbatim string copied directly from the text (preserving original casing, spelling, and hyphenation).\n\n"
                "   **Eligible Entity Categories**:\n"
                "   - **Proper Nouns & Entities**: Names of people, organizations, standards, locations, or products (e.g., \"John Doe\", \"Acme Corp\", \"ISO 27001\", \"Project Apollo\").\n"
                "   - **Domains & Frameworks**: Field titles, academic theories, or overall domain names (e.g., \"Cybersecurity\", \"Zero Trust Architecture\", \"Agile Methodology\").\n"
                "   - **Technical Terms & Methodologies**: Specific algorithms, architecture components, or technical mechanisms (e.g., \"AES-256\", \"Load Balancer\", \"Gradient Descent\").\n"
                "   - **Key Operational Concepts**: Core terms, parameters, or policies explicitly discussed in context (e.g., \"Retention Period\", \"Authentication Factor\", \"Throughput Limit\").\n\n"
                "   **Exclusion Rules**:\n"
                "   - Exclude generic common nouns lacking unique semantic context (e.g., \"document\", \"item\", \"example\", \"result\").\n"
                "   - Do NOT split multi-word technical concepts or titles into separated atomic words (e.g., extract \"Zero Trust Architecture\", NOT just \"Architecture\").\n\n"
                "4. **Empty / Low-Information Chunks**:\n"
                "   - If the input text consists solely of headers, layout metadata, conversational transitions, or lacks substantive facts, return empty structures for both triples and tags.\n"
                "   - Do NOT force extraction if no explicit entities or domain keywords exist in the text.\n\n"
                "### OUTPUT STRUCTURE SPECIFICATION\n\n"
                "The output contains two primary fields:\n\n"
                "* **`triples`**: A list of structured items capturing precise semantic relationships using verbatim text, enriched with normalized metrics when applicable.\n"
                "  - `subject`: (String) Exact source entity or phrase appearing in the text.\n"
                "  - `subject_resolved`: (Optional String / Null) Explicit entity name if `subject` lacks the concrete proper noun/entity name. Set to null if `subject` already contains the explicit entity name.\n"
                "  - `predicate`: (String) Exact linking operator or relationship phrase appearing in the text.\n"
                "  - `object`: (String) Exact target entity or constraint value appearing in the text.\n"
                "  - `data_target`: (Optional String / Null) Strictly choose from ['subject', 'object', null]. Must be set whenever `normalized_value` is present, indicating which field contains the raw quantitative text span.\n"
                "  - `metric_name`: (Optional String / Null) Standardized metric identifier representing the core semantic concept (e.g., 'budget', 'duration', 'sla_time', 'penalty', 'area'). Set to null if non-quantitative or fuzzy.\n"
                "  - `normalized_value`: (Optional Float / Null) Pure numeric value converted strictly to the standard base unit. Set to null if non-quantitative or fuzzy.\n"
                "  - `unit`: (Optional String / Null) Standard unit symbol (e.g., 'TWD', 'USD', 'day', 'hour', 'm2', 'kg', 'GB', '%'). Set to null if non-quantitative or fuzzy.\n"
                "  - `operator`: (Optional String / Null) Boundary condition operator strictly chosen from ['<=', '>=', '==', '<', '>']. Defaults to '<=' for limits or limits of budget/time. Set to null if non-quantitative or fuzzy.\n"
                "* **`tags`**: (List of Strings) An unconstrained list of exact key terms, proper nouns, domain concepts, and technical entities extracted directly from the text as verbatim strings.\n"
            )
        }

    def postprocess_output(self, output_string: str) -> str:
        """Processes and cleans the semantic data structure returned by the Agent."""
        json_object = self.output_parser.parse_json(output_string)

        # 1. Clean tags
        if "tags" in json_object and isinstance(json_object["tags"], list):
            json_object["tags"] = [
                cleaned
                for tag in json_object["tags"]
                if (cleaned := self.clean_extracted_text(tag))
            ]

        # 2. Clean triples (subject, predicate, object)
        if "triples" in json_object and isinstance(json_object["triples"], list):
            json_object["triples"] = self.process_triples(json_object["triples"])

        return json.dumps(json_object, ensure_ascii=False)

    def clean_extracted_text(self, text: str) -> str:
        """Removes leading and trailing invalid punctuation marks from a single string."""
        # Define quote brackets and punctuation to be stripped (includes Chinese title brackets, standard quotes, etc.)
        STRIP_CHARS = '《》""''「」『』【】（）()〈〉'
        if not text:
            return text
        # Strip whitespace and specified punctuation marks from both ends
        cleaned = text.strip().strip(STRIP_CHARS).strip()
        return cleaned

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
        """
        Safely converts raw metric value to float.
        Responsibility: Metric Normalization Only.
        """
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
        """
        Constructs the final dictionary representation of a single triple.
        Responsibility: Schema / Dictionary Assembly Only.
        """
        triple_dict: Dict[str, Any] = {
            "subject": subj,
            "predicate": pred,
            "object": obj
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

    def process_triples(
        self,
        triples: List[Dict[str, Any]],
        patterns: KeywordPatternCollection = DEFAULT_KEYWORD_PATTERNS
    ) -> List[Dict[str, Any]]:
        """
        Orchestrates triple processing: cleaning, object splitting, and metric formatting.
        Responsibility: High-level Flow Orchestration.
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
