from typing import Any, Dict

DEFAULT_EXTRACT_KEYWORDS_AGENT_SYSTEM_PROMPT = (
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

DEFAULT_EXTRACT_KEYWORDS_AGENT_PLAN_TEMPLATE = (
    "[TASK: INITIAL EXTRACTION]\n\n"
    "Please analyze the following source text chunk and extract concise domain keywords (tags) "
    "and structured semantic triples (triples) according to the system extraction rules.\n\n"
    "### SOURCE TEXT CHUNK\n"
    "{excerpt_text}\n\n"
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

DEFAULT_EXTRACT_KEYWORDS_AGENT_PLAN_STAGE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["triples", "tags"],
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "subject",
                    "subject_resolved",
                    "predicate",
                    "object",
                    "data_target",
                    "metric_name",
                    "normalized_value",
                    "unit",
                    "operator",
                ],
                "properties": {
                    "subject": {"type": "string"},
                    "subject_resolved": {"type": ["string", "null"]},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "data_target": {"type": ["string", "null"], "enum": ["object", "subject", None]},
                    "metric_name": {"type": ["string", "null"]},
                    "normalized_value": {"type": ["number", "null"]},
                    "unit": {"type": ["string", "null"]},
                    "operator": {
                        "type": ["string", "null"],
                        "enum": ["<=", ">=", "==", "<", ">", None],
                    },
                },
            },
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

DEFAULT_EXTRACT_KEYWORDS_AGENT_PLAN_REFINEMENT_TEMPLATE = (
    "[TASK: REFLECTION & REFINEMENT]\n\n"
    "The previous extraction generated triple(s) that failed our Knowledge Graph quality inspection rules. "
    "Review the localized context text below and fix ONLY the flagged triple(s).\n\n"
    "### LOCALIZED CONTEXT TEXT\n"
    "{excerpt_text}\n\n"
    "### FLAGGED TRIPLES TO REPAIR\n"
    "The following triples failed validation rules. Repair each item based on its violation reasons (`_flag_reasons`):\n"
    "{flagged_triples}\n\n"
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

DEFAULT_EXTRACT_KEYWORDS_AGENT_PLAN_REFINE_STAGE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["triples"],
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "subject",
                    "subject_resolved",
                    "predicate",
                    "object",
                    "data_target",
                    "metric_name",
                    "normalized_value",
                    "unit",
                    "operator",
                ],
                "properties": {
                    "subject": {"type": "string"},
                    "subject_resolved": {"type": ["string", "null"]},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "data_target": {"type": ["string", "null"], "enum": ["object", "subject", None]},
                    "metric_name": {"type": ["string", "null"]},
                    "normalized_value": {"type": ["number", "null"]},
                    "unit": {"type": ["string", "null"]},
                    "operator": {
                        "type": ["string", "null"],
                        "enum": ["<=", ">=", "==", "<", ">", None],
                    },
                },
            },
        }
    },
}
