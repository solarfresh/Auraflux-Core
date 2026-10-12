from typing import Any, Dict

DEFAULT_THRESHOLD_CLAIM_AGENT_SYSTEM_PROMPT_ZH = (
    "你是一名數據門檻與量化對焦稽核專家（Threshold & Quantitative Verification Specialist）。\n"
    "你的核心任務是針對傳入命題進行數據門檻與邊界衝突分析：\n"
    "1. 實施「單位標準化與量化診斷」，提取明確可稽核的數據指標（如金額、期限、SLA 等）與硬性條件邊界。\n"
    "2. 構建精準的事證檢索計畫，撈取相關基線條款。\n"
    "在驗證階段，你必須嚴格依據精確運算的結果進行事實導向的判決（VERIFIED / VIOLATED / PARTIALLY_VERIFIED / UNSUPPORTED），絕不允許主觀臆測。"
)

DEFAULT_THRESHOLD_CLAIM_AGENT_SYSTEM_PROMPT_EN = (
    "You are a Threshold & Quantitative Verification Specialist responsible for metric normalization "
    "and boundary checking of input claims.\n"
    "Your core responsibilities:\n"
    "1. Perform unit normalization and quantitative diagnostics to extract precise metrics and boundary criteria.\n"
    "2. Formulate targeted retrieval strategies to fetch relevant baseline policies or clauses.\n"
    "During verification, you must strictly adhere to exact mathematical computation results "
    "to render fact-based determinations (VERIFIED / VIOLATED / PARTIALLY_VERIFIED / UNSUPPORTED) without subjective hallucination."
)

DEFAULT_THRESHOLD_CLAIM_AGENT_PLAN_TEMPLATE = (
    "Proposition ID: {proposition_id}\n"
    "Claim: {claim_text}\n\n"
    "Task:\n"
    "1. Extract semantic triples (Subject -> Predicate -> Object).\n"
    "2. Perform quantitative threshold diagnosis:\n"
    "   - implicit_premises: Unstated baseline assumptions or environmental factors.\n"
    "   - quantification_requirements: Extract explicit metric targets and normalize them strictly according to NormalizedMetric schema.\n"
    "   - boundary_conflicts: Detect potential paradoxes or mathematical boundary conflicts.\n"
    "3. Formulate targeted retrieval queries (`queries`). Generate 1 to 2 distinct query items "
    "tailored to fetch baseline clauses or policy rules:\n"
    "   - `query_text`: The rewritten query string optimized for retrieval (MUST BE IN ENGLISH).\n"
    "   - `target_type`: Chosen strictly from ['question', 'concept_title', 'concept_desc', 'evidence'].\n"
    "   - `text_field`: Exactly ONE text field name from ['target_question', 'concept_title', 'concept_description', 'evidence_text'].\n"
    "   - `vector_field`: Exactly ONE vector field name from ['question_vector', 'concept_vector', 'evidence_vector'].\n\n"
    "CRITICAL FORMAT RULES:\n"
    "- `query_text` MUST BE IN ENGLISH regardless of the input claim language.\n"
    "- `quantification_requirements` MUST BE A LIST of NormalizedMetric objects.\n"
    "- `queries` MUST BE A NON-EMPTY ARRAY containing 1 or 2 targeted query objects.\n\n"
    "Required Output JSON Format:\n"
    "{{\n"
    '  "triples": [\n'
    '    {{"subject": "...", "predicate": "...", "object": "..."}}\n'
    '  ],\n'
    '  "diagnostics": {{\n'
    '    "implicit_premises": ["..."],\n'
    '    "quantification_requirements": [\n'
    '      {{\n'
    '        "raw_text": "500萬TWD",\n'
    '        "metric_name": "budget",\n'
    '        "normalized_value": 5000000.0,\n'
    '        "unit": "TWD",\n'
    '        "operator": "<="\n'
    '      }}\n'
    '    ],\n'
    '    "boundary_conflicts": {{\n'
    '      "has_conflict": false\n'
    '    }}\n'
    '  }},\n'
    '  "queries": [\n'
    '    {{\n'
    '      "target_type": "evidence",\n'
    '      "query_text": "...",\n'
    '      "text_field": "evidence_text",\n'
    '      "vector_field": "evidence_vector"\n'
    '    }}\n'
    '  ]\n'
    "}}"
)

DEFAULT_THRESHOLD_CLAIM_AGENT_PLAN_STAGE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["triples", "diagnostics", "queries"],
    "properties": {
        "triples": {
            "type": "array",
            "description": "Extracted semantic triples.",
            "items": {
                "type": "object",
                "required": ["subject", "predicate", "object"],
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                },
            },
        },
        "diagnostics": {
            "type": "object",
            "required": [
                "implicit_premises",
                "quantification_requirements",
                "boundary_conflicts",
            ],
            "properties": {
                "implicit_premises": {
                    "type": "object",
                    "required": [
                        "capability_assumptions",
                        "resource_assumptions",
                        "causality_assumptions",
                        "environment_assumptions",
                    ],
                    "properties": {
                        "capability_assumptions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "resource_assumptions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "causality_assumptions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "environment_assumptions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
                "quantification_requirements": {
                    "type": "array",
                    "description": "List of quantitative metrics and requirements.",
                    "items": {
                        "type": "object",
                        "required": [
                            "raw_text",
                            "metric_name",
                            "normalized_value",
                            "unit",
                            "operator",
                        ],
                        "properties": {
                            "raw_text": {"type": "string"},
                            "metric_name": {"type": "string"},
                            "normalized_value": {"type": "number"},
                            "unit": {"type": "string"},
                            "operator": {"type": "string"},
                        },
                    },
                },
                "boundary_conflicts": {
                    "type": "object",
                    "required": ["has_conflict"],
                    "properties": {
                        "has_conflict": {"type": "boolean"},
                        "description": {"type": "string"},
                    },
                },
            },
        },
        "queries": {
            "type": "array",
            "minItems": 1,
            "maxItems": 2,
            "items": {
                "type": "object",
                "required": [
                    "target_type",
                    "query_text",
                    "text_field",
                    "vector_field",
                ],
                "properties": {
                    "target_type": {
                        "type": "string",
                        "enum": [
                            "question",
                            "concept_title",
                            "concept_desc",
                            "evidence",
                        ],
                    },
                    "query_text": {
                        "type": "string",
                        "description": "Rewritten query string in English.",
                    },
                    "text_field": {
                        "type": "string",
                        "enum": [
                            "target_question",
                            "concept_title",
                            "concept_description",
                            "evidence_text",
                        ],
                    },
                    "vector_field": {
                        "type": "string",
                        "enum": [
                            "question_vector",
                            "concept_vector",
                            "evidence_vector",
                        ],
                    },
                },
            },
        },
    },
}

DEFAULT_THRESHOLD_CLAIM_AGENT_SYNTHESIS_TEMPLATE = (
    "Proposition ID: {proposition_id}\n"
    "Claim: {claim_text}\n"
    "Stage 1 Quantification Requirements: {quantification_requirements}\n\n"
    "Deterministic Calculator Computation Result (HARD GROUND TRUTH):\n{calc_summary}\n\n"
    "Retrieved Baseline Policy Context:\n{formatted_evidence_text}\n\n"
    "Task:\n"
    "1. Evaluate claim against the Deterministic Calculator Computation Result.\n"
    "2. YOU MUST STRICTLY ADHERE to the calculator result status:\n"
    "   - 'VERIFIED': Fully met boundary requirements and supported by baseline metrics.\n"
    "   - 'PARTIALLY_VERIFIED': Partially satisfied or has minor boundary ambiguity.\n"
    "   - 'VIOLATED': Direct boundary conflict or out-of-bounds violation.\n"
    "   - 'UNSUPPORTED': Insufficient baseline metric data to verify.\n"
    "3. If status is 'PARTIALLY_VERIFIED' or 'UNSUPPORTED', provide trade-off choices or next steps in `preset_options`.\n"
    "4. Fill `compliance_gap` with detailed explanation of the numeric paradox, violation, or lack of data; use null if VERIFIED.\n\n"
    "Required Output JSON Format:\n"
    "{{\n"
    '  "status": "VERIFIED" | "PARTIALLY_VERIFIED" | "VIOLATED" | "UNSUPPORTED",\n'
    '  "preset_options": ["Option 1...", "Option 2..."],\n'
    '  "compliance_gap": "Detailed explanation string or null",\n'
    '  "verification_proofs": ["Location or clause reference..."]\n'
    "}}"
)

DEFAULT_THRESHOLD_CLAIM_AGENT_SYNTHESIS_STAGE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": [
        "diagnostic_evaluation",
        "status",
        "preset_options",
        "verification_proofs",
        "compliance_gap",
    ],
    "properties": {
        "diagnostic_evaluation": {
            "type": "object",
            "description": "Evaluation of each diagnostic dimension against retrieved evidence.",
            "required": [
                "capability_eval",
                "resource_eval",
                "causality_eval",
                "environment_eval",
                "quantification_eval",
                "boundary_conflicts_eval",
            ],
            "properties": {
                "capability_eval": {
                    "type": "string",
                    "description": "Evaluation on capability assumption.",
                },
                "resource_eval": {
                    "type": "string",
                    "description": "Evaluation on resource/authorization assumption.",
                },
                "causality_eval": {
                    "type": "string",
                    "description": "Evaluation on assumed causal chain.",
                },
                "environment_eval": {
                    "type": "string",
                    "description": "Evaluation on external environment stability.",
                },
                "quantification_eval": {
                    "type": "string",
                    "description": "Evaluation on whether vague expectations are now quantifiable.",
                },
                "boundary_conflicts_eval": {
                    "type": "string",
                    "description": "Evaluation on potential policy or constraint conflicts.",
                },
            },
        },
        "status": {
            "type": "string",
            "enum": ["VERIFIED", "PARTIALLY_VERIFIED", "VIOLATED", "UNSUPPORTED"],
            "description": "Final synthesis alignment status.",
        },
        "preset_options": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Preset resolution or alternative options.",
        },
        "verification_proofs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Exact supporting quotes, locations, or triples from evidence.",
        },
        "compliance_gap": {
            "type": ["string", "null"],
            "description": "Summary of gap/unresolved assumption. MUST be null if status is VERIFIED, or string if VIOLATED/UNSUPPORTED.",
        },
    },
}
