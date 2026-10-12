from typing import Any, Dict

DEFAULT_OBJECTIVE_CLAIM_AGENT_SYSTEM_PROMPT_ZH = (
    "你是一名客觀聲明稽核專家（Objective Claim Audit Specialist）。\n"
    "你的核心任務是針對傳入命題進行正交診斷分析（Orthogonal Diagnostics）：\n"
    "1. 剖析隱性前提（資源授權、團隊能力與因果依賴）。\n"
    "2. 實施「標準與量化診斷」，剝離主觀形容與抽象描述，提取明確可稽核的數據指標與所需產物類型（Required Artifact Types, 如 Log、 Commit、驗收報告）。\n"
    "3. 檢測足跡衝突（Footprint Conflicts）並構建精準的事證檢索計畫。\n"
    "在驗證階段，你必須嚴格基於檢索到的客觀佐證進行事實導向的判決（VERIFIED / PARTIALLY_VERIFIED / UNSUPPORTED），絕不允許主觀臆測。"
)

DEFAULT_OBJECTIVE_CLAIM_AGENT_SYSTEM_PROMPT_EN = (
    "You are an Objective Claim Audit Specialist responsible for orthogonal diagnostic analysis "
    "and evidence-based verification of input claims.\n"
    "Your core responsibilities:\n"
    "1. Analyze implicit premises (resource authorization, capability assumptions, and causality).\n"
    "2. Execute standardization and quantification diagnostics to strip subjective phrasing and extract audit-ready metrics alongside required artifact types (e.g., logs, commits, spec docs).\n"
    "3. Detect footprint conflicts and formulate targeted evidence retrieval strategies.\n"
    "During verification, you must strictly evaluate claims against retrieved objective evidence "
    "to render fact-based determinations (VERIFIED / PARTIALLY_VERIFIED / UNSUPPORTED) without subjective hallucination."
)

DEFAULT_OBJECTIVE_CLAIM_AGENT_PLAN_TEMPLATE = (
    "Proposition ID: {proposition_id}\n"
    "Claim: {claim_text}\n\n"
    "Task:\n"
    "1. Extract semantic triples (Subject -> Predicate -> Object).\n"
    "2. Perform orthogonal diagnosis:\n"
    "   - implicit_premises: Unstated assumptions regarding Capability, Resources, Causality, or Environment.\n"
    "   - quantification_requirements: Clear metrics, required artifact types, and acceptance criteria.\n"
    "   - boundary_conflicts: Potential resource or rule conflicts.\n"
    "3. Formulate targeted retrieval queries (`queries`). Generate 1 to 2 distinct query items "
    "tailored to specific search intents. Each item must strictly bind to ONE text field and ONE vector field:\n"
    "   - `query_text`: The rewritten query string optimized for retrieval (MUST BE IN ENGLISH to match knowledge base).\n"
    "   - `target_type`: The retrieval intent, chosen strictly from ['question', 'concept_title', 'concept_desc', 'evidence']:\n"
    "       * 'question': Target underlying core questions. (text_field: 'target_question', vector_field: 'question_vector')\n"
    "       * 'concept_title': Target specific concept names/terms. (text_field: 'concept_title', vector_field: 'concept_vector')\n"
    "       * 'concept_desc': Target concept definitions and mechanisms. (text_field: 'concept_description', vector_field: 'concept_vector')\n"
    "       * 'evidence': Target concrete empirical facts or evidence. (text_field: 'evidence_text', vector_field: 'evidence_vector')\n"
    "   - `text_field`: Exactly ONE text field name from ['target_question', 'concept_title', 'concept_description', 'evidence_text'].\n"
    "   - `vector_field`: Exactly ONE vector field name from ['question_vector', 'concept_vector', 'evidence_vector'].\n\n"
    "CRITICAL FORMAT RULES:\n"
    "- `query_text` MUST BE IN ENGLISH regardless of the input claim language.\n"
    "- `quantification_requirements` MUST BE A DICTIONARY OBJECT (NOT A LIST/ARRAY).\n"
    "- `queries` MUST BE A NON-EMPTY ARRAY containing 1 or 2 targeted query objects.\n\n"
    "Required Output JSON Format:\n"
    "{{\n"
    '  "triples": [\n'
    '    {{"subject": "...", "predicate": "...", "object": "..."}}\n'
    '  ],\n'
    '  "diagnostics": {{\n'
    '    "implicit_premises": ["..."],\n'
    '    "quantification_requirements": {{\n'
    '      "required_artifact_types": ["..."],\n'
    '      "acceptance_criteria": "..."\n'
    '    }},\n'
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

DEFAULT_OBJECTIVE_CLAIM_AGENT_PLAN_STAGE_OUTPUT_SCHEMA: Dict[str, Any] = {
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
                    "type": "array",
                    "description": "Unstated assumptions regarding Capability, Resources, Causality, or Environment.",
                    "items": {"type": "string"},
                },
                "quantification_requirements": {
                    "type": "object",
                    "description": "Clear metrics, required artifact types, and acceptance criteria.",
                    "required": ["required_artifact_types", "acceptance_criteria"],
                    "properties": {
                        "required_artifact_types": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "acceptance_criteria": {"type": "string"},
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
                        "description": "Rewritten query string optimized for retrieval in English.",
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

DEFAULT_OBJECTIVE_CLAIM_AGENT_SYNTHESIS_TEMPLATE = (
    "Proposition ID: {proposition_id}\n"
    "Claim: {claim_text}\n"
    "Extracted Claim Triples (Stage 1): {triples}\n"
    "Diagnostics (Stage 1): {diagnostics}\n\n"
    "Retrieved Structured Core Context Evidence:\n{formatted_evidence_text}\n\n"
    "Task:\n"
    "1. Perform a thorough cross-examination of the Claim and Stage 1 Diagnostics against the Retrieved Evidence.\n"
    "2. Evaluate and EXPLICITLY OUTPUT the `diagnostic_evaluation` for each diagnostic item:\n"
    "   - `implicit_premises_eval`: Assess if the unstated assumptions are supported by evidence.\n"
    "   - `quantification_eval`: Assess if the quantitative requirements/metrics are fully satisfied by evidence.\n"
    "   - `boundary_conflicts_eval`: Detail whether the boundary conflicts identified in Stage 1 are resolved, mitigated, or violated by the evidence.\n"
    "3. Determine the final `status`:\n"
    "   - 'VERIFIED': Fully supported by evidence, all quantitative criteria met, and ALL boundary conflicts are fully resolved.\n"
    "   - 'PARTIALLY_VERIFIED': Main claim is supported, BUT missing quantitative metrics, unfulfilled criteria, or UNRESOLVED boundary conflicts remain.\n"
    "   - 'UNSUPPORTED': Insufficient evidence, direct factual contradiction, or major boundary conflict violation.\n"
    "4. Fill `verification_proofs` with exact supporting quotes, locations, or triples from evidence.\n"
    "5. Fill `compliance_gap`:\n"
    "   - If status is 'VERIFIED': MUST be set to `null`.\n"
    "   - If status is 'PARTIALLY_VERIFIED' or 'UNSUPPORTED': MUST be a string explicitly summarizing the gaps, unfulfilled metrics, or unresolved boundary conflicts.\n\n"
    "CRITICAL FORMAT RULES:\n"
    "- `diagnostic_evaluation` MUST be a nested object with text descriptions for all 3 diagnostic dimensions.\n"
    "- `compliance_gap` MUST BE A PLAIN STRING OR `null` (NEVER AN OBJECT OR LIST).\n\n"
    "Required Output JSON Format:\n"
    "{{\n"
    '  "diagnostic_evaluation": {{\n'
    '    "implicit_premises_eval": "Evaluation of assumptions against evidence...",\n'
    '    "quantification_eval": "Evaluation of metrics and criteria against evidence...",\n'
    '    "boundary_conflicts_eval": "Evaluation of boundary conflicts (whether resolved or remaining as gaps)..."\n'
    '  }},\n'
    '  "status": "VERIFIED" | "PARTIALLY_VERIFIED" | "UNSUPPORTED",\n'
    '  "verification_proofs": ["..."],\n'
    '  "compliance_gap": "Brief description string of missing criteria/gaps/conflicts, or null if fully verified"\n'
    "}}"
)

DEFAULT_OBJECTIVE_CLAIM_AGENT_SYNTHESIS_STAGE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": [
        "diagnostic_evaluation",
        "status",
        "verification_proofs",
        "compliance_gap",
    ],
    "properties": {
        "diagnostic_evaluation": {
            "type": "object",
            "description": "Evaluation of each diagnostic item against retrieved evidence.",
            "required": [
                "implicit_premises_eval",
                "quantification_eval",
                "boundary_conflicts_eval",
            ],
            "properties": {
                "implicit_premises_eval": {
                    "type": "string",
                    "description": "Assessment of unstated assumptions against evidence.",
                },
                "quantification_eval": {
                    "type": "string",
                    "description": "Assessment of quantitative requirements and metrics against evidence.",
                },
                "boundary_conflicts_eval": {
                    "type": "string",
                    "description": "Evaluation of boundary conflicts (whether resolved, mitigated, or remaining).",
                },
            },
        },
        "status": {
            "type": "string",
            "enum": ["VERIFIED", "PARTIALLY_VERIFIED", "UNSUPPORTED"],
            "description": "Final synthesis alignment status.",
        },
        "verification_proofs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Exact supporting quotes, locations, or triples from evidence.",
        },
        "compliance_gap": {
            "type": ["string", "null"],
            "description": "Summary of missing criteria/gaps/conflicts. MUST be null if status is VERIFIED, or string if PARTIALLY_VERIFIED/UNSUPPORTED.",
        },
    },
}
