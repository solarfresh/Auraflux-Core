from typing import Any, Dict

DEFAULT_MENTAL_MODEL_AGENT_SYSTEM_PROMPT_ZH = (
    "你是一名心智模型與主觀期待稽核專家（Mental Model Audit Specialist）。\n"
    "你的核心任務是針對傳入命題進行正交診斷分析（Orthogonal Diagnostics）：\n"
    "1. 【主戰場】發動四維度防遺漏矩陣（Capability, Resources, Causality, Environment），"
    "剖析主體對另一實體隱含且未經證實的認知假設。\n"
    "2. 標註主觀期待中的模糊形容詞與抽象概念，指出待量化的指標（如 SLA 時間、驗收標準）。\n"
    "3. 捕捉主觀期待與客觀規章/限制間可能存在的認知落差，並構建精準的事證檢索計畫。\n"
    "在驗證階段，你必須嚴格基於檢索到的客觀證據判斷主觀期待是否成立，"
    "判決結果為 VERIFIED（認知吻合）/ VIOLATED（證實落差）/ UNSUPPORTED（關鍵假設無佐證），絕不允許臆測評分。"
)

DEFAULT_MENTAL_MODEL_AGENT_SYSTEM_PROMPT_EN = (
    "You are a Mental Model Audit Specialist responsible for orthogonal diagnostic analysis "
    "of subjective expectations, beliefs, and implicit assumptions held by one party about another.\n"
    "Your core responsibilities:\n"
    "1. [Main battlefield] Run the four-dimension assumption matrix "
    "(Capability, Resources, Causality, Environment) to surface implicit, "
    "unverified cognitive assumptions.\n"
    "2. Flag vague subjective adjectives and abstract expectations, identifying which "
    "metrics still require quantification (e.g., SLA duration, acceptance criteria).\n"
    "3. Detect potential gaps between the subjective expectation and objective policy/"
    "constraints, and formulate a targeted evidence retrieval strategy.\n"
    "During verification, you must judge whether the subjective expectation holds strictly "
    "against retrieved objective evidence, rendering VERIFIED (aligned) / VIOLATED (confirmed gap) / "
    "UNSUPPORTED (key assumption has no evidence) without subjective hallucination."
)

DEFAULT_MENTAL_MODEL_AGENT_PLAN_TEMPLATE = (
    "Proposition ID: {proposition_id}\n"
    "Claim: {claim_text}\n\n"
    "Task:\n"
    "1. Extract semantic triples (Subject -> Predicate -> Object).\n"
    "2. Perform orthogonal diagnosis:\n"
    "   - implicit_premises (MAIN BATTLEFIELD): Run the 4D assumption matrix. For EACH of the "
    "four dimensions below, list unverified assumptions found in the claim (empty array if none):\n"
    "       * capability_assumptions: assumptions about the counterparty's technical/execution capability.\n"
    "       * resource_assumptions: assumptions about resource, budget, or authorization scope.\n"
    "       * causality_assumptions: unexamined links taken for granted in the causal chain.\n"
    "       * environment_assumptions: assumptions that the external environment stays static.\n"
    "   - quantification_requirements: Vague subjective adjectives/abstract expectations mapped to "
    "the metric that still needs to be quantified (e.g. {{'quickly': 'needs SLA in days'}}).\n"
    "   - boundary_conflicts: Potential gaps between this subjective expectation and known policy/"
    "constraints (has_conflict flag + description).\n"
    "3. Formulate targeted retrieval queries (`queries`) to search Core Context for evidence that "
    "supports or refutes this expectation (e.g., design docs, prior commitments, org policy). "
    "Generate 1 to 2 distinct query items tailored to specific search intents. Each item must "
    "strictly bind to ONE text field and ONE vector field:\n"
    "   - `query_text`: The rewritten query string optimized for retrieval (MUST BE IN ENGLISH).\n"
    "   - `target_type`: chosen strictly from ['question', 'concept_title', 'concept_desc', 'evidence']:\n"
    "       * 'question': (text_field: 'target_question', vector_field: 'question_vector')\n"
    "       * 'concept_title': (text_field: 'concept_title', vector_field: 'concept_vector')\n"
    "       * 'concept_desc': (text_field: 'concept_description', vector_field: 'concept_vector')\n"
    "       * 'evidence': (text_field: 'evidence_text', vector_field: 'evidence_vector')\n"
    "   - `text_field`: Exactly ONE text field name from ['target_question', 'concept_title', "
    "'concept_description', 'evidence_text'].\n"
    "   - `vector_field`: Exactly ONE vector field name from ['question_vector', 'concept_vector', "
    "'evidence_vector'].\n\n"
    "CRITICAL FORMAT RULES:\n"
    "- `query_text` MUST BE IN ENGLISH regardless of the input claim language.\n"
    "- `quantification_requirements` and `boundary_conflicts` MUST BE DICTIONARY OBJECTS (NOT ARRAYS).\n"
    "- `implicit_premises` MUST be an object with exactly the four keys listed above, each an array.\n"
    "- `queries` MUST BE A NON-EMPTY ARRAY containing 1 or 2 targeted query objects.\n\n"
    "Required Output JSON Format:\n"
    "{{\n"
    '  "triples": [\n'
    '    {{"subject": "...", "predicate": "...", "object": "..."}}\n'
    '  ],\n'
    '  "diagnostics": {{\n'
    '    "implicit_premises": {{\n'
    '      "capability_assumptions": ["..."],\n'
    '      "resource_assumptions": ["..."],\n'
    '      "causality_assumptions": ["..."],\n'
    '      "environment_assumptions": ["..."]\n'
    '    }},\n'
    '    "quantification_requirements": {{"...": "..."}},\n'
    '    "boundary_conflicts": {{"has_conflict": false}}\n'
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

DEFAULT_MENTAL_MODEL_AGENT_PLAN_STAGE_OUTPUT_SCHEMA: Dict[str, Any] = {
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
                    "type": "object",
                    "description": "Mapping of vague subjective adjectives to required metrics.",
                    "additionalProperties": {"type": "string"},
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
    }
}

DEFAULT_MENTAL_MODEL_AGENT_SYNTHESIS_TEMPLATE = (
    "Proposition ID: {proposition_id}\n"
    "Claim (Subjective Expectation): {claim_text}\n"
    "Extracted Claim Triples (Stage 1): {triples}\n"
    "Diagnostics (Stage 1): {diagnostics}\n\n"
    "Retrieved Structured Core Context Evidence:\n{formatted_evidence_text}\n\n"
    "Task:\n"
    "1. Perform a thorough cross-examination of the subjective expectation and Stage 1 4D "
    "assumption matrix against the Retrieved Evidence.\n"
    "2. Evaluate and EXPLICITLY OUTPUT `diagnostic_evaluation` for each diagnostic dimension:\n"
    "   - `capability_eval`: Is the capability assumption supported by evidence?\n"
    "   - `resource_eval`: Is the resource/authorization assumption supported by evidence?\n"
    "   - `causality_eval`: Does evidence support the assumed causal chain?\n"
    "   - `environment_eval`: Does evidence confirm the environment assumption still holds?\n"
    "   - `quantification_eval`: Are the flagged vague expectations now quantifiable from evidence?\n"
    "   - `boundary_conflicts_eval`: Is the subjective expectation in conflict with objective "
    "policy/constraints found in evidence?\n"
    "3. Determine the final `status`:\n"
    "   - 'VERIFIED': The subjective expectation is fully aligned with retrieved evidence "
    "(equivalent to ALIGNED) — no unresolved assumption or conflict remains.\n"
    "   - 'VIOLATED': Evidence confirms a cognitive gap between the expectation and actual "
    "state (equivalent to MISALIGNED).\n"
    "   - 'UNSUPPORTED': A key assumption in the 4D matrix has no supporting or refuting "
    "evidence at all (equivalent to OPAQUE_RISK) — this MUST trigger a BLOCK requiring "
    "human clarification.\n"
    "4. Fill `verification_proofs` with exact supporting quotes, locations, or triples from evidence.\n"
    "5. Fill `compliance_gap`:\n"
    "   - If status is 'VERIFIED': MUST be set to `null`.\n"
    "   - If status is 'VIOLATED' or 'UNSUPPORTED': MUST be a string explicitly summarizing the "
    "gap, unresolved assumption, or conflict.\n\n"
    "CRITICAL FORMAT RULES:\n"
    "- `diagnostic_evaluation` MUST be a nested object with text descriptions for all 6 dimensions above.\n"
    "- `compliance_gap` MUST BE A PLAIN STRING OR `null` (NEVER AN OBJECT OR LIST).\n\n"
    "Required Output JSON Format:\n"
    "{{\n"
    '  "diagnostic_evaluation": {{\n'
    '    "capability_eval": "...",\n'
    '    "resource_eval": "...",\n'
    '    "causality_eval": "...",\n'
    '    "environment_eval": "...",\n'
    '    "quantification_eval": "...",\n'
    '    "boundary_conflicts_eval": "..."\n'
    '  }},\n'
    '  "status": "VERIFIED" | "VIOLATED" | "UNSUPPORTED",\n'
    '  "verification_proofs": ["..."],\n'
    '  "compliance_gap": "Brief description string of the gap/unresolved assumption, or null"\n'
    "}}"
)

DEFAULT_MENTAL_MODEL_AGENT_SYNTHESIS_STAGE_OUTPUT_SCHEMA: Dict[str, Any] = {
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
            "enum": ["VERIFIED", "VIOLATED", "UNSUPPORTED"],
            "description": "Final synthesis alignment status.",
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
