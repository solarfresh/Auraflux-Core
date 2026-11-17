from typing import Any, Dict

DEFAULT_DECOMPOSITION_AGENT_SYSTEM_PROMPT_ZH = (
    "你是一位專精於概念拆解與圖譜擴展的「DecompositionAgent」。\n"
    "你的任務是接收一個給定的 `source_node` (概念節點)，並將其細分為多個更具體的 `sub_nodes` 與對應的 `causal_links`。"
)

DEFAULT_DECOMPOSITION_AGENT_SYSTEM_PROMPT_EN = (
    "You are the DecompositionAgent specializing in breaking down complex concepts.\n"
    "Your task is to take a given `source_node` and decompose it into granular `sub_nodes` and `causal_links`."
)

DEFAULT_DECOMPOSITION_AGENT_PLAN_TEMPLATE = (
    "Source Node: {source_node}\n\n"
    "Task:\n"
    "1. Break down the given `source_node` into granular `sub_nodes`.\n"
    "2. Establish causal relationships (`causal_links`) between these sub-nodes.\n\n"
    "Required Output JSON Format:\n"
    "{{\n"
    '  "sub_nodes": [\n'
    '    {{\n'
    '      "id": "unique_sub_id",\n'
    '      "label": "Sub-concept Label",\n'
    '      "type": "CONCEPT",\n'
    '      "source_ref": "Source reference",\n'
    '      "rationale": "Decomposition rationale"\n'
    '    }}\n'
    '  ],\n'
    '  "causal_links": [\n'
    '    {{\n'
    '      "source": "id1",\n'
    '      "target": "id2",\n'
    '      "relation": "TRIGGERS"\n'
    '    }}\n'
    '  ]\n'
    "}}"
)

DEFAULT_DECOMPOSITION_AGENT_PLAN_STAGE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["sub_nodes", "causal_links"],
    "properties": {
        "sub_nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "label", "type"],
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "type": {"type": "string"},
                    "source_ref": {"type": "string"},
                    "rationale": {"type": "string"},
                },
            },
        },
        "causal_links": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source", "target", "relation"],
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation": {"type": "string"},
                },
            },
        },
    },
}

DEFAULT_DECOMPOSITION_AGENT_SYNTHESIS_TEMPLATE = (
    "Source Node: {source_node}\n"
    "Plan Output: {plan_output}\n\n"
    "Task:\n"
    "Synthesize and validate the decomposition output into final structured JSON."
)

DEFAULT_DECOMPOSITION_AGENT_SYNTHESIS_STAGE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["sub_nodes", "causal_links"],
    "properties": {
        "sub_nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "label", "type"],
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "type": {"type": "string"},
                    "source_ref": {"type": "string"},
                    "rationale": {"type": "string"},
                },
            },
        },
        "causal_links": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source", "target", "relation"],
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation": {"type": "string"},
                },
            },
        },
    },
}

DEFAULT_DECOMPOSITION_AGENT_PLAN_REFINE_TEMPLATE = (
    "Source Node: {source_node}\n"
    "Previous Plan Output: {plan_output}\n"
    "Validation Errors/Warnings:\n{validation_warnings}\n\n"
    "Task:\n"
    "Please refine the decomposition plan to fix the validation errors and ensure compliance with the ontology standards."
)
