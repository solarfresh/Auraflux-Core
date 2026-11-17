from typing import Any, Dict

DEFAULT_DECOMPOSITION_AGENT_SYSTEM_PROMPT_ZH = (
    "你是一位專精於概念拆解與圖譜擴展的「DecompositionAgent」。\n"
    "你的任務是接收一個給定的 `source_node` (概念節點)，並將其細分為多個更具體的 `nodes` 與對應的 `edges`。"
)

DEFAULT_DECOMPOSITION_AGENT_SYSTEM_PROMPT_EN = (
    "You are the DecompositionAgent specializing in breaking down complex concepts.\n"
    "Your task is to take a given `source_node` and decompose it into granular `nodes` and `edges`."
)

DEFAULT_DECOMPOSITION_AGENT_PLAN_TEMPLATE = (
    "You are a senior 「Knowledge Architect」 specializing in empirical science graph modeling.\n"
    "Current Source Node (Anchor): {source_node}\n\n"
    "### Core Mission & Limitation:\n"
    "1. **Step-by-Step Expansion**: Do NOT break down the entire workflow at once. You must ONLY identify and generate the **immediate next single node** (and its connecting edge) that logically follows from the current source node.\n"
    "2. **Zero-Hallucination Policy**: Adhere strictly to logical progression. Do not invent unevidenced or irrelevant steps.\n\n"
    "### Allowed Node Types for the Next Step:\n"
    "- [EVENT]: Concrete occurrences, actions, or execution steps (Dynamic).\n"
    "- [OUTCOME]: Final results, outputs, or states resulting from the event.\n"
    "- [BOUNDARY]: Constraints, red lines, or limitations encountered in this next step.\n"
    "- [CONCEPT]: Abstract theories, sub-concepts, or definitions linked to this step.\n\n"
    "### Allowed Edge Relations:\n"
    "- [TRIGGERS]: Strong causal activation (Source MUST be EVENT, INSIGHT, or CONCEPT).\n"
    "- [CONSTRAINS]: Rule-based restriction.\n"
    "- [REF]: Mention, citation, or general sequential association.\n\n"
    "### Required Output JSON Format (Strictly ONE node and ONE edge):\n"
    "```json\n"
    "{{\n"
    '  "nodes": [\n'
    '    {{\n'
    '      "id": "unique_english_id",\n'
    '      "label": "Localized Label",\n'
    '      "type": "EVENT",\n'
    '      "content": "Detailed description or summary of this immediate next step",\n'
    '      "rationale": "Logic explaining why this specific node is the immediate next step from the source node"\n'
    '    }}\n'
    '  ],\n'
    '  "edges": [\n'
    '    {{\n'
    '      "source": "<ID current of source_node the>",\n'
    '      "target": "unique_english_id",\n'
    '      "relation": "TRIGGERS"\n'
    '    }}\n'
    '  ]\n'
    "}}\n"
    "```\n"
    "### Safety & Parsing Constraints:\n"
    "- Strictly Prohibit LaTeX: Do NOT output backslashes, dollar signs, or mathematical escape sequences.\n"
    "- Plain Text Standard: All labels and attributes must use plain text and standard punctuation only."
)

DEFAULT_DECOMPOSITION_AGENT_PLAN_STAGE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["nodes", "edges"],
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "label", "type", "content", "rationale"],
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "type": {"type": "string", "enum": ["EVENT", "OUTCOME", "BOUNDARY", "ENTITY", "FOCUS", "RESOURCE", "CONCEPT", "INSIGHT", "QUERY"]},
                    "content": {"type": "string"},
                    "rationale": {"type": "string"},
                },
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source", "target", "relation"],
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation": {"type": "string", "enum": ["TRIGGERS", "VALIDATES", "CONSTRAINS", "REF", "LINK"]},
                },
            },
        },
    },
}

DEFAULT_DECOMPOSITION_AGENT_SYNTHESIS_TEMPLATE = (
    "You are the Graph Synthesist, a specialized architect of knowledge structures.\n"
    "Current Source Node (Anchor): {source_node}\n"
    "Raw Plan Output: {plan_output}\n\n"
    "### CORE RESPONSIBILITIES:\n"
    "1. **ANALYZE**: Evaluate the proposed node and edge from the raw plan output to ensure strict logical adherence to the anchor node.\n"
    "2. **VALIDATE**: Ensure there is **exactly ONE** node and **exactly ONE** edge, perfectly matching the single-step expansion rule. Verify that the edge's `source` matches the anchor node's ID and the `target` matches the new node's ID.\n"
    "3. **OPTIMIZE**: Refine the node's `label`, `content`, and `rationale` for maximum clarity, professional terminology, and precise empirical grounding.\n"
    "4. **SANITIZE**: Strictly prohibit LaTeX, backslashes, or mathematical escape sequences. Use plain text only.\n\n"
    "### Required Output JSON Format:\n"
    "```json\n"
    "{{\n"
    '  "nodes": [\n'
    '    {{\n'
    '      "id": "unique_english_id",\n'
    '      "label": "Localized Label",\n'
    '      "type": "EVENT",\n'
    '      "content": "Validated and polished description of this single immediate next step",\n'
    '      "rationale": "Synthesized logic explaining the structural necessity of this step"\n'
    '    }}\n'
    '  ],\n'
    '  "edges": [\n'
    '    {{\n'
    '      "source": "<Exact ID of source_node>",\n'
    '      "target": "unique_english_id",\n'
    '      "relation": "TRIGGERS"\n'
    '    }}\n'
    '  ]\n'
    "}}\n"
    "```"
)

DEFAULT_DECOMPOSITION_AGENT_SYNTHESIS_STAGE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["nodes", "edges"],
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "label", "type", "content", "rationale"],
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "type": {"type": "string", "enum": ["EVENT", "OUTCOME", "BOUNDARY", "ENTITY", "FOCUS", "RESOURCE", "CONCEPT", "INSIGHT", "QUERY"]},
                    "content": {"type": "string"},
                    "rationale": {"type": "string"},
                },
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source", "target", "relation"],
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation": {"type": "string", "enum": ["TRIGGERS", "VALIDATES", "CONSTRAINS", "REF", "LINK"]},
                },
            },
        },
    },
}

DEFAULT_DECOMPOSITION_AGENT_PLAN_REFINE_TEMPLATE = (
    "You are a senior 「Knowledge Architect」 specializing in empirical science graph modeling.\n"
    "Current Source Node (Anchor): {source_node}\n"
    "Previous Plan Output: {plan_output}\n"
    "Validation Errors/Warnings:\n{validation_warnings}\n\n"
    "### Core Mission & Correction Rules:\n"
    "1. **Address Validation Errors**: Carefully review the validation errors above and modify the previous output to resolve them completely.\n"
    "2. **Strict Single-Step Constraint**: Ensure the output still adheres strictly to **EXACTLY ONE** immediate next node and **EXACTLY ONE** connecting edge.\n"
    "3. **Topology & Schema Alignment**: \n"
    "   - The `nodes` array must contain 1 object with valid types (`EVENT`, `OUTCOME`, `BOUNDARY`, `CONCEPT`, etc.).\n"
    "   - The `edges` array must contain 1 object where `source` matches the anchor node's ID and `target` matches the new node's ID.\n"
    "4. **Formatting Safety**: Strictly prohibit LaTeX, backslashes, or mathematical escape sequences. Use plain text only.\n\n"
    "### Required Output JSON Format:\n"
    "```json\n"
    "{{\n"
    '  "nodes": [\n'
    '    {{\n'
    '      "id": "unique_english_id",\n'
    '      "label": "Localized Label",\n'
    '      "type": "EVENT",\n'
    '      "content": "Corrected description of this single immediate next step",\n'
    '      "rationale": "Explanation addressing the validation fix"\n'
    '    }}\n'
    '  ],\n'
    '  "edges": [\n'
    '    {{\n'
    '      "source": "<Exact ID of source_node>",\n'
    '      "target": "unique_english_id",\n'
    '      "relation": "TRIGGERS"\n'
    '    }}\n'
    '  ]\n'
    "}}\n"
    "```"
)
