def get_json_output_format_en(content_placeholder: str, rationale_placeholder: str) -> str:
    return (
        "### Required Output JSON Format:\n"
        "```json\n"
        "{{\n"
        '  "nodes": [\n'
        '    {{\n'
        '      "id": "unique_english_id",\n'
        '      "label": "Localized Label",\n'
        '      "type": "EVENT",\n'
        f'      "content": "{content_placeholder}",\n'
        f'      "rationale": "{rationale_placeholder}"\n'
        '    }}\n'
        '  ],\n'
        '  "edges": [\n'
        '    {{\n'
        '      "source": "<Exact ID of source_node>",\n'
        '      "target": "unique_english_id",\n'
        '      "type": "TRIGGERS",\n'
        '      "rationale": "Logic explaining why this edge connects source and target"\n'
        '    }}\n'
        '  ]\n'
        "}}\n"
        "```"
    )

SHARED_SAFETY_CONSTRAINTS_EN = (
    "### Safety & Parsing Constraints:\n"
    "- Strictly Prohibit LaTeX: Do NOT output backslashes, dollar signs, or mathematical escape sequences.\n"
    "- Plain Text Standard: All labels and attributes must use plain text and standard punctuation only."
)
