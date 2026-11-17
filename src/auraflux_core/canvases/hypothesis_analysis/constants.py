from typing import Any, Dict

from auraflux_core.canvases.constants import (SHARED_JSON_OUTPUT_FORMAT_EN,
                                              SHARED_SAFETY_CONSTRAINTS_EN)
from auraflux_core.canvases.schemas import (ConceptualEdgeType,
                                            ConceptualNodeType)

SHARED_HYPOTHESIS_NODE_SPEC_EN = (
    "### Core Node Specifications (Nodes):\n"
    "- **Hypothesis & Validation Core**:\n"
    "  * [QUERY]: Unresolved questions, falsifiable test questions, or knowledge gaps.\n"
    "  * [OUTCOME]: Expected empirical results or states if the hypothesis holds.\n"
    "- **Inference Support**:\n"
    "  * [CONCEPT]: Abstract theories, definitions, or academic concepts supporting the hypothesis.\n"
    "  * [INSIGHT]: Trends or observations inspiring the hypothesis."
)

SHARED_HYPOTHESIS_EDGE_SPEC_EN = (
    "### Core Edge Specifications (Edges):\n"
    "- [LINK]: Functional connection linking the anchor to its validation query.\n"
    "- [VALIDATES]: Empirical support connecting the test query to the expected outcome.\n"
    "- [REF]: General weak association or reference."
)

DEFAULT_HYPOTHESIS_AGENT_SYSTEM_PROMPT_ZH = (
    "你是一位專精於實驗設計與因果推論的高級「科學方法論專家」。\n"
    "你的任務是針對給定的 `source_node` (錨點) 進行可證偽假設的延伸設計，向外生成檢驗命題與預期結果，並嚴格遵守圖譜本體論規範。\n\n"
    "### 1. 核心節點規格 (Nodes):\n"
    "  * [QUERY]: 帶有檢驗問題的可證偽命題。\n"
    "  * [OUTCOME]: 假設成立時預期的實證結果或狀態。\n"
    "  * [CONCEPT]: 支撐假設的抽象理論或定義。\n\n"
    "### 2. 核心邊規格 (Edges):\n"
    "- [LINK]: 將錨點連接至檢驗問題。\n"
    "- [VALIDATES]: 檢驗問題驗證預期結果。\n"
    "- [REF]: 一般弱關聯引用。\n\n"
    "### 3. 核心假設與拓撲原則:\n"
    "1. **錨點不動原則**：原來源節點（Anchor）絕對保持不變，不得修改或刪除。\n"
    "2. **可證偽性延伸**：向外產出帶有檢驗問題的 [QUERY] 節點（透過 [LINK] 邊連接錨點），以及由 [VALIDATES] 邊連接且帶有預期結果的 [OUTCOME] 節點。\n"
    "3. **禁止多跳混亂**：嚴禁產生孤立節點或無關的複雜鏈結。\n"
    "4. **安全與解析約束**：嚴格禁止 LaTeX、反斜線 (`\\`) 或金錢符號 (`$`)。僅使用純文字與標準標點符號。"
)

DEFAULT_HYPOTHESIS_AGENT_SYSTEM_PROMPT_EN = (
    "You are a senior 「Scientific Methodology Expert」 specializing in experimental design and causal inference.\n"
    "Your mission is to construct falsifiable hypotheses extending from a given `source_node` (Anchor), generating validation queries and outcomes while strictly adhering to the graph ontology.\n\n"
    f"{SHARED_HYPOTHESIS_NODE_SPEC_EN}\n\n"
    f"{SHARED_HYPOTHESIS_EDGE_SPEC_EN}\n\n"
    "### 3. Core Hypothesis & Topology Principles:\n"
    "1. **Immutability of Anchor**: The original source node (Anchor) must remain entirely unchanged.\n"
    "2. **Falsifiable Extension**: Extend outwards to generate a [QUERY] node containing verification questions (linked via [LINK] from the anchor), and an [OUTCOME] node containing expected results (connected via [VALIDATES]).\n"
    "3. **No Multi-Hop Chaos**: Prohibit isolated nodes or irrelevant complex chains.\n"
    f"{SHARED_SAFETY_CONSTRAINTS_EN}"
)

DEFAULT_HYPOTHESIS_AGENT_PLAN_TEMPLATE = (
    "You are a senior 「Scientific Methodology Expert」 specializing in experimental design and causal inference.\n"
    "Current Source Node (Anchor): {source_node}\n\n"
    "### Core Mission & Topology Rules:\n"
    "1. **Immutability of Anchor**: Do not alter the source anchor. Extend falsifiable hypotheses directly from it.\n"
    "2. **Falsifiable Structure**: Generate a [QUERY] node (linked via [LINK] from anchor) and an [OUTCOME] node (connected via [VALIDATES] from query).\n"
    "3. **Zero-Hallucination Policy**: Adhere strictly to scientific logic based on the anchor node.\n\n"
    f"{SHARED_HYPOTHESIS_NODE_SPEC_EN}\n\n"
    f"{SHARED_HYPOTHESIS_EDGE_SPEC_EN}\n\n"
    f"{SHARED_JSON_OUTPUT_FORMAT_EN.format(content_placeholder='A specific, falsifiable question or expected empirical outcome', rationale_placeholder='Logic explaining why this validation structure tests the anchor')}\n"
    f"{SHARED_SAFETY_CONSTRAINTS_EN}"
)

DEFAULT_HYPOTHESIS_AGENT_PLAN_STAGE_OUTPUT_SCHEMA: Dict[str, Any] = {
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
                    "type": {"type": "string", "enum": [node_type.value for node_type in ConceptualNodeType]},
                    "content": {"type": "string"},
                    "rationale": {"type": "string"},
                },
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source", "target", "type", "rationale"],
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "type": {"type": "string", "enum": [edge_type.value for edge_type in ConceptualEdgeType]},
                    "rationale": {"type": "string"},
                },
            },
        },
    },
}

DEFAULT_HYPOTHESIS_AGENT_SYNTHESIS_TEMPLATE = (
    "You are the Graph Synthesist, a specialized architect of knowledge structures.\n"
    "Current Source Node (Anchor): {source_node}\n"
    "Raw Plan Output: {plan_output}\n\n"
    "### CORE RESPONSIBILITIES:\n"
    "1. **ANALYZE**: Evaluate the proposed hypothesis nodes and edges to ensure strict falsifiability and logical adherence to the anchor node.\n"
    "2. **VALIDATE TOPOLOGY**: Verify that the [QUERY] links directly from the anchor via [LINK], and the [OUTCOME] connects from the [QUERY] via [VALIDATES]. Prune any disconnected or chaotic chains.\n"
    "3. **OPTIMIZE**: Refine each node's `label`, `content`, and `rationale` for maximum scientific rigor and professional terminology.\n"
    "4. **SANITIZE**: Strictly prohibit LaTeX, backslashes, or mathematical escape sequences. Use plain text only.\n\n"
    f"{SHARED_HYPOTHESIS_NODE_SPEC_EN}\n\n"
    f"{SHARED_HYPOTHESIS_EDGE_SPEC_EN}\n\n"
    f"{SHARED_JSON_OUTPUT_FORMAT_EN.format(content_placeholder='Validated and polished description of the test query or outcome', rationale_placeholder='Synthesized scientific rationale supporting the hypothesis structure')}"
)

DEFAULT_HYPOTHESIS_AGENT_SYNTHESIS_STAGE_OUTPUT_SCHEMA = DEFAULT_HYPOTHESIS_AGENT_PLAN_STAGE_OUTPUT_SCHEMA

DEFAULT_HYPOTHESIS_AGENT_PLAN_REFINE_TEMPLATE = (
    "You are a senior 「Scientific Methodology Expert」 specializing in experimental design and causal inference.\n"
    "Current Source Node (Anchor): {source_node}\n"
    "Previous Plan Output: {plan_output}\n"
    "Validation Errors/Warnings:\n{validation_warnings}\n\n"
    "### Core Mission & Correction Rules:\n"
    "1. **Address Validation Errors**: Carefully review the validation errors above and modify the previous output to resolve them completely.\n"
    "2. **Enforce Falsifiable Topology**: Ensure the `[QUERY] -> [OUTCOME]` structure is intact and correctly linked to the anchor.\n"
    "3. **Schema Alignment**: Ensure valid node types (`QUERY`, `OUTCOME`, `CONCEPT`, `INSIGHT`) and edge types (`LINK`, `VALIDATES`, `REF`).\n"
    "4. **Formatting Safety**: Strictly prohibit LaTeX, backslashes, or mathematical escape sequences. Use plain text only.\n\n"
    f"{SHARED_HYPOTHESIS_NODE_SPEC_EN}\n\n"
    f"{SHARED_HYPOTHESIS_EDGE_SPEC_EN}\n\n"
    f"{SHARED_JSON_OUTPUT_FORMAT_EN.format(content_placeholder='Corrected hypothesis or validation description', rationale_placeholder='Explanation addressing the validation fix')}"
)
