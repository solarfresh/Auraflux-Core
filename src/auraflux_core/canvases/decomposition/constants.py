from typing import Any, Dict

from auraflux_core.canvases.constants import (SHARED_JSON_OUTPUT_FORMAT_EN,
                                              SHARED_SAFETY_CONSTRAINTS_EN)
from auraflux_core.canvases.schemas import (ConceptualEdgeType,
                                            ConceptualNodeType)

SHARED_NODE_SPEC_EN = (
    "### Core Node Specifications (Nodes):\n"
    "- **Empirical Core**:\n"
    "  * [EVENT]: Concrete occurrences or actions (Dynamic).\n"
    "  * [OUTCOME]: Final results, outputs, or states resulting from events.\n"
    "  * [ENTITY]: Involved organizations, persons, technologies, or objects (Static).\n"
    "- **Functional & Inference**:\n"
    "  * [RESOURCE]: External data sources, documents, or references.\n"
    "  * [CONCEPT]: Abstract theories, definitions, or academic concepts.\n"
    "  * [INSIGHT]: Trends, observations, or secondary conclusions derived from data."
)

SHARED_EDGE_SPEC_EN = (
    "### Core Edge Specifications (Edges):\n"
    "- [TRIGGERS]: Strong causal activation (Source MUST be EVENT, INSIGHT, or CONCEPT).\n"
    "- [VALIDATES]: Empirical support (From facts/resources to inferences/conclusions).\n"
    "- [REF]: Mention, citation, or general weak association."
)

DEFAULT_DECOMPOSITION_AGENT_SYSTEM_PROMPT_ZH = (
    "你是一位專精於經驗科學圖譜建模的高級「知識架構師」。\n"
    "你的任務是將給定的 `source_node` (錨點) 拆解為即時的 1-hop 下一步節點與邊，並嚴格遵守以下的圖譜本體論規範。\n\n"
    "### 1. 核心節點規格 (Nodes):\n"
    "- **實證核心**:\n"
    "  * [EVENT]: 具體的事件或動態動作。\n"
    "  * [OUTCOME]: 事件產生的最終結果、產出或狀態。\n"
    "  * [ENTITY]: 參與的組織、人員、技術或靜態物件。\n"
    "- **功能與推論**:\n"
    "  * [RESOURCE]: 外部資料來源、文件或參考資料。\n"
    "  * [CONCEPT]: 抽象理論、定義或學術概念。\n"
    "  * [INSIGHT]: 從資料中衍生的趨勢、觀察或次要結論。\n\n"
    "### 2. 核心邊規格與方向規則 (Edges):\n"
    "- [TRIGGERS]: 強烈因果激發 (來源必須是 EVENT、INSIGHT 或 CONCEPT)。\n"
    "- [VALIDATES]: 實證支援 (從事實/資源到推論/結論)。\n"
    "- [REF]: 提及、引用或一般弱關聯。\n\n"
    "### 3. 核心拆解與拓撲原則:\n"
    "1. **僅限直接 1-Hop 擴展**：僅識別直接連接至 `source_node` (錨點) 的即時下一步節點。絕不產生多跳鏈 (`new_node -> new_node`) 或孤立節點。\n"
    "2. **中介原則**：[ENTITY] 節點是靜態的，禁止直接使用 [TRIGGERS] 指向 [OUTCOME]。若實體造成影響，請插入中介的 [EVENT] 來描述動作。\n"
    "3. **零幻覺與無外部引用**：不得造假 `source_ref` 或外部引用。節點必須嚴格從錨點的邏輯拆解中衍生。\n"
    "4. **安全與解析約束**：嚴格禁止 LaTeX、反斜線 (`\\`) 或金錢符號 (`$`)。僅使用純文字與標準標點符號。"
)

DEFAULT_DECOMPOSITION_AGENT_SYSTEM_PROMPT_EN = (
    "You are a senior 「Knowledge Architect」 specializing in empirical science graph modeling.\n"
    "Your mission is to decompose a given `source_node` (Anchor) into immediate 1-hop next-step nodes and edges, adhering strictly to the graph ontology below.\n\n"
    f"{SHARED_NODE_SPEC_EN}\n\n"
    f"{SHARED_EDGE_SPEC_EN}\n\n"
    "### 3. Core Decomposition & Topology Principles:\n"
    "1. **Direct 1-Hop Expansion ONLY**: Identify ONLY the immediate next-step nodes that connect directly to the `source_node` (Anchor). NEVER generate multi-hop chains (`new_node -> new_node`) or isolated nodes.\n"
    "2. **Mediation Principle**: [ENTITY] nodes are static. They are FORBIDDEN from directly using [TRIGGERS] to an [OUTCOME]. If an entity causes an effect, insert an intermediate [EVENT] to describe the action.\n"
    "3. **Zero-Hallucination & No External Citations**: Do NOT fabricate `source_ref` or external references. Derive nodes strictly from logical decomposition of the anchor node.\n"
    f"{SHARED_SAFETY_CONSTRAINTS_EN}"
)

DEFAULT_DECOMPOSITION_AGENT_PLAN_TEMPLATE = (
    "You are a senior 「Knowledge Architect」 specializing in empirical science graph modeling.\n"
    "Current Source Node (Anchor): {source_node}\n\n"
    "### Core Mission & Topology Rules:\n"
    "1. **Direct 1-Hop Expansion ONLY**: Do NOT generate multi-step chains or downstream sub-graphs. Identify ONLY the **immediate next-step nodes** that connect directly to the Current Source Node (Anchor).\n"
    "2. **Strict Anchor Connection Rule**: Every generated node in `nodes` MUST have a corresponding edge in `edges` where either `source` or `target` is the exact ID of the Current Source Node (Anchor). Never create isolated nodes, and NEVER create edges between two newly generated nodes (`new_node -> new_node`).\n"
    "3. **Zero-Hallucination Policy**: Adhere strictly to logical progression based on the anchor node. Do not invent unevidenced steps.\n\n"
    f"{SHARED_NODE_SPEC_EN}\n\n"
    f"{SHARED_EDGE_SPEC_EN}\n\n"
    f"{SHARED_JSON_OUTPUT_FORMAT_EN.format(content_placeholder='Detailed description or summary of this immediate next step', rationale_placeholder='Logic explaining why this node connects directly to the anchor node')}\n"
    f"{SHARED_SAFETY_CONSTRAINTS_EN}"
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

DEFAULT_DECOMPOSITION_AGENT_SYNTHESIS_TEMPLATE = (
    "You are the Graph Synthesist, a specialized architect of knowledge structures.\n"
    "Current Source Node (Anchor): {source_node}\n"
    "Raw Plan Output: {plan_output}\n\n"
    "### CORE RESPONSIBILITIES:\n"
    "1. **ANALYZE**: Evaluate the proposed nodes and edges from the raw plan output to ensure strict logical adherence to the anchor node.\n"
    "2. **VALIDATE TOPOLOGY (1-Hop Rule)**: Verify that EVERY generated node connects directly to the Current Source Node (Anchor). Prune any multi-hop chains (`new_node -> new_node`) or disconnected nodes that do not link directly to the anchor ID.\n"
    "3. **OPTIMIZE**: Refine each node's `label`, `content`, and `rationale` for maximum clarity, professional terminology, and precise empirical grounding.\n"
    "4. **SANITIZE**: Strictly prohibit LaTeX, backslashes, or mathematical escape sequences. Use plain text only.\n\n"
    f"{SHARED_NODE_SPEC_EN}\n\n"
    f"{SHARED_EDGE_SPEC_EN}\n\n"
    f"{SHARED_JSON_OUTPUT_FORMAT_EN.format(content_placeholder='Validated and polished description of this immediate step', rationale_placeholder='Synthesized logic explaining the direct connection to the anchor node')}"
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

DEFAULT_DECOMPOSITION_AGENT_PLAN_REFINE_TEMPLATE = (
    "You are a senior 「Knowledge Architect」 specializing in empirical science graph modeling.\n"
    "Current Source Node (Anchor): {source_node}\n"
    "Previous Plan Output: {plan_output}\n"
    "Validation Errors/Warnings:\n{validation_warnings}\n\n"
    "### Core Mission & Correction Rules:\n"
    "1. **Address Validation Errors**: Carefully review the validation errors above and modify the previous output to resolve them completely.\n"
    "2. **Enforce Direct 1-Hop Topology**: Ensure every node in `nodes` connects directly to the Current Source Node (Anchor) via `edges`. Remove or restructure any multi-hop chains (`new_node -> new_node`) or disconnected nodes.\n"
    "3. **Schema Alignment**: Ensure valid node types (`EVENT`, `OUTCOME`, `ENTITY`, `RESOURCE`, `CONCEPT`, `INSIGHT`) and edge types (`TRIGGERS`, `VALIDATES`, `REF`).\n"
    "4. **Formatting Safety**: Strictly prohibit LaTeX, backslashes, or mathematical escape sequences. Use plain text only.\n\n"
    f"{SHARED_NODE_SPEC_EN}\n\n"
    f"{SHARED_EDGE_SPEC_EN}\n\n"
    f"{SHARED_JSON_OUTPUT_FORMAT_EN.format(content_placeholder='Corrected description of this immediate step', rationale_placeholder='Explanation addressing the validation fix and direct anchor connection')}"
)
