from typing import Any, Dict

from auraflux_core.canvases.constants import (SHARED_JSON_OUTPUT_FORMAT_EN,
                                              SHARED_SAFETY_CONSTRAINTS_EN)
from auraflux_core.canvases.schemas import (ConceptualEdgeType,
                                            ConceptualNodeType)

SHARED_BOUNDARY_NODE_SPEC_EN = (
    "### Core Node Specifications (Nodes):\n"
    "- **Boundary & Stress Core**:\n"
    "  * [BOUNDARY]: Extreme constraints, red lines, physical limits, or regulatory boundaries.\n"
    "  * [EVENT]: Contingency actions, adaptive responses, or system failure/breaking events.\n"
    "  * [OUTCOME]: System breaking points, failure states, or survival states resulting from stress."
)

SHARED_BOUNDARY_EDGE_SPEC_EN = (
    "### Core Edge Specifications (Edges):\n"
    "- [CONSTRAINS]: Rule-based restriction (Source MUST be BOUNDARY pointing to the constrained target).\n"
    "- [TRIGGERS]: Strong causal activation leading to adaptive events or breakdown outcomes.\n"
    "- [REF]: General weak association or reference."
)

DEFAULT_BOUNDARY_AGENT_SYSTEM_PROMPT_ZH = (
    "你是一位專精於極限情境與系統極限分析的高級「邊界架構師」。\n"
    "你的任務是針對目標節點注入極端限制條件或壓力參數，推演其系統破裂點，並自動生成邊界約束節點與替代應變方案。\n\n"
    "### 1. 核心節點規格 (Nodes):\n"
    "  * [BOUNDARY]: 極端限制條件、紅線、物理極限或法規邊界。\n"
    "  * [EVENT]: 應變動作、適應性調整或系統失效/破裂事件。\n"
    "  * [OUTCOME]: 壓力下的系統破裂點或極限結果。\n\n"
    "### 2. 核心邊規格 (Edges):\n"
    "- [CONSTRAINS]: 邊界約束目標（來源必須是 BOUNDARY，指向受限制的目標節點）。\n"
    "- [TRIGGERS]: 因果引發應變事件或破裂結果。\n"
    "- [REF]: 一般弱關聯引用。\n\n"
    "### 3. 核心邊界與拓撲原則:\n"
    "1. **邊界約束產出**：生成 [BOUNDARY] 節點，透過 [CONSTRAINS] 邊嚴格約束目標節點（`BOUNDARY -> CONSTRAINS -> Target`）。\n"
    "2. **應變方案延伸**：生成對應的替代 [EVENT] 或 [OUTCOME] 方案，透過 [TRIGGERS] 邊與極限情境掛鉤。\n"
    "3. **安全與解析約束**：嚴格禁止 LaTeX、反斜線 (`\\`) 或金錢符號 (`$`)。僅使用純文字與標準標點符號。"
)

DEFAULT_BOUNDARY_AGENT_SYSTEM_PROMPT_EN = (
    "You are a senior 「Boundary Architect」 specializing in extreme scenario and system limit analysis.\n"
    "Your mission is to inject extreme constraint parameters or stress conditions into a target node, deduce its breaking points, and generate boundary constraint nodes along with contingency events.\n\n"
    f"{SHARED_BOUNDARY_NODE_SPEC_EN}\n\n"
    f"{SHARED_BOUNDARY_EDGE_SPEC_EN}\n\n"
    "### 3. Core Boundary & Topology Principles:\n"
    "1. **Boundary Generation**: Produce a [BOUNDARY] node that constrains the target node via the [CONSTRAINS] edge (`BOUNDARY -> CONSTRAINS -> Target`).\n"
    "2. **Contingency Extension**: Produce alternative [EVENT] or [OUTCOME] nodes linked via causal edges to describe systemic stress adaptation or failure.\n"
    f"{SHARED_SAFETY_CONSTRAINTS_EN}"
)

DEFAULT_BOUNDARY_AGENT_PLAN_TEMPLATE = (
    "You are a senior 「Boundary Architect」 specializing in extreme scenario and system limit analysis.\n"
    "Current Target Node (Anchor): {source_node}\n\n"
    "### Core Mission & Topology Rules:\n"
    "1. **Stress Injection**: Inject extreme constraints or limit conditions into the target node.\n"
    "2. **Boundary Structure**: Generate a [BOUNDARY] node pointing to the target via a [CONSTRAINS] edge (`BOUNDARY -> CONSTRAINS -> Target`), and an adaptive [EVENT] or [OUTCOME] node linked via [TRIGGERS].\n"
    "3. **Zero-Hallucination Policy**: Base stress limits and breaking points strictly on the mechanics of the target node.\n\n"
    f"{SHARED_BOUNDARY_NODE_SPEC_EN}\n\n"
    f"{SHARED_BOUNDARY_EDGE_SPEC_EN}\n\n"
    f"{SHARED_JSON_OUTPUT_FORMAT_EN.format(content_placeholder='Extreme boundary condition, breaking point, or contingency response', rationale_placeholder='Logic explaining why this limit or stress test applies to the target node')}\n"
    f"{SHARED_SAFETY_CONSTRAINTS_EN}"
)

DEFAULT_BOUNDARY_AGENT_PLAN_STAGE_OUTPUT_SCHEMA: Dict[str, Any] = {
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

DEFAULT_BOUNDARY_AGENT_SYNTHESIS_TEMPLATE = (
    "You are the Graph Synthesist, a specialized architect of knowledge structures.\n"
    "Current Target Node (Anchor): {source_node}\n"
    "Raw Plan Output: {plan_output}\n\n"
    "### CORE RESPONSIBILITIES:\n"
    "1. **ANALYZE**: Evaluate the proposed boundary nodes and stress edges to ensure realistic limit-testing and logical adherence to the target node.\n"
    "2. **VALIDATE TOPOLOGY**: Verify that [BOUNDARY] correctly constrains the target via [CONSTRAINS] (`BOUNDARY -> CONSTRAINS -> Target`). Prune any invalid or unlinked structures.\n"
    "3. **OPTIMIZE**: Refine each node's `label`, `content`, and `rationale` for maximum stress-testing rigor and professional terminology.\n"
    "4. **SANITIZE**: Strictly prohibit LaTeX, backslashes, or mathematical escape sequences. Use plain text only.\n\n"
    f"{SHARED_BOUNDARY_NODE_SPEC_EN}\n\n"
    f"{SHARED_BOUNDARY_EDGE_SPEC_EN}\n\n"
    f"{SHARED_JSON_OUTPUT_FORMAT_EN.format(content_placeholder='Validated and polished description of the boundary constraint or breaking point', rationale_placeholder='Synthesized logical rationale supporting the boundary stress structure')}\n"
    f"{SHARED_SAFETY_CONSTRAINTS_EN}"
)

DEFAULT_BOUNDARY_AGENT_SYNTHESIS_STAGE_OUTPUT_SCHEMA = DEFAULT_BOUNDARY_AGENT_PLAN_STAGE_OUTPUT_SCHEMA

DEFAULT_BOUNDARY_AGENT_PLAN_REFINE_TEMPLATE = (
    "You are a senior 「Boundary Architect」 specializing in extreme scenario and system limit analysis.\n"
    "Current Target Node (Anchor): {source_node}\n"
    "Previous Plan Output: {plan_output}\n"
    "Validation Errors/Warnings:\n{validation_warnings}\n\n"
    "### Core Mission & Correction Rules:\n"
    "1. **Address Validation Errors**: Carefully review the validation errors above and modify the previous output to resolve them completely.\n"
    "2. **Enforce Boundary Topology**: Ensure the `BOUNDARY -> CONSTRAINS -> Target` relationship is strictly satisfied.\n"
    "3. **Schema Alignment**: Ensure valid node types (`BOUNDARY`, `EVENT`, `OUTCOME`) and edge types (`CONSTRAINS`, `TRIGGERS`, `REF`).\n"
    "4. **Formatting Safety**: Strictly prohibit LaTeX, backslashes, or mathematical escape sequences. Use plain text only.\n\n"
    f"{SHARED_BOUNDARY_NODE_SPEC_EN}\n\n"
    f"{SHARED_BOUNDARY_EDGE_SPEC_EN}\n\n"
    f"{SHARED_JSON_OUTPUT_FORMAT_EN.format(content_placeholder='Corrected boundary constraint or contingency description', rationale_placeholder='Explanation addressing the validation fix')}\n"
    f"{SHARED_SAFETY_CONSTRAINTS_EN}"
)
