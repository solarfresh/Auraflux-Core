from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from auraflux_core.core.schemas.agents import AgentConfig
from auraflux_core.core.schemas.tools import ToolConfig


class NodeHandle(str, Enum):
    NORTH = 'n'
    EAST = 'e'
    WEST = 'w'
    SOUTH = 's'


class Position(BaseModel):
    x: float
    y: float


class ConceptualNodeType(str, Enum):
    """
    Unified Node Type for Auraflux.
    Integrates Empirical Science standards with Canvas Functional roles.
    """
    # --- Empirical Core ---

    EVENT = "EVENT"
    """[Empirical] Objective observations, facts, or raw data points extracted from source."""

    OUTCOME = "OUTCOME"
    """[Empirical] Final results, proven hypotheses, or the 'North Star' of the research."""

    BOUNDARY = "BOUNDARY"
    """[Empirical] Constraints, limitations, or the specific scope of the study."""

    ENTITY = "ENTITY"
    """[Empirical] Subjects, instruments, agents, or specific tools involved in the research."""

    # --- Canvas Functional ---

    FOCUS = "FOCUS"
    """[Functional] The central question or focal point of the current canvas view."""

    RESOURCE = "RESOURCE"
    """[Functional] External evidence items or documents linked to the knowledge layer."""

    CONCEPT = "CONCEPT"
    """[Functional] Logical bridges or TopicKeywords that manage Panel-Canvas synchronization."""

    INSIGHT = "INSIGHT"
    """[Shared] Synthesis generated from AI analysis or reflection logs. Bridging logic and data."""

    QUERY = "QUERY"
    """[Functional] Represents a research gap or an unanswered 'need-to-know' compass."""

    NAVIGATION = "NAVIGATION"
    """[System] Portal for transitioning between different space-time CanvasViews."""

    GROUP = "GROUP"
    """[Container] Defines physical boundaries and categorization of nodes on the canvas."""


class ConceptualNode(BaseModel):
    """
    The Single Source of Truth for all Auraflux nodes.
    Combines Empirical logic and Spatial layout attributes.
    """
    # --- Identity & UI ---
    id: str = Field(default_factory=lambda: str(uuid4()))
    label: str = Field(..., min_length=1)
    type: ConceptualNodeType = Field(..., description="The unified role of this node")

    # --- Knowledge Content (Empirical Layer) ---
    content: Optional[str] = Field(None, description="Detailed text or data snippet")
    source_ref: Optional[str] = Field(None, description="Zero-inference grounding reference")

    # --- AI Reasoning (Logic Layer) ---
    rationale: Optional[str] = Field(None, description="AI's justification for this node")
    anchor_id: Optional[str] = Field(None, description="Parent node reference for growth tracking")

    # --- Spatial Information (Layout Layer) ---
    position: Optional[Position] = None

    class Config:
        use_enum_values = True
        populate_by_name = True


class ConceptualEdgeType(str, Enum):
    """
    Unified Edge Type for Auraflux.
    Integrates Empirical Science (4-Edge) with Canvas Functional links.
    """
    # --- Empirical Core (Phase 1: Knowledge Reconstruction) ---

    VALIDATES = "VALIDATES"
    """[Empirical] Strong logical support or evidence verification. (e.g., Event -> Insight)"""

    CONSTRAINS = "CONSTRAINS"
    """[Empirical] Definitive restriction or boundary setting. (e.g., Boundary -> Event)"""

    TRIGGERS = "TRIGGERS"
    """[Empirical] Causal relationship or sequential activation. (e.g., Event -> Outcome)"""

    # --- Canvas & General Functional ---

    REF = "REF"
    """[Shared] Weak association, mention, or general reference between nodes."""

    LINK = "LINK"
    """[Functional] Direct navigation or structural connection in the UI."""


class ConceptualEdge(BaseModel):
    """
    The Single Source of Truth for all Auraflux relationships.
    Combines Empirical logic verification with Spatial connection properties.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    label: Optional[str] = None
    type: ConceptualEdgeType = Field(default=ConceptualEdgeType.REF)

    # --- Connection Core ---
    source: str = Field(..., description="ID of the source node")
    target: str = Field(..., description="ID of the target node")

    # --- Knowledge & Grounding (Empirical Layer) ---
    evidence: Optional[str] = Field(
        None,
        description="Logical justification for this link (Crucial for Agentic Audit)"
    )
    weight: float = Field(default=1.0, description="Strength of the relationship")

    # --- AI Reasoning & Metadata (Logic Layer) ---
    rationale: Optional[str] = Field(None, description="AI's reasoning for creating this link")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Bucket for scenario-specific data (e.g., timestamps, confidence scores)."
    )

    # --- Spatial Information (Layout Layer) ---
    source_handle: Optional[NodeHandle] = Field(None, description="Starting point (n,s,e,w)")
    target_handle: Optional[NodeHandle] = Field(None, description="Ending point (n,s,e,w)")

    class Config:
        use_enum_values = True
        populate_by_name = True


class ConceptualGraph(BaseModel):
    """
    Represents the current ground truth of the canvas.
    """
    nodes: Dict[str, ConceptualNode] = Field(default_factory=dict)
    edges: List[ConceptualEdge] = Field(default_factory=list)


class ExpansionNodes(BaseModel):
    """
    A batch of new nodes and the intended growth strategy.
    """
    nodes: List[ConceptualNode]
    # Maps to Graphviz engines: dot (Hierarchical), twopi (Radial), fdp (Force)
    layout_intent: str = Field(default="twopi")
    # The reference node(s) from which the expansion originates
    anchor_ids: List[str] = Field(default_factory=list)


class SemanticGravity(BaseModel):
    """
    Physical constants for the force-directed layout.
    High values = High Tension (closer to anchor).
    Low values = Low Tension (peripheral).
    """
    focus: float = Field(default=1.0, ge=0.0, le=2.0)
    concept: float = Field(default=0.8, ge=0.0, le=2.0)
    insight: float = Field(default=0.5, ge=0.0, le=2.0)
    resource: float = Field(default=0.3, ge=0.0, le=2.0)
    query: float = Field(default=0.2, ge=0.0, le=2.0)
    group: float = Field(default=0.1, ge=0.0, le=2.0)


class SpatialLocateToolConfig(ToolConfig):
    """
    Global configuration for the SpatialLocateTool.
    """
    node_clearance: int = Field(default=5, description="Minimum pixel gap")
    max_iterations: int = Field(default=50, description="Physics refinement passes")
    semantic_gravity: SemanticGravity = Field(
        default_factory=SemanticGravity,
        description="Mass constants for different node types"
    )
    aspect_ratio: float = Field(default=1.7, description="aspect ratio for a conceptual node, used to calculate the position of the handle")


class GraphSynthesistAgentConfig(AgentConfig):
    tool_use: Literal['TOOL_USE_DIRECT', 'TOOL_USE_AND_PROCESS', 'NO_TOOL_USE'] = 'TOOL_USE_DIRECT'
    tool_configs: Dict[str, Any] = {
        'spatial_locate': SpatialLocateToolConfig()
    }
