from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from auraflux_core.core.schemas.agents import AgentConfig
from auraflux_core.core.schemas.tools import ToolConfig


class ConceptualNodeType(str, Enum):
    FOCUS = "FOCUS"
    RESOURCE = "RESOURCE"
    CONCEPT = "CONCEPT"
    INSIGHT = "INSIGHT"
    QUERY = "QUERY"
    NAVIGATION = "NAVIGATION"
    GROUP = "GROUP"


class NodeHandle(str, Enum):
    NORTH = 'n'
    EAST = 'e'
    WEST = 'w'
    SOUTH = 's'


class Position(BaseModel):
    x: float
    y: float


class ConceptualNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    label: str
    type: ConceptualNodeType
    position: Optional[Position] = None

    # Metadata for AI reasoning/UI feedback
    rationale: Optional[str] = None
    anchor_id: Optional[str] = None


class ConceptualEdge(BaseModel):
    """
    Minimal representation of a directed or undirected relationship.
    """
    source: str
    source_handle: Optional[NodeHandle] = Field(default=None)
    target: str
    target_handle: Optional[NodeHandle] = Field(default=None)
    weight: Optional[float] = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
