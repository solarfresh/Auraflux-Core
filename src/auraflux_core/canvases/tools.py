import copy
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from auraflux_core.canvases.schemas import (ConceptualEdge, ConceptualGraph, NodeHandle,
                                            ConceptualNodeType, ExpansionNodes,
                                            Position, SpatialLocateToolConfig)
from auraflux_core.core.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)


class SpatialLocateTool(BaseTool):
    """
    Calculates absolute coordinates for new graph nodes using Graphviz engines
    and spring-force refinement to ensure collision-free placement.
    """
    def __init__(self, config: SpatialLocateToolConfig):
        super().__init__(config=config)
        self.node_clearance = config.node_clearance
        self.max_iterations = config.max_iterations
        self.semantic_gravity = config.semantic_gravity
        self.aspect_ratio = config.aspect_ratio

    async def run(self, **kwargs) -> str:
        """
        Executes the spatial mapping of new nodes onto the existing graph.

        Args:
            expansion_data: AI-generated nodes, anchor_ids, and layout_intent.
            existing_graph_state: The current {nodes, edges} data from the canvas.

        Returns:
            JSON string mapping New_Node_IDs to finalized {x, y} coordinates.
        """
        try:
            expansion = ExpansionNodes(**kwargs.get('expansion_data', {}))
            graph_state = ConceptualGraph(**kwargs.get('existing_graph_state', {}))
            existing_node_ids = list(graph_state.nodes.keys())

            logger.debug(f"expansion_data: {kwargs.get('expansion_data', {})}")
            logger.debug(f"existing_graph_state: {kwargs.get('existing_graph_state', {})}")
            if not expansion.nodes:
                return json.dumps({"error": "Expansion batch is empty."})

            if existing_node_ids:
                anchor_ids = set([node.anchor_id for node in expansion.nodes if node.anchor_id])
                for anchor_id in anchor_ids:
                    expansion.nodes.append(copy.deepcopy(graph_state.nodes[anchor_id]))

            # Generates local geometry based on AI's 'layout_intent' (dot, twopi, etc.)
            sim_pos = self._simulate_semantic_topology(expansion)

            # Maps relative Graphviz coordinates to absolute Canvas coordinates
            # Returns: (dx, dy) offset
            offset = self._calculate_position_offset(
                graph_state=graph_state,
                existing_node_ids=existing_node_ids,
                simulated_pos=sim_pos
            )

            # Reconstructs G and injects new nodes with projected 'coords'
            # Returns: NetworkX Graph object 'G'
            G = self._build_nx_graph(
                expansion=sim_pos,
                existing_graph_state=graph_state,
                offset=offset
            )

            # Use spring_layout to fine-tune 'new_node' positions while locking 'existing_nodes'
            # init_pos: Dictionary of existing 'position' from G.nodes
            init_pos = {node_id: G.nodes[node_id]['position'] for node_id in G.nodes()}

            # Fine-tune to prevent overlap within the batch
            final_layout = nx.spring_layout(
                G,
                pos=init_pos,
                fixed=existing_node_ids if existing_node_ids else None,
                k=self.node_clearance,
                iterations=self.max_iterations,
                scale=None
            )

            for node in expansion.nodes:
                node.position = Position(
                    x=float(final_layout[node.id][0]),
                    y=float(final_layout[node.id][1])
                )
                graph_state.nodes[node.id] = node

            for source, taget, data in G.edges(data=True):
                source_handle, target_handle = self._calculate_node_handle(
                    source_position=graph_state.nodes[source].position,
                    target_position=graph_state.nodes[taget].position
                )
                conceptual_edge = ConceptualEdge(
                    source=source,
                    source_handle=source_handle,
                    target=taget,
                    target_handle=target_handle,
                    weight=data['weight']
                )
                graph_state.edges.append(conceptual_edge)

            self.logger.info(f"SpatialLocateTool: Successfully mapped {len(expansion.nodes)} nodes.")
            return graph_state.model_dump_json()

        except Exception as e:
            self.logger.error(f"SpatialLocateTool error: {str(e)}")
            return json.dumps({"error": str(e)})

    def _build_nx_graph(
            self,
            expansion: ExpansionNodes,
            existing_graph_state: Optional[ConceptualGraph] = None,
            offset: Optional[tuple[float, float]] = None
        ) -> nx.Graph:
        """
        Populates the NetworkX graph with new nodes and establishes
        weighted semantic edges
        """
        G = nx.Graph()
        dx, dy = (0., 0.) if offset is None else offset

        if existing_graph_state is not None:
            for node_id, node in existing_graph_state.nodes.items():
                if node.position is None:
                    raise ValueError("All nodes in graph_state must have defined positions for offset calculation.")

                G.add_node(
                    node_id,
                    label=node.label,
                    type=node.type,
                    rationale=node.rationale,
                    position=(node.position.x, node.position.y),
                )

            for edge in existing_graph_state.edges:
                G.add_edge(edge.source, edge.target, weight=edge.weight)

        for node in expansion.nodes:
            aligned_position = None
            if node.position is not None:
                raw_x = node.position.x
                raw_y = node.position.y
                aligned_position = (raw_x + dx, raw_y + dy)

            if node.id in G.nodes():
                G.nodes[node.id]['position'] = aligned_position
            else:
                G.add_node(
                    node.id,
                    label=node.label,
                    type=node.type,
                    rationale=node.rationale,
                    position=aligned_position
                )

        # Add Edges with Semantic Tension
        for node in expansion.nodes:
            if node.anchor_id is not None and node.anchor_id in G.nodes():
                semantic_weight = getattr(self.semantic_gravity, node.type.lower())
                G.add_edge(node.anchor_id, node.id, weight=float(semantic_weight))

        return G

    def _calculate_node_handle(
        self,
        source_position: Position | None,
        target_position: Position | None,
    ) -> Tuple[Optional[NodeHandle], Optional[NodeHandle]]:
        if source_position is None or target_position is None:
            return None, None

        reciprocal_slope = (target_position.x - source_position.x) / (target_position.y - source_position.y)
        if abs(reciprocal_slope) < self.aspect_ratio:
            if target_position.y > source_position.y:
                return NodeHandle.SOUTH, NodeHandle.NORTH
            else:
                return NodeHandle.NORTH, NodeHandle.SOUTH
        else:
            if target_position.x > source_position.x:
                return NodeHandle.EAST, NodeHandle.WEST
            else:
                return NodeHandle.WEST, NodeHandle.EAST

    def _calculate_position_offset(
        self,
        graph_state: ConceptualGraph,
        existing_node_ids: List[str],
        simulated_pos: ExpansionNodes
    ) -> tuple[float, float]:
        """
        Calculates the (X, Y) translation delta required to align the
        simulated subgraph with the current canvas coordinates.
        """

        for node in simulated_pos.nodes:
            if node.position is None:
                raise ValueError("All nodes in simulated_pos must have defined positions for offset calculation.")

            if existing_node_ids and (node.id in existing_node_ids):
                canvas_node = graph_state.nodes[node.id]
                if canvas_node.position is None:
                    continue

                return (canvas_node.position.x - node.position.x, canvas_node.position.y - node.position.y)
            elif node.type.lower() == ConceptualNodeType.FOCUS.value.lower():
                return (-node.position.x, -node.position.y)

        raise ValueError("No anchor nodes found for offset calculation, and no FOCUS node present as fallback.")

    def _simulate_semantic_topology(self, expansion: ExpansionNodes) -> ExpansionNodes:
        layout_intent = expansion.layout_intent
        G = self._build_nx_graph(expansion=expansion)
        prog_engine = layout_intent.lower() if layout_intent else 'neato'

        try:
            pos = nx.nx_agraph.graphviz_layout(
                G,
                prog=prog_engine,
            )
        except Exception as e:
            self.logger.warning(f"Layout engine {prog_engine} failed: {e}")
            pos = nx.spring_layout(
                G,
                k=self.node_clearance,
                iterations=self.max_iterations,
                scale=None
            )

        for node in expansion.nodes:
            if node.id in pos:
                position = pos[node.id]
                node.position = Position(x=position[0], y=position[1])

        return expansion

    def get_name(self) -> str:
        return "spatial_locate"

    def get_description(self) -> str:
        return "Transforms semantic expansion nodes into absolute (X, Y) coordinates."

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expansion_data": {
                    "type": "object",
                    "description": "The JSON object containing recommended nodes and layout_intent."
                },
                "existing_graph_state": {
                    "type": "object",
                    "description": "Current coordinates of nodes on the canvas to prevent overlaps."
                }
            },
            "required": ["expansion_data", "existing_graph_state"]
        }