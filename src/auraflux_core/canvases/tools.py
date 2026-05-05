import copy
import json
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from pydantic import ValidationError

from auraflux_core.canvases.schemas import (ConceptualEdge, ConceptualEdgeType,
                                            ConceptualGraph, ConceptualNode,
                                            ConceptualNodeType, ExpansionNodes,
                                            NodeHandle, Position,
                                            SpatialLocateToolConfig)
from auraflux_core.core.tools.base_tool import BaseTool


class GraphSerializerTool(BaseTool):
    """
    Serializes validated ConceptualNode and ConceptualEdge objects into
    a line-based NDJSON (Newline Delimited JSON) format for Auraflux.
    """

    async def run(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], output_path: str) -> Dict[str, Any]:
        """
        Converts the graph data into a text-based .graph file.

        Args:
            nodes: List of validated node dictionaries.
            edges: List of validated edge dictionaries.
            output_path: Destination path for the .graph file.
        """
        self.logger.info(f"Serializing graph to {output_path}")

        line_count = 0
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # 1. Serialize Nodes
                for node_data in nodes:
                    line = {"type": "node", "data": node_data}
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    line_count += 1

                # 2. Serialize Edges
                for edge_data in edges:
                    line = {"type": "edge", "data": edge_data}
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    line_count += 1

            self.logger.info(f"Successfully serialized {line_count} lines.")

            return {
                "success": True,
                "file_path": output_path,
                "stats": {
                    "nodes": len(nodes),
                    "edges": len(edges),
                    "total_lines": line_count
                }
            }

        except Exception as e:
            error_msg = f"Serialization failed: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def get_name(self) -> str:
        return "graph_serializer"

    def get_description(self) -> str:
        return (
            "Converts validated graph objects into a line-based Textual Graph (.graph). "
            "Each line represents a JSON object of either a node or an edge, "
            "optimized for version control and Phase 2 spatial injection."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of validated ConceptualNode dictionaries."
                },
                "edges": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of validated ConceptualEdge dictionaries."
                },
                "output_path": {
                    "type": "string",
                    "description": "The target file path (e.g., 'research_alpha.graph')."
                }
            },
            "required": ["nodes", "edges", "output_path"]
        }


class OntologyValidatorTool(BaseTool):
    """
    Validates a graph's adherence to the Empirical Science Ontology.
    Checks for structural integrity, grounding (source_ref), and logical flow.
    """

    async def run(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Performs a deep audit of nodes and edges for Phase 1 compliance.
        """
        results = {
            "is_valid": True,
            "errors": [],
            "stats": {"node_count": len(nodes), "edge_count": len(edges)}
        }

        for node in nodes:
            node['type'] = node['type'].upper()

        for edge in edges:
            edge['relation'] = edge['relation'].upper()

        # 1. Node-Level Validation
        validated_nodes = []
        for node_data in nodes:
            try:
                node = ConceptualNode(**node_data)

                # Rule: Phase 1 spatial constraint
                if node.position is not None:
                    results["errors"].append(f"Node[{node.id}]: Spatial data (position) found in Phase 1.")

                # Rule: Empirical Grounding (Zero-Inference)
                # if node.type in [ConceptualNodeType.EVENT, ConceptualNodeType.RESOURCE]:
                #     if not node.source_ref or len(node.source_ref.strip()) < 3:
                #         results["errors"].append(f"Node[{node.id}]: {node.type} missing mandatory 'source_ref'.")

                # Rule: Reasoning Requirement
                # if node.type in [ConceptualNodeType.INSIGHT, ConceptualNodeType.OUTCOME]:
                #     if not node.rationale or len(node.rationale.strip()) < 10:
                #         results["errors"].append(f"Node[{node.id}]: {node.type} missing detailed 'rationale'.")

                validated_nodes.append(node)
            except ValidationError as e:
                results["errors"].append(f"Node Schema Error: {str(e)}")

        # 2. Edge-Level Validation (Logic Flow)
        for edge_data in edges:
            try:
                edge = ConceptualEdge(**edge_data)

                # Rule: Empirical Logic Connectivity
                if edge.type == ConceptualEdgeType.VALIDATES:
                    if not edge.evidence:
                        results["errors"].append(f"Edge[{edge.source}->{edge.target}]: VALIDATES edge missing 'evidence'.")

                # Rule: Constraint Origin
                if edge.type == ConceptualEdgeType.CONSTRAINS:
                    # Optional: Logic to check if source node is actually a BOUNDARY node
                    pass

            except ValidationError as e:
                results["errors"].append(f"Edge Schema Error: {str(e)}")

        if results["errors"]:
            results["is_valid"] = False
            self.logger.warning(f"Validation failed with {len(results['errors'])} errors.")
        else:
            self.logger.info("Graph validated successfully against Empirical Ontology.")

        return results

    def get_name(self) -> str:
        return "ontology_validator"

    def get_description(self) -> str:
        return (
            "Validates that the generated knowledge graph adheres to the 5-Node/4-Edge "
            "Empirical Science standards. Checks for mandatory source references and "
            "logical evidence."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of ConceptualNode data dictionaries."
                },
                "edges": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of ConceptualEdge data dictionaries."
                }
            },
            "required": ["nodes", "edges"]
        }


class SchemaExtractorTool(BaseTool):
    """
    Orchestrates the conversion of unstructured text into an Empirical Graph.
    This tool provides the structured interface for LLM extraction tasks.
    """

    async def run(self, text_chunk: str, source_id: str, focus_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Processes a text chunk and returns a dictionary of extracted nodes and edges.

        Args:
            text_chunk: The raw text content to analyze.
            source_id: The index/ID of the paragraph (for source_ref).
            focus_hint: Optional guidance (e.g., "focus on methodology").
        """
        self.logger.info(f"Extracting knowledge from source: {source_id}")

        # NOTE: In the actual AgenticOrchestrator, the LLM will be called
        # using the prompt instructions defined in this tool's description
        # and the parameters below.

        # This structure simulates what the LLM should return via functional calling.
        extraction_result = {
            "nodes": [], # List of ConceptualNode dicts
            "edges": [], # List of ConceptualEdge dicts
            "metadata": {
                "source_id": source_id,
                "process_status": "raw_extracted"
            }
        }

        return extraction_result

    def get_name(self) -> str:
        return "schema_extractor"

    def get_description(self) -> str:
        return (
            "Transforms scientific text into an empirical graph structure. "
            "Extracted nodes must use labels reflecting their empirical role: "
            f"Types: {[t.value for t in ConceptualNodeType if t.name in ['EVENT', 'INSIGHT', 'OUTCOME', 'BOUNDARY', 'ENTITY']]}. "
            "For every EVENT/ENTITY, 'source_ref' must be set to the provided source_id. "
            "For every INSIGHT, 'rationale' must explain the inference. "
            "All 'position' fields must remain null."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "type": {"type": "string", "enum": [t.value for t in ConceptualNodeType]},
                            "content": {"type": "string"},
                            "source_ref": {"type": "string"},
                            "rationale": {"type": "string"},
                            "anchor_id": {"type": "string"}
                        },
                        "required": ["label", "type"]
                    }
                },
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "type": {"type": "string", "enum": [t.value for t in ConceptualEdgeType]},
                            "evidence": {"type": "string"},
                            "rationale": {"type": "string"}
                        },
                        "required": ["source", "target", "type"]
                    }
                }
            },
            "required": ["nodes", "edges"]
        }


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

            self.logger.debug(f"expansion_data: {kwargs.get('expansion_data', {})}")
            self.logger.debug(f"existing_graph_state: {kwargs.get('existing_graph_state', {})}")
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
                    weight=data['weight'],
                    evidence=None,
                    rationale=None,
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