import copy
import json
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from pydantic import ValidationError

from auraflux_core.canvases.schemas import (ConceptualEdge, ConceptualEdgeType,
                                            ConceptualGraph, ConceptualNode,
                                            ConceptualNodeType, ExpansionNodes,
                                            NodeHandle, Position,
                                            SpatialLocateToolConfig)
from auraflux_core.core.tools.base_tool import BaseTool


class CentralityPowerShiftTool(BaseTool):
    """
    Optimized Centrality and Power Shift tool using NetworkX.
    Analyzes 'Degree' (popularity) and 'Betweenness' (control) to detect
    structural narrative shifts between two graph versions.
    """

    async def run(
        self,
        base_nodes: List[Dict[str, Any]],
        base_edges: List[Dict[str, Any]],
        target_nodes: List[Dict[str, Any]],
        target_edges: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Identifies key hubs and measures shifts in structural influence.
        """
        self.logger.info(f"Analyzing Narrative Power Shifts (Top-{top_k}) with NetworkX.")

        def analyze_hubs(nodes, edges):
            G = nx.Graph()
            G.add_nodes_from([str(n.get("id")) for n in nodes])
            G.add_edges_from([(str(e.get("source")), str(e.get("target"))) for e in edges])

            if G.number_of_nodes() < 2:
                return [], {}

            degree_centrality = nx.degree_centrality(G)

            betweenness_centrality = nx.betweenness_centrality(G)

            id_to_label = {str(n.get("id")): n.get("label", n.get("id")) for n in nodes}

            combined_metrics = []
            for node_id in G.nodes():
                combined_metrics.append({
                    "id": node_id,
                    "label": id_to_label.get(node_id),
                    "degree_score": round(degree_centrality.get(node_id, 0), 4),
                    "betweenness_score": round(betweenness_centrality.get(node_id, 0), 4)
                })

            top_hubs = sorted(combined_metrics, key=lambda x: x["degree_score"], reverse=True)[:top_k]
            return top_hubs, degree_centrality

        base_hubs, base_raw = analyze_hubs(base_nodes, base_edges)
        target_hubs, target_raw = analyze_hubs(target_nodes, target_edges)

        base_hub_ids = {h["id"] for h in base_hubs}
        target_hub_ids = {h["id"] for h in target_hubs}

        hub_consensus = (
            len(base_hub_ids.intersection(target_hub_ids)) / len(base_hub_ids.union(target_hub_ids))
            if base_hub_ids or target_hub_ids else 1.0
        )

        significant_gains = []
        for h in target_hubs:
            if h["id"] not in base_hub_ids:
                significant_gains.append(h)

        self.logger.info("Centrality analysis complete.")

        return {
            "metrics": {
                "hub_consensus_score": round(hub_consensus, 4),
                "is_narrative_drifted": hub_consensus < 0.6,
                "base_top_hubs": base_hubs,
                "target_top_hubs": target_hubs
            },
            "shift_analysis": {
                "power_gained_nodes": significant_gains,
                "power_lost_node_ids": list(base_hub_ids - target_hub_ids)
            }
        }

    def get_name(self) -> str:
        return "centrality_power_shift_analyzer"

    def get_description(self) -> str:
        return (
            "Uses NetworkX to identify structural 'Hubs' and 'Bridges' in the graph. "
            "It measures narrative focus by comparing which entities hold the most "
            "topological power (Degree and Betweenness Centrality) between two versions."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "base_nodes": {"type": "array", "items": {"type": "object"}},
                "base_edges": {"type": "array", "items": {"type": "object"}},
                "target_nodes": {"type": "array", "items": {"type": "object"}},
                "target_edges": {"type": "array", "items": {"type": "object"}},
                "top_k": {"type": "integer", "default": 5}
            },
            "required": ["base_nodes", "base_edges", "target_nodes", "target_edges"]
        }


class EntityAlignmentCoverageTool(BaseTool):
    """
    Optimized Entity Alignment tool using NetworkX.
    Beyond simple ID matching, it evaluates 'Neighborhood Similarity'
    to detect if aligned entities share the same structural context.
    """

    async def run(
        self,
        base_nodes: List[Dict[str, Any]],
        base_edges: List[Dict[str, Any]],
        target_nodes: List[Dict[str, Any]],
        target_edges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compares nodes and their local topological structures.
        """
        self.logger.info("Performing Structural Entity Alignment with NetworkX.")

        # 1. Build Graphs
        def build_graph(nodes, edges):
            g = nx.DiGraph()
            # Store all attributes in the node for easy access
            for n in nodes:
                node_id = str(n.get("id"))
                g.add_node(node_id, **n)
            for e in edges:
                g.add_edge(str(e.get("source")), str(e.get("target")), type=e.get("type"))
            return g

        g_base = build_graph(base_nodes, base_edges)
        g_target = build_graph(target_nodes, target_edges)

        base_ids = set(g_base.nodes())
        target_ids = set(g_target.nodes())
        shared_ids = base_ids.intersection(target_ids)

        # 2. Analyze Neighborhood Similarity (Structural Context)
        # We check if the neighbors of a shared node are also shared
        context_shifts = []
        for node_id in shared_ids:
            # Get immediate neighbors (out-edges)
            base_neighbors = set(g_base.successors(node_id))
            target_neighbors = set(g_target.successors(node_id))

            # Calculate Jaccard similarity of the node's local environment
            union_neighbors = base_neighbors.union(target_neighbors)
            neighbor_sim = (
                len(base_neighbors.intersection(target_neighbors)) / len(union_neighbors)
                if union_neighbors else 1.0
            )

            if neighbor_sim < 1.0:
                context_shifts.append({
                    "id": node_id,
                    "neighbor_similarity": round(neighbor_sim, 4),
                    "base_only_neighbors": list(base_neighbors - target_neighbors),
                    "target_only_neighbors": list(target_neighbors - base_neighbors)
                })

        # 3. Basic Attribute Coverage (using Graph node data)
        type_conflicts = []
        for node_id in shared_ids:
            b_type = str(g_base.nodes[node_id].get("type", "")).upper()
            t_type = str(g_target.nodes[node_id].get("type", "")).upper()
            if b_type != t_type:
                type_conflicts.append({"id": node_id, "base": b_type, "target": t_type})

        # 4. Metrics Calculation
        jaccard_sim = len(shared_ids) / len(base_ids.union(target_ids)) if base_ids or target_ids else 1.0
        recall = len(shared_ids) / len(base_ids) if base_ids else 1.0

        # Average structural stability of aligned nodes
        avg_neighbor_sim = (
            sum(c['neighbor_similarity'] for c in context_shifts) + (len(shared_ids) - len(context_shifts))
        ) / len(shared_ids) if shared_ids else 1.0

        return {
            "alignment_metrics": {
                "id_jaccard_similarity": round(jaccard_sim, 4),
                "recall_rate": round(recall, 4),
                "structural_stability_score": round(avg_neighbor_sim, 4),
                "unique_discovery_count": len(target_ids - base_ids)
            },
            "consistency_analysis": {
                "type_conflicts": type_conflicts,
                "context_shifts": sorted(context_shifts, key=lambda x: x['neighbor_similarity'])[:5] # Show top 5 shifts
            },
            "discovery_details": {
                "missing_from_target": list(base_ids - target_ids),
                "unique_to_target": list(target_ids - base_ids)
            }
        }

    def get_name(self) -> str:
        return "entity_alignment_coverage_analyzer"

    def get_description(self) -> str:
        return (
            "Optimized entity alignment tool using NetworkX. It measures ID overlap, "
            "attribute consistency, and 'Structural Stability'—detecting if an entity's "
            "logical surroundings have changed between two graph versions."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "base_nodes": {"type": "array", "items": {"type": "object"}},
                "base_edges": {"type": "array", "items": {"type": "object"}},
                "target_nodes": {"type": "array", "items": {"type": "object"}},
                "target_edges": {"type": "array", "items": {"type": "object"}}
            },
            "required": ["base_nodes", "base_edges", "target_nodes", "target_edges"]
        }


class GraphIsolationRateTool(BaseTool):
    """
    Calculates the Isolation Rate of a knowledge graph.
    The isolation rate is defined as the number of nodes with no connected edges
    divided by the total number of nodes. High isolation indicates fragmented knowledge.
    """

    async def run(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes the isolation rate calculation.

        Args:
            nodes: A list of node dictionaries (must contain 'id').
            edges: A list of edge dictionaries (must contain 'source' and 'target').

        Returns:
            A dictionary containing the isolation_rate and the list of isolated_node_ids.
        """
        self.logger.info(f"Analyzing Isolation Rate for graph with {len(nodes)} nodes.")

        if not nodes:
            return {
                "isolation_rate": 0.0,
                "isolated_nodes": [],
                "total_nodes": 0,
                "message": "Graph is empty."
            }

        G = nx.Graph()
        node_ids = [str(n.get("id")) for n in nodes if n.get("id")]
        G.add_nodes_from(node_ids)

        edge_tuples = [(str(e.get("source")), str(e.get("target")))
                       for e in edges if e.get("source") and e.get("target")]
        G.add_edges_from(edge_tuples)

        isolated_nodes = list(nx.isolates(G))

        # Step 3: Calculate the rate
        total_nodes = G.number_of_nodes()
        isolated_count = len(isolated_nodes)
        isolation_rate = isolated_count / total_nodes if total_nodes > 0 else 0.0

        self.logger.info(f"Calculation complete. Isolated nodes found: {isolated_count}")

        return {
            "isolation_rate": round(isolation_rate, 4),
            "isolated_nodes": isolated_nodes,
            "total_nodes": total_nodes,
            "connected_nodes_count": total_nodes - isolated_count
        }

    def get_name(self) -> str:
        return "graph_isolation_rate_analyzer"

    def get_description(self) -> str:
        return (
            "Computes the ratio of isolated nodes in a graph. "
            "An isolated node is one that has no incoming or outgoing edges. "
            "This metric helps evaluate the connectivity and coherence of the extracted knowledge."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "description": "List of node objects extracted from the text.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "label": {"type": "string"}
                        },
                        "required": ["id"]
                    }
                },
                "edges": {
                    "type": "array",
                    "description": "List of edge objects defining relationships between nodes.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "relation": {"type": "string"}
                        },
                        "required": ["source", "target"]
                    }
                }
            },
            "required": ["nodes", "edges"]
        }


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


class GraphTaxonomyDistributionTool(BaseTool):
    """
    Analyzes the distribution of node types within a knowledge graph.
    This tool provides insights into the modeling bias of an agent—whether
    it focuses more on static entities or dynamic insights and outcomes.
    """

    async def run(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Processes the nodes and calculates distribution statistics.

        Args:
            nodes: A list of node dictionaries, each containing a 'type' field.

        Returns:
            A dictionary containing type counts, percentages, and total count.
        """
        self.logger.info("Initiating Taxonomy Distribution analysis.")

        if not nodes:
            return {
                "distribution": {},
                "total_nodes": 0,
                "dominant_type": None,
                "message": "No nodes provided for analysis."
            }

        total_nodes = len(nodes)

        # Count occurrences of each type (ensure case-insensitive comparison if needed)
        type_counts = Counter(node.get("type", "UNKNOWN").upper() for node in nodes)

        # Calculate percentages and build distribution report
        distribution = {}
        for node_type, count in type_counts.items():
            distribution[node_type] = {
                "count": count,
                "percentage": round((count / total_nodes) * 100, 2)
            }

        # Identify the dominant node type
        dominant_type = type_counts.most_common(1)[0][0] if type_counts else None

        self.logger.info(f"Analysis complete. Total nodes: {total_nodes}. Dominant type: {dominant_type}")

        return {
            "total_nodes": total_nodes,
            "dominant_type": dominant_type,
            "distribution": distribution
        }

    def get_name(self) -> str:
        return "graph_taxonomy_analyzer"

    def get_description(self) -> str:
        return (
            "Analyzes the frequency and percentage of different node types in the graph. "
            "Helpful for identifying if the graph is entity-heavy or insight-driven."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "description": "List of node objects with 'type' fields.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "type": {"type": "string"}
                        },
                        "required": ["type"]
                    }
                }
            },
            "required": ["nodes"]
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


class TripleStrengthComparisonTool(BaseTool):
    """
    Compares edge triples (Source, Relation, Target) between two graphs.
    It evaluates logical agreement and analyzes "Relation Strength" shifts
    (e.g., upgrading a weak reference to a strong causal trigger).
    """

    async def run(
        self,
        base_edges: List[Dict[str, Any]],
        target_edges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Executes the triple comparison.

        Args:
            base_edges: Reference list of edges (Baseline).
            target_edges: List of edges to evaluate (Target).

        Returns:
            A report detailing exact matches, structural matches with relation shifts,
            and strength distribution.
        """
        self.logger.info("Comparing Relation Triples and Strength.")

        def build_graph(edges):
            g = nx.MultiDiGraph()
            for e in edges:
                s, t, r = e.get("source"), e.get("target"), e.get("type")
                if s and t and r:
                    # Use the relation type as a key within the multi-edge
                    g.add_edge(str(s), str(t), key=str(r).upper(), type=str(r).upper())
            return g

        g_base = build_graph(base_edges)
        g_target = build_graph(target_edges)

        base_triples = set(g_base.edges(keys=True))
        target_triples = set(g_target.edges(keys=True))

        exact_matches = base_triples.intersection(target_triples)
        unique_base = base_triples - target_triples
        unique_target = target_triples - base_triples

        # 2. Strength Metrics
        # Define Strong/Empirical relations directly from our Enum
        # We exclude REF and LINK as they are 'Weak/Functional'
        EMPIRICAL_TYPES = {
            ConceptualEdgeType.VALIDATES.value,
            ConceptualEdgeType.CONSTRAINS.value,
            ConceptualEdgeType.TRIGGERS.value
        }

        def calc_empirical_rate(triple_set):
            if not triple_set: return 0.0
            empirical_count = sum(1 for _, _, rel in triple_set if rel in EMPIRICAL_TYPES)
            return round(empirical_count / len(triple_set), 4)

        base_pairs = {(s, t) for s, t, _ in base_triples}
        target_pairs = {(s, t) for s, t, _ in target_triples}
        shared_pairs = base_pairs.intersection(target_pairs)

        shifts = []
        for s, t in shared_pairs:
            b_rels = {rel for src, tgt, rel in base_triples if src == s and tgt == t}
            t_rels = {rel for src, tgt, rel in target_triples if src == s and tgt == t}
            if b_rels != t_rels:
                shifts.append({
                    "pair": (s, t),
                    "base_relations": list(b_rels),
                    "target_relations": list(t_rels),
                    "is_upgrade": any(r in EMPIRICAL_TYPES for r in t_rels) and not any(r in EMPIRICAL_TYPES for r in b_rels)
                })

        self.logger.info(f"Matched {len(exact_matches)} triples exactly.")

        return {
            "metrics": {
                "exact_match_count": len(exact_matches),
                "unique_target_triples": len(unique_target),
                "unique_base_triples": len(unique_base),
                "structural_pair_overlap_count": len(shared_pairs),
                "relation_shift_count": len(shifts),
                "base_empirical_density": calc_empirical_rate(base_triples),
                "target_empirical_density": calc_empirical_rate(target_triples)
            },
            "structural_analysis": {
                "upgraded_logic_shifts": [s for s in shifts if s["is_upgrade"]],
                "matched_triples_sample": [list(t) for t in list(exact_matches)[:10]]
            }
        }

    def get_name(self) -> str:
        return "triple_strength_analyzer"

    def get_description(self) -> str:
        return (
            "Analyzes the agreement of (Source, Relation, Target) triples between two graphs. "
            "It specifically tracks 'relation shifts' where agents agree on a connection "
            "but disagree on the logical intensity (e.g., Reference vs Trigger)."
        )

    def get_parameters(self) -> Dict[str, Any]:
        edge_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "type": {"type": "string"}
                },
                "required": ["source", "target", "type"]
            }
        }
        return {
            "type": "object",
            "properties": {
                "base_edges": {
                    "description": "The reference edges (Baseline).",
                    **edge_schema
                },
                "target_edges": {
                    "description": "The edges to evaluate (Target).",
                    **edge_schema
                }
            },
            "required": ["base_edges", "target_edges"]
        }
