"""
Graph data structures for the Request Confirmation Network (ReCoN).

This module defines the core data structures used to represent the network topology:
- Unit: Individual computational elements with state and activation
- Edge: Directed connections between units with typed relationships
- Graph: Container for units and edges with utility methods
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .enums import LinkType, Message, State, UnitType

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


@dataclass
class Unit:
    """
    A computational unit in the ReCoN network.

    Units are the basic elements that process information and communicate via messages.
    Each unit has a type (SCRIPT or TERMINAL), maintains internal state, and participates
    in the network's activation dynamics.

    Attributes:
        id: Unique identifier for this unit
        kind: Type of unit (SCRIPT for orchestration, TERMINAL for perception)
        state: Current activation state (INACTIVE, REQUESTED, ACTIVE, etc.)
        a: Activation level (0.0 to 1.0) representing confidence/evidence
        thresh: Threshold for confirmation (default 0.5)
        meta: Dictionary for additional metadata and configuration
        inbox: Queue of incoming messages as (sender_id, message) tuples
        outbox: Queue of outgoing messages as (receiver_id, message) tuples
    """

    id: str
    """Unique identifier for this unit in the network."""

    kind: UnitType
    """Type of unit: SCRIPT (orchestration) or TERMINAL (perception)."""

    state: State = State.INACTIVE
    """Current activation state of the unit."""

    a: float = 0.0
    """Activation level (0.0-1.0) representing confidence/evidence strength."""

    thresh: float = 0.5
    """Threshold for state transitions (confirmation threshold)."""

    meta: dict = field(default_factory=dict)
    """Additional metadata and configuration parameters."""

    inbox: List[Tuple[str, Message]] = field(default_factory=list)
    """Incoming message queue: list of (sender_id, message) tuples."""

    outbox: List[Tuple[str, Message]] = field(default_factory=list)
    """Outgoing message queue: list of (receiver_id, message) tuples."""


@dataclass
class Edge:
    """
    A directed connection between units in the ReCoN network.

    Edges define relationships and information flow between units. Different link types
    (SUB, SUR, POR, RET) determine how activation and messages propagate through
    the network.

    Attributes:
        src: Source unit ID (origin of the edge)
        dst: Destination unit ID (target of the edge)
        type: Type of relationship (SUB, SUR, POR, RET)
        w: Edge weight (strength of connection, default 1.0)
    """

    src: str
    """Source unit ID (unit where this edge originates)."""

    dst: str
    """Destination unit ID (unit where this edge terminates)."""

    type: LinkType
    """Type of relationship: SUB (evidence), SUR (request), POR (temporal), RET (feedback)."""

    w: float = 1.0
    """Connection strength/weight for activation propagation."""


class Graph:
    """
    Container for units and edges representing a complete ReCoN network topology.

    The Graph class manages the network structure and provides utility methods for
    traversing relationships between units. It maintains both forward (outgoing) and
    backward (incoming) edge mappings for efficient graph operations.

    Attributes:
        units: Dictionary mapping unit IDs to Unit objects
        out_edges: Dictionary mapping unit IDs to lists of outgoing edges
        in_edges: Dictionary mapping unit IDs to lists of incoming edges
    """

    def __init__(self):
        """Initialize an empty graph with no units or edges."""
        self.units: Dict[str, Unit] = {}
        self.out_edges: Dict[str, List[Edge]] = {}
        self.in_edges: Dict[str, List[Edge]] = {}
        self._custom_validation_rules: Dict[str, Any] = {}

    def add_unit(self, u: Unit):
        """
        Add a unit to the graph.

        Initializes edge lists for the new unit in both outgoing and incoming
        edge dictionaries.

        Args:
            u: Unit object to add to the graph
        """
        self.units[u.id] = u
        self.out_edges.setdefault(u.id, [])
        self.in_edges.setdefault(u.id, [])

    def add_edge(self, e: Edge):
        """
        Add a directed edge between existing units.

        The edge connects the source unit to the destination unit with the
        specified relationship type and weight.

        Args:
            e: Edge object defining the connection

        Raises:
            AssertionError: If source or destination units don't exist in the graph
        """
        assert (
            e.src in self.units and e.dst in self.units
        ), "Both source and destination units must exist"
        self.out_edges[e.src].append(e)
        self.in_edges[e.dst].append(e)

    def neighbors(self, u_id: str, direction: str = "out") -> List[Edge]:
        """
        Get all edges connected to a unit in the specified direction.

        Args:
            u_id: Unit ID to get neighbors for
            direction: 'out' for outgoing edges, 'in' for incoming edges

        Returns:
            List of Edge objects connected to the specified unit
        """
        return (self.out_edges if direction == "out" else self.in_edges).get(u_id, [])

    def sub_children(self, parent_id: str) -> List[str]:
        """
        Get IDs of all child units connected via SUB (part-of) links.

        SUB links represent hierarchical relationships where children provide
        evidence to their parent units.

        Args:
            parent_id: ID of the parent unit

        Returns:
            List of child unit IDs connected via SUB links
        """
        return [e.src for e in self.in_edges[parent_id] if e.type == LinkType.SUB]

    def por_successors(self, u_id: str) -> List[str]:
        """
        Get IDs of successor units connected via POR (temporal precedence) links.

        POR links define sequential ordering in temporal processes.

        Args:
            u_id: ID of the predecessor unit

        Returns:
            List of successor unit IDs connected via POR links
        """
        return [e.dst for e in self.out_edges[u_id] if e.type == LinkType.POR]

    def sur_children(self, parent_id: str) -> List[str]:
        """
        Get IDs of child units connected via SUR (request) links.

        SUR links enable top-down request propagation from parent to children.

        Args:
            parent_id: ID of the parent unit

        Returns:
            List of child unit IDs connected via SUR links
        """
        return [e.dst for e in self.out_edges[parent_id] if e.type == LinkType.SUR]

    def to_networkx(self) -> "nx.DiGraph":
        """
        Convert the ReCoN graph to a NetworkX DiGraph for export/visualization.

        Returns:
            NetworkX DiGraph with nodes and edges representing the ReCoN network

        Raises:
            ImportError: If NetworkX is not available
        """
        if not HAS_NETWORKX:
            raise ImportError(
                "NetworkX is required for graph conversion. Install with: pip install networkx"
            )

        G = nx.DiGraph()

        # Add nodes with their attributes
        for unit_id, unit in self.units.items():
            node_attrs = {
                "kind": str(unit.kind.name),  # Convert enum to string name
                "state": str(unit.state.name),  # Convert enum to string name
                "activation": unit.a,
                "threshold": unit.thresh,
            }
            # Add meta attributes if they exist
            if unit.meta:
                for k, v in unit.meta.items():
                    node_attrs[f"meta_{k}"] = v

            G.add_node(unit_id, **node_attrs)

        # Add edges with their attributes
        for unit_id, edges in self.out_edges.items():
            for edge in edges:
                edge_attrs = {
                    "type": str(edge.type.name),  # Convert enum to string name
                    "weight": edge.w,
                }
                G.add_edge(edge.src, edge.dst, **edge_attrs)

        return G

    def export_graphml(self, filepath: str) -> None:
        """
        Export the ReCoN graph to GraphML format.

        GraphML is an XML-based graph format that preserves node and edge attributes,
        making it suitable for import into graph analysis tools, visualization software,
        and other graph processing libraries.

        Args:
            filepath: Path where to save the GraphML file

        Raises:
            ImportError: If NetworkX is not available
        """
        nx_graph = self.to_networkx()
        nx.write_graphml(nx_graph, filepath)

    def validate_cycles(
        self, link_type: LinkType | None = None
    ) -> Dict[str, List[str]]:
        """
        Detect cycles in the graph for specified link types.

        For ReCoN networks:
        - SUB cycles: Generally problematic (infinite evidence propagation)
        - SUR cycles: Generally problematic (infinite request propagation)
        - POR cycles: May be acceptable for circular temporal processes
        - RET cycles: Expected and normal for feedback loops

        Args:
            link_type: Specific link type to check, or None for all types

        Returns:
            Dictionary mapping link types to lists of cycle descriptions
        """
        issues = {}

        if link_type is None or link_type == LinkType.SUB:
            sub_cycles = self._detect_cycles(LinkType.SUB)
            if sub_cycles:
                issues["SUB"] = sub_cycles

        if link_type is None or link_type == LinkType.SUR:
            sur_cycles = self._detect_cycles(LinkType.SUR)
            if sur_cycles:
                issues["SUR"] = sur_cycles

        if link_type is None or link_type == LinkType.POR:
            por_cycles = self._detect_cycles(LinkType.POR)
            if por_cycles:
                issues["POR"] = por_cycles

        if link_type is None or link_type == LinkType.RET:
            ret_cycles = self._detect_cycles(LinkType.RET)
            if ret_cycles:
                issues["RET"] = ret_cycles

        return issues

    def _detect_cycles(self, link_type: LinkType) -> List[str]:
        """
        Detect cycles for a specific link type using DFS.

        Args:
            link_type: The link type to check for cycles

        Returns:
            List of cycle descriptions as strings
        """
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node_id: str, path: List[str]) -> None:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            # Get successors for this link type
            successors = []
            if link_type == LinkType.SUB:
                successors = [
                    e.src for e in self.in_edges.get(node_id, []) if e.type == link_type
                ]
            elif link_type == LinkType.SUR:
                successors = [
                    e.dst
                    for e in self.out_edges.get(node_id, [])
                    if e.type == link_type
                ]
            elif link_type == LinkType.POR:
                successors = [
                    e.dst
                    for e in self.out_edges.get(node_id, [])
                    if e.type == link_type
                ]
            elif link_type == LinkType.RET:
                successors = [
                    e.src for e in self.in_edges.get(node_id, []) if e.type == link_type
                ]

            for successor in successors:
                if successor not in visited:
                    dfs(successor, path[:])
                elif successor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(successor)
                    cycle = path[cycle_start:] + [successor]
                    cycles.append(" -> ".join(cycle))

            path.pop()
            rec_stack.remove(node_id)

        for unit_id in self.units:
            if unit_id not in visited:
                dfs(unit_id, [])

        return cycles

    def validate_graph_integrity(self) -> Dict[str, List[str]]:
        """
        Comprehensive graph integrity validation.

        Checks for various structural issues that could cause problems:
        - Orphaned units (no connections)
        - Isolated components
        - Invalid edge references
        - Missing required connections

        Returns:
            Dictionary of validation issues by category
        """
        issues = {
            "orphaned_units": [],
            "invalid_edges": [],
            "connectivity_issues": [],
            "structure_warnings": [],
        }

        # Check for orphaned units
        for unit_id, _ in self.units.items():
            has_connections = (
                unit_id in self.out_edges and self.out_edges[unit_id]
            ) or (unit_id in self.in_edges and self.in_edges[unit_id])
            if not has_connections:
                issues["orphaned_units"].append(f"Unit '{unit_id}' has no connections")

        # Check edge validity
        for unit_id, edges in self.out_edges.items():
            for edge in edges:
                if edge.src != unit_id:
                    issues["invalid_edges"].append(
                        f"Edge source mismatch: {edge.src} != {unit_id}"
                    )
                if edge.src not in self.units:
                    issues["invalid_edges"].append(
                        f"Edge references non-existent source: {edge.src}"
                    )
                if edge.dst not in self.units:
                    issues["invalid_edges"].append(
                        f"Edge references non-existent destination: {edge.dst}"
                    )

        for unit_id, edges in self.in_edges.items():
            for edge in edges:
                if edge.dst != unit_id:
                    issues["invalid_edges"].append(
                        f"Edge destination mismatch: {edge.dst} != {unit_id}"
                    )

        # Check connectivity
        connected_components = self._find_connected_components()
        if len(connected_components) > 1:
            issues["connectivity_issues"].append(
                f"Graph has {len(connected_components)} disconnected components"
            )

        # Structure warnings
        terminals = [
            uid for uid, u in self.units.items() if u.kind == UnitType.TERMINAL
        ]

        # Check if terminals have proper evidence flow
        for term_id in terminals:
            has_sub_links = any(
                e.type == LinkType.SUB for e in self.out_edges.get(term_id, [])
            )
            if not has_sub_links:
                issues["structure_warnings"].append(
                    f"Terminal '{term_id}' has no SUB (evidence) links to parent scripts"
                )

        return {k: v for k, v in issues.items() if v}  # Remove empty categories

    def _find_connected_components(self) -> List[List[str]]:
        """
        Find all connected components in the graph (ignoring edge direction and type).

        Returns:
            List of lists, where each inner list contains unit IDs in one component
        """
        visited = set()
        components = []

        def dfs_component(node_id: str, component: List[str]) -> None:
            visited.add(node_id)
            component.append(node_id)

            # Visit all neighbors (both directions, all link types)
            for edges in [
                self.out_edges.get(node_id, []),
                self.in_edges.get(node_id, []),
            ]:
                for edge in edges:
                    neighbor = edge.dst if edge.src == node_id else edge.src
                    if neighbor not in visited:
                        dfs_component(neighbor, component)

        for unit_id in self.units:
            if unit_id not in visited:
                component = []
                dfs_component(unit_id, component)
                components.append(sorted(component))

        return components

    def validate_link_consistency(self) -> Dict[str, List[str]]:
        """
        Validate that link types are used consistently with unit types.

        ReCoN link type rules:
        - SUB links: Should connect terminals to scripts (evidence flow)
        - SUR links: Should connect scripts to their children (request flow)
        - POR links: Should only connect scripts (temporal sequencing)
        - RET links: Should only connect scripts (temporal feedback)

        Returns:
            Dictionary of validation issues by category
        """
        issues = {
            "sub_link_issues": [],
            "sur_link_issues": [],
            "por_link_issues": [],
            "ret_link_issues": [],
            "edge_weight_issues": [],
        }

        for unit_id, edges in self.out_edges.items():
            src_unit = self.units.get(unit_id)
            if not src_unit:
                continue

            for edge in edges:
                dst_unit = self.units.get(edge.dst)
                if not dst_unit:
                    continue

                # SUB link validation
                if edge.type == LinkType.SUB:
                    # SUB should flow from terminals to scripts (evidence)
                    if src_unit.kind != UnitType.TERMINAL:
                        issues["sub_link_issues"].append(
                            f"SUB link from {src_unit.kind.name} '{unit_id}' to "
                            f"{dst_unit.kind.name} '{edge.dst}' - SUB should originate from terminals"
                        )
                    if dst_unit.kind != UnitType.SCRIPT:
                        issues["sub_link_issues"].append(
                            f"SUB link from {src_unit.kind.name} '{unit_id}' to "
                            f"{dst_unit.kind.name} '{edge.dst}' - SUB should target scripts"
                        )

                # SUR link validation
                elif edge.type == LinkType.SUR:
                    # SUR should flow from scripts to their children
                    if src_unit.kind != UnitType.SCRIPT:
                        issues["sur_link_issues"].append(
                            f"SUR link from {src_unit.kind.name} '{unit_id}' to "
                            f"{dst_unit.kind.name} '{edge.dst}' - SUR should originate from scripts"
                        )

                # POR link validation
                elif edge.type == LinkType.POR:
                    # POR should only connect scripts (temporal precedence)
                    if (
                        src_unit.kind != UnitType.SCRIPT
                        or dst_unit.kind != UnitType.SCRIPT
                    ):
                        issues["por_link_issues"].append(
                            f"POR link between {src_unit.kind.name} '{unit_id}' and "
                            f"{dst_unit.kind.name} '{edge.dst}' - POR should only connect scripts"
                        )

                # RET link validation
                elif edge.type == LinkType.RET:
                    # RET should only connect scripts (temporal feedback)
                    if (
                        src_unit.kind != UnitType.SCRIPT
                        or dst_unit.kind != UnitType.SCRIPT
                    ):
                        issues["ret_link_issues"].append(
                            f"RET link between {src_unit.kind.name} '{unit_id}' and "
                            f"{dst_unit.kind.name} '{edge.dst}' - RET should only connect scripts"
                        )

                # Edge weight validation
                if not (0.0 <= edge.w <= 1.0):
                    issues["edge_weight_issues"].append(
                        f"Edge weight {edge.w} for {edge.type.name} link from '{unit_id}' to "
                        f"'{edge.dst}' is outside valid range [0.0, 1.0]"
                    )

        return {k: v for k, v in issues.items() if v}  # Remove empty categories

    def validate_unit_relationships(self) -> Dict[str, List[str]]:
        """
        Validate unit type relationships and structural constraints.

        Checks:
        - Terminals should only participate in SUB/SUR relationships
        - Scripts should form proper hierarchical structures
        - Root scripts should exist and be properly connected
        - Leaf terminals should be reachable from root scripts

        Returns:
            Dictionary of validation issues by category
        """
        issues = {
            "terminal_relationships": [],
            "script_hierarchy": [],
            "root_script_issues": [],
            "reachability_issues": [],
        }

        terminals = [
            uid for uid, u in self.units.items() if u.kind == UnitType.TERMINAL
        ]
        scripts = [uid for uid, u in self.units.items() if u.kind == UnitType.SCRIPT]

        # Check terminal relationships
        for term_id in terminals:
            # Get all link types this terminal participates in
            outgoing_types = {e.type for e in self.out_edges.get(term_id, [])}
            incoming_types = {e.type for e in self.in_edges.get(term_id, [])}

            # Terminals should only have SUB (outgoing) and SUR (incoming) links
            invalid_outgoing = outgoing_types - {LinkType.SUB}
            invalid_incoming = incoming_types - {LinkType.SUR}

            if invalid_outgoing:
                issues["terminal_relationships"].append(
                    f"Terminal '{term_id}' has invalid outgoing links: {invalid_outgoing}"
                )
            if invalid_incoming:
                issues["terminal_relationships"].append(
                    f"Terminal '{term_id}' has invalid incoming links: {invalid_incoming}"
                )

            # Terminals should have at least one parent script
            has_parent = any(
                e.type == LinkType.SUB for e in self.out_edges.get(term_id, [])
            )
            if not has_parent:
                issues["terminal_relationships"].append(
                    f"Terminal '{term_id}' has no parent script (missing SUB link)"
                )

        # Check script hierarchy
        for script_id in scripts:
            outgoing_types = {e.type for e in self.out_edges.get(script_id, [])}
            incoming_types = {e.type for e in self.in_edges.get(script_id, [])}

            # Scripts should not have SUB links (they receive evidence, don't send it)
            if LinkType.SUB in outgoing_types:
                issues["script_hierarchy"].append(
                    f"Script '{script_id}' has outgoing SUB links - scripts should receive SUB links"
                )

            # Scripts should not receive SUR links (they send requests, don't receive them)
            if LinkType.SUR in incoming_types:
                issues["script_hierarchy"].append(
                    f"Script '{script_id}' has incoming SUR links - scripts should send SUR links"
                )

        # Check for root scripts (scripts with no incoming SUB links)
        root_scripts = []
        for script_id in scripts:
            has_incoming_sub = any(
                e.type == LinkType.SUB for e in self.in_edges.get(script_id, [])
            )
            if not has_incoming_sub:
                root_scripts.append(script_id)

        if not root_scripts:
            issues["root_script_issues"].append(
                "No root scripts found (scripts with no incoming SUB links)"
            )
        elif len(root_scripts) > 1:
            issues["root_script_issues"].append(
                f"Multiple root scripts found: {root_scripts} - consider having a single root"
            )

        # Check reachability from root scripts
        if root_scripts:
            reachable_from_root = set()
            for root_id in root_scripts:
                self._collect_reachable_units(root_id, reachable_from_root)

            unreachable_units = set(self.units.keys()) - reachable_from_root
            if unreachable_units:
                issues["reachability_issues"].append(
                    f"Units not reachable from root scripts: {sorted(unreachable_units)}"
                )

        return {k: v for k, v in issues.items() if v}  # Remove empty categories

    def _collect_reachable_units(self, start_unit: str, reachable: set) -> None:
        """
        Collect all units reachable from a starting unit through any link type.

        Args:
            start_unit: Unit ID to start from
            reachable: Set to add reachable units to
        """
        if start_unit in reachable:
            return

        reachable.add(start_unit)

        # Follow all outgoing edges
        for edge in self.out_edges.get(start_unit, []):
            self._collect_reachable_units(edge.dst, reachable)

    def validate_activation_bounds(self, strict: bool = True) -> Dict[str, List[str]]:
        """
        Validate unit activation levels and thresholds are within proper bounds.

        Args:
            strict: If True, treat out-of-bounds values as errors. If False, treat as warnings.

        Returns:
            Dictionary of validation issues by category
        """
        issue_type = "errors" if strict else "warnings"
        issues = {
            f"activation_{issue_type}": [],
            f"threshold_{issue_type}": [],
            "consistency_warnings": [],
        }

        for unit_id, unit in self.units.items():
            # Activation bounds validation
            if not (0.0 <= unit.a <= 1.0):
                issues[f"activation_{issue_type}"].append(
                    f"Unit '{unit_id}' activation {unit.a} is outside valid range [0.0, 1.0]"
                )

            # Threshold bounds validation
            if not (0.0 <= unit.thresh <= 1.0):
                issues[f"threshold_{issue_type}"].append(
                    f"Unit '{unit_id}' threshold {unit.thresh} is outside valid range [0.0, 1.0]"
                )

            # Consistency checks
            if unit.thresh == 0.0 and unit.a < unit.thresh:
                issues["consistency_warnings"].append(
                    f"Unit '{unit_id}' has threshold 0.0 but activation {unit.a} < 0.0 (impossible)"
                )

            if unit.thresh == 1.0 and unit.a > unit.thresh:
                issues["consistency_warnings"].append(
                    f"Unit '{unit_id}' has threshold 1.0 but activation {unit.a} > 1.0 (impossible)"
                )

            # Check for unusual threshold/activation relationships
            if unit.a > 0.0 and unit.thresh > unit.a * 2:
                issues["consistency_warnings"].append(
                    f"Unit '{unit_id}' threshold {unit.thresh} is much higher than activation {unit.a}"
                )

        return {k: v for k, v in issues.items() if v}  # Remove empty categories

    def validate_all(
        self, strict_activation: bool = True
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Run comprehensive validation on the entire graph.

        This method combines all validation checks into a single comprehensive validation
        that covers cycles, link consistency, unit relationships, activation bounds,
        and graph integrity.

        Args:
            strict_activation: If True, treat out-of-bounds activation/threshold values as errors

        Returns:
            Dictionary organized by validation category, each containing issues
        """
        validation_results = {}

        # Cycle detection
        cycle_issues = self.validate_cycles()
        if cycle_issues:
            validation_results["cycles"] = cycle_issues

        # Link consistency
        link_issues = self.validate_link_consistency()
        if link_issues:
            validation_results["link_consistency"] = link_issues

        # Unit relationships
        relationship_issues = self.validate_unit_relationships()
        if relationship_issues:
            validation_results["unit_relationships"] = relationship_issues

        # Activation bounds
        activation_issues = self.validate_activation_bounds(strict=strict_activation)
        if activation_issues:
            validation_results["activation_bounds"] = activation_issues

        # Graph integrity
        integrity_issues = self.validate_graph_integrity()
        if integrity_issues:
            validation_results["graph_integrity"] = integrity_issues

        return validation_results

    def get_validation_summary(
        self, validation_results: Dict[str, Dict[str, List[str]]]
    ) -> Dict[str, int]:
        """
        Generate a summary of validation results.

        Args:
            validation_results: Results from validate_all() or other validation methods

        Returns:
            Dictionary with counts by category and severity
        """
        summary = {
            "total_issues": 0,
            "errors": 0,
            "warnings": 0,
            "categories_with_issues": 0,
        }

        for _, issues in validation_results.items():
            category_issue_count = 0
            for issue_type, issue_list in issues.items():
                issue_count = len(issue_list)
                category_issue_count += issue_count

                if "error" in issue_type:
                    summary["errors"] += issue_count
                elif "warning" in issue_type:
                    summary["warnings"] += issue_count

            if category_issue_count > 0:
                summary["categories_with_issues"] += 1
                summary["total_issues"] += category_issue_count

        return summary

    def is_valid(self, strict_activation: bool = True) -> bool:
        """
        Check if the graph passes all validations.

        Args:
            strict_activation: If True, out-of-bounds activation values are considered invalid

        Returns:
            True if graph passes all validations, False otherwise
        """
        results = self.validate_all(strict_activation=strict_activation)
        summary = self.get_validation_summary(results)

        # Graph is valid only if there are no errors
        return summary["errors"] == 0

    def analyze_performance_metrics(self) -> Dict[str, Any]:
        """
        Analyze graph performance and efficiency metrics.

        Returns:
            Dictionary with various performance metrics and efficiency indicators
        """
        metrics = {
            "structure_metrics": {},
            "complexity_metrics": {},
            "efficiency_indicators": {},
            "bottleneck_warnings": [],
        }

        total_units = len(self.units)
        total_edges = sum(len(edges) for edges in self.out_edges.values())

        # Structure metrics
        terminals = [
            uid for uid, u in self.units.items() if u.kind == UnitType.TERMINAL
        ]
        scripts = [uid for uid, u in self.units.items() if u.kind == UnitType.SCRIPT]

        metrics["structure_metrics"] = {
            "total_units": total_units,
            "total_edges": total_edges,
            "terminals": len(terminals),
            "scripts": len(scripts),
            "edges_per_unit": total_edges / total_units if total_units > 0 else 0,
            "terminal_ratio": len(terminals) / total_units if total_units > 0 else 0,
        }

        # Complexity metrics
        max_degree = 0
        avg_degree = 0
        degree_distribution = {}

        for unit_id in self.units:
            degree = len(self.out_edges.get(unit_id, [])) + len(
                self.in_edges.get(unit_id, [])
            )
            max_degree = max(max_degree, degree)
            avg_degree += degree

            if degree not in degree_distribution:
                degree_distribution[degree] = 0
            degree_distribution[degree] += 1

        avg_degree = avg_degree / total_units if total_units > 0 else 0

        metrics["complexity_metrics"] = {
            "max_degree": max_degree,
            "avg_degree": avg_degree,
            "degree_distribution": degree_distribution,
        }

        # Link type distribution
        link_distribution = {link_type: 0 for link_type in LinkType}
        for edges in self.out_edges.values():
            for edge in edges:
                link_distribution[edge.type] += 1

        metrics["complexity_metrics"]["link_type_distribution"] = {
            link_type.name: count for link_type, count in link_distribution.items()
        }

        # Efficiency indicators
        connected_components = self._find_connected_components()
        isolated_units = sum(
            1 for component in connected_components if len(component) == 1
        )

        # Calculate hierarchy depth (simplified)
        hierarchy_depth = self._calculate_hierarchy_depth()

        metrics["efficiency_indicators"] = {
            "connected_components": len(connected_components),
            "isolated_units": isolated_units,
            "hierarchy_depth": hierarchy_depth,
            "connectivity_ratio": (
                (total_units - isolated_units) / total_units if total_units > 0 else 0
            ),
        }

        # Bottleneck warnings
        if max_degree > 10:
            metrics["bottleneck_warnings"].append(
                f"High maximum degree ({max_degree}) may indicate performance bottlenecks"
            )

        if avg_degree > 5:
            metrics["bottleneck_warnings"].append(
                f"High average degree ({avg_degree:.1f}) may impact processing efficiency"
            )

        if len(connected_components) > total_units * 0.5:
            metrics["bottleneck_warnings"].append(
                f"High number of disconnected components ({len(connected_components)}) may indicate structural issues"
            )

        if hierarchy_depth > 5:
            metrics["bottleneck_warnings"].append(
                f"Deep hierarchy ({hierarchy_depth} levels) may cause propagation delays"
            )

        # Check for potential infinite loops (cycles in SUB/SUR)
        cycle_check = self.validate_cycles()
        if "SUB" in cycle_check or "SUR" in cycle_check:
            metrics["bottleneck_warnings"].append(
                "Cycles detected in SUB/SUR links - potential for infinite activation loops"
            )

        return metrics

    def _calculate_hierarchy_depth(self) -> int:
        """
        Calculate the maximum depth of the hierarchy from root scripts to leaf terminals.

        Returns:
            Maximum hierarchy depth
        """
        if not self.units:
            return 0

        # Find root scripts (no incoming SUB links)
        root_scripts = []
        for unit_id, unit in self.units.items():
            if unit.kind == UnitType.SCRIPT:
                has_incoming_sub = any(
                    e.type == LinkType.SUB for e in self.in_edges.get(unit_id, [])
                )
                if not has_incoming_sub:
                    root_scripts.append(unit_id)

        if not root_scripts:
            return 0

        max_depth = 0

        def calculate_depth(unit_id: str, visited: set) -> int:
            if unit_id in visited:
                return 0  # Avoid cycles

            visited.add(unit_id)
            max_child_depth = 0

            # Follow SUR links (script to children) and SUB links (evidence flow)
            for edge in self.out_edges.get(unit_id, []):
                if edge.type in [LinkType.SUR, LinkType.SUB]:
                    child_depth = calculate_depth(edge.dst, visited.copy())
                    max_child_depth = max(max_child_depth, child_depth)

            return max_child_depth + 1

        for root_id in root_scripts:
            depth = calculate_depth(root_id, set())
            max_depth = max(max_depth, depth)

        return max_depth

    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive graph statistics for monitoring and analysis.

        Returns:
            Dictionary with graph statistics and health indicators
        """
        validation_results = self.validate_all()
        summary = self.get_validation_summary(validation_results)
        performance = self.analyze_performance_metrics()

        stats = {
            "basic_stats": {
                "units": len(self.units),
                "edges": sum(len(edges) for edges in self.out_edges.values()),
                "unit_types": {
                    "terminals": len(
                        [u for u in self.units.values() if u.kind == UnitType.TERMINAL]
                    ),
                    "scripts": len(
                        [u for u in self.units.values() if u.kind == UnitType.SCRIPT]
                    ),
                },
            },
            "validation_summary": summary,
            "performance_metrics": performance,
            "health_score": self._calculate_health_score(summary, performance),
        }

        return stats

    def _calculate_health_score(
        self, validation_summary: Dict[str, int], performance_metrics: Dict[str, Any]
    ) -> float:
        """
        Calculate an overall health score for the graph (0.0 to 1.0).

        Args:
            validation_summary: Results from validation
            performance_metrics: Performance analysis results

        Returns:
            Health score between 0.0 (very unhealthy) and 1.0 (perfect)
        """
        score = 1.0

        # Penalize errors heavily
        if validation_summary["errors"] > 0:
            error_penalty = min(0.5, validation_summary["errors"] * 0.1)
            score -= error_penalty

        # Penalize warnings moderately
        if validation_summary["warnings"] > 0:
            warning_penalty = min(0.3, validation_summary["warnings"] * 0.05)
            score -= warning_penalty

        # Penalize performance issues
        complexity = performance_metrics.get("complexity_metrics", {})
        efficiency = performance_metrics.get("efficiency_indicators", {})

        # Penalize very high degree
        if complexity.get("max_degree", 0) > 15:
            score -= 0.2

        # Penalize disconnected components
        connectivity = efficiency.get("connectivity_ratio", 1.0)
        if connectivity < 0.8:
            score -= (1.0 - connectivity) * 0.3

        # Penalize bottleneck warnings
        bottleneck_penalty = (
            len(performance_metrics.get("bottleneck_warnings", [])) * 0.05
        )
        score -= min(0.2, bottleneck_penalty)

        return max(0.0, score)

    def add_custom_validation_rule(
        self, rule_name: str, validation_function: callable
    ) -> None:
        """
        Add a custom validation rule to the graph.

        Args:
            rule_name: Name identifier for the custom rule
            validation_function: Function that takes (self) and returns Dict[str, List[str]] of issues
        """
        self._custom_validation_rules[rule_name] = validation_function

    def validate_custom_rules(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Run all custom validation rules.

        Returns:
            Dictionary mapping rule names to their validation results
        """
        results = {}

        if hasattr(self, "_custom_validation_rules"):
            for rule_name, validation_func in self._custom_validation_rules.items():
                try:
                    rule_result = validation_func(self)
                    if rule_result:  # Only include rules that found issues
                        results[rule_name] = rule_result
                except (AttributeError, ValueError, TypeError, KeyError, RuntimeError) as e:
                    results[rule_name] = {
                        "execution_errors": [
                            f"Error running custom rule '{rule_name}': {str(e)}"
                        ]
                    }

        return results
