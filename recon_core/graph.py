"""
Graph data structures for the Request Confirmation Network (ReCoN).

This module defines the core data structures used to represent the network topology:
- Unit: Individual computational elements with state and activation
- Edge: Directed connections between units with typed relationships
- Graph: Container for units and edges with utility methods
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

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
