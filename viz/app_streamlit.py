"""
Streamlit visualization interface for ReCoN networks.

This module provides an interactive web-based demonstration of Request Confirmation
Network (ReCoN) dynamics with enhanced visualization features. Users can generate
synthetic scenes, watch the network process them through multiple time steps, and
visualize both the scene and the network's internal state in real-time.

Enhanced interface includes:
- Scene generation with synthetic geometric shapes and fovea path visualization
- Interactive network graph visualization with state-based coloring and animations
- Message flow animations showing communication between units
- Step-by-step simulation control with run/pause/reset functionality
- Timeline scrubber for reviewing network evolution
- Detailed unit information panels
"""

import os
import sys
import time

# Add project root to Python path BEFORE any imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import streamlit as st
from contextlib import contextmanager

from perception.terminals import sample_scene_and_terminals
from perception.dataset import make_house_scene
from recon_core.engine import Engine
from recon_core.enums import LinkType, State, UnitType
from recon_core.graph import Edge, Graph, Unit

# Speed control mapping constants
SPEED_DELAY_MAPPING = {
    "Slow": 0.8,
    "Normal": 0.5,
    "Fast": 0.2,
}

st.set_page_config(layout="wide", page_title="ReCoN Demo")

# Global visual styles
plt.style.use("seaborn-v0_8-whitegrid")
st.markdown(
    """
    <style>
    .block-container { padding-top: 3rem; padding-bottom: 2rem; }
    [data-testid="stSidebar"] { background-color: rgba(15, 23, 42, 0.03); }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; }
    div[data-testid="stMetricLabel"] { color: #64748b; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Global state management
class ReCoNSimulation:
    """Manages the ReCoN simulation state and history."""

    def __init__(self):
        self.graph = self.init_graph()
        self.engine = Engine(self.graph)
        self.history = []  # Store snapshots for timeline
        self.message_history = []  # Store message flows for animation
        self.fovea_path = []  # Track fovea/scanning positions
        self.max_history = 100

    def init_graph(self):
        """Initialize the house recognition network topology."""
        g = Graph()
        # script units
        for init_unit_id in ["u_root", "u_roof", "u_body", "u_door"]:
            g.add_unit(Unit(init_unit_id, UnitType.SCRIPT, state=State.INACTIVE, a=0.0))
        # initial minimal terminals (will be rebuilt dynamically on Generate Scene)
        for term_id in ["t_mean", "t_vert", "t_horz"]:
            g.add_unit(
                Unit(
                    term_id, UnitType.TERMINAL, state=State.INACTIVE, a=0.0, thresh=0.5
                )
            )
        # hierarchy: terminals -> scripts via SUB; parent -> child via SUR
        g.add_edge(Edge("t_horz", "u_roof", LinkType.SUB, w=1.0))
        g.add_edge(Edge("t_mean", "u_body", LinkType.SUB, w=1.0))
        g.add_edge(Edge("t_vert", "u_door", LinkType.SUB, w=1.0))
        g.add_edge(Edge("t_mean", "u_door", LinkType.SUB, w=0.6))

        for c in ["u_roof", "u_body", "u_door"]:
            g.add_edge(Edge("u_root", c, LinkType.SUR, w=1.0))
            g.add_edge(Edge(c, "u_root", LinkType.SUB, w=1.0))

        # sequence (roof -> body -> door)
        g.add_edge(Edge("u_roof", "u_body", LinkType.POR, w=1.0))
        g.add_edge(Edge("u_body", "u_door", LinkType.POR, w=1.0))
        return g

    def _remove_existing_terminals(self):
        """Remove all existing TERMINAL units and associated edges."""
        terminal_ids = [
            uid for uid, u in self.graph.units.items() if u.kind == UnitType.TERMINAL
        ]
        if not terminal_ids:
            return
        # Remove edges touching terminals
        # Convert to set for O(1) lookups instead of O(n) list searches
        # This optimizes performance when there are many terminals and edges
        terminal_set = set(terminal_ids)

        # Remove outgoing edges from non-terminal nodes that connect to terminals
        for src_id, edges in list(self.graph.out_edges.items()):
            self.graph.out_edges[src_id] = [
                e
                for e in edges
                if e.src not in terminal_set and e.dst not in terminal_set
            ]

        # Remove incoming edges to non-terminal nodes that originate from terminals
        for dst_id, edges in list(self.graph.in_edges.items()):
            self.graph.in_edges[dst_id] = [
                e
                for e in edges
                if e.src not in terminal_set and e.dst not in terminal_set
            ]
        # Remove terminal units
        for tid in terminal_ids:
            self.graph.units.pop(tid, None)
            self.graph.out_edges.pop(tid, None)
            self.graph.in_edges.pop(tid, None)

    def _choose_parent_for_terminal(self, term_id: str) -> str:
        lid = term_id.lower()
        if term_id == "t_horz":
            return "u_roof"
        if term_id == "t_vert":
            return "u_door"
        if term_id == "t_mean":
            return "u_body"
        if any(k in lid for k in ["door"]):
            return "u_door"
        if any(
            k in lid
            for k in ["rect", "texture", "contrast", "n_shape", "aspect", "compact"]
        ):
            return "u_body"
        if any(
            k in lid
            for k in ["corner", "edge", "orient", "triangle", "vsym", "line_aniso"]
        ):
            return "u_roof"
        if lid.startswith("t_ae_") or lid.startswith("t_cnn_"):
            try:
                idx = int(lid.split("_")[-1])
            except ValueError:
                idx = 0
            return ["u_roof", "u_body", "u_door"][idx % 3]
        return "u_body"

    def _rebuild_terminals(self, terminals: dict):
        self._remove_existing_terminals()
        for term_id, _ in terminals.items():
            self.graph.add_unit(
                Unit(
                    term_id, UnitType.TERMINAL, state=State.INACTIVE, a=0.0, thresh=0.5
                )
            )
            parent = self._choose_parent_for_terminal(term_id)
            self.graph.add_edge(Edge(term_id, parent, LinkType.SUB, w=1.0))
            if term_id == "t_mean":
                self.graph.add_edge(Edge(term_id, "u_door", LinkType.SUB, w=0.6))
        for term_id, term_val in terminals.items():
            u = self.graph.units[term_id]
            u.a = float(term_val)
            u.state = State.REQUESTED if term_val > 0.1 else State.INACTIVE

    def generate_scene(self):
        """Generate a new scene and initialize terminals using Basic feature source."""
        img = make_house_scene()
        _, tvals = sample_scene_and_terminals()

        self._rebuild_terminals(tvals)
        self.graph.units["u_root"].a = 1.0
        self.graph.units["u_root"].state = State.ACTIVE
        self.engine.t = 0
        self.history = []
        self.message_history = []
        self.fovea_path = [(32, 32)]
        return img, tvals

    def step_simulation(self, n_steps=1):
        """Step the simulation forward and record history."""
        messages_this_step = []

        @contextmanager
        def patch_send_message(engine, on_send):
            original = engine.send_message
            def wrapper(sender_id, receiver_id, message):
                on_send(sender_id, receiver_id, message)
                return original(sender_id, receiver_id, message)
            engine.send_message = wrapper  # type: ignore
            try:
                yield
            finally:
                engine.send_message = original  # type: ignore

        def on_send(sender_id, receiver_id, message):
            messages_this_step.append((sender_id, receiver_id, message))

        with patch_send_message(self.engine, on_send):
            snap = self.engine.step(n_steps)

        # Update fovea path (simulate scanning behavior)
        if self.graph.units["u_root"].state == State.ACTIVE:
            # Simple scanning pattern: move fovea based on active units
            center_x, center_y = 32, 32
            if self.graph.units["u_roof"].state in [State.REQUESTED, State.ACTIVE]:
                center_y -= 10  # Look up for roof
            elif self.graph.units["u_body"].state in [State.REQUESTED, State.ACTIVE]:
                center_y += 5  # Look down for body
            elif self.graph.units["u_door"].state in [State.REQUESTED, State.ACTIVE]:
                center_x += 5  # Look right for door
            self.fovea_path.append((center_x, center_y))

        # Record history
        self.history.append(snap)
        self.message_history.append(messages_this_step)

        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.message_history.pop(0)

        return snap

    def reset_simulation(self):
        """Reset the simulation to initial state."""
        self.engine.reset()
        self.history = []
        self.message_history = []
        self.fovea_path = [(32, 32)]
        return self.engine.snapshot()


def get_speed_label_from_delay(delay):
    """Convert run delay value to human-readable speed label."""
    # Find the speed label that corresponds to the given delay
    for speed, delay_val in SPEED_DELAY_MAPPING.items():
        if delay == delay_val:
            return speed
    # Fallback for unknown delay values
    if delay > 0.5:
        return "Slow"
    elif delay < 0.5:
        return "Fast"
    else:
        return "Normal"


# Initialize simulation
if "sim" not in st.session_state:
    st.session_state.sim = ReCoNSimulation()

st.title("🖼️ Request Confirmation Network — Interactive Demo")

# Sidebar: refined control panel
with st.sidebar:
    st.header("🎛️ Controls")

    # Scene controls
    st.caption("Scene")

    col_scene_gen, col_scene_reset = st.columns(2)
    with col_scene_gen:
        if st.button("🎲 Generate Scene", type="primary", use_container_width=True):
            img, terminal_vals = st.session_state.sim.generate_scene()
            st.session_state.img = img
            st.session_state.tvals = terminal_vals
            st.session_state.snap = st.session_state.sim.engine.snapshot()
    with col_scene_reset:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.snap = st.session_state.sim.reset_simulation()

    st.divider()

    # Playback controls
    st.caption("Playback")
    if st.button("⏭️ Step", use_container_width=True):
        st.session_state.snap = st.session_state.sim.step_simulation(1)

    st.divider()

    # Timeline scrubber
    if len(st.session_state.sim.history) > 1:
        timeline_idx = st.slider(
            "⏱️ Timeline",
            0,
            len(st.session_state.sim.history) - 1,
            len(st.session_state.sim.history) - 1,
        )
        current_snap = st.session_state.sim.history[timeline_idx]
    elif st.session_state.sim.history:
        timeline_idx = 0
        current_snap = st.session_state.sim.history[0]
    else:
        timeline_idx = 0
        current_snap = st.session_state.get(
            "snap", st.session_state.sim.engine.snapshot()
        )

    st.divider()

    # Unit selector for hover functionality (scoped to sidebar)
    st.header("🔍 Unit Inspection")
    unit_options = list(st.session_state.sim.graph.units.keys())
    selected_unit = st.selectbox(
        "Select unit for details:",
        unit_options,
        index=unit_options.index("u_root") if "u_root" in unit_options else 0,
    )

# Main display
col_scene, col_graph = st.columns([1, 1.2])

with col_scene:
    st.subheader("🏠 Scene with Fovea Path")

    # Get current scene
    current_img = st.session_state.get("img", np.zeros((64, 64), dtype=np.float32))

    # Create visualization with fovea path and overlays
    fig, (ax_scene, ax_bars) = plt.subplots(
        1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [2, 1]}
    )

    # Scene with overlays
    ax_scene.imshow(current_img, cmap="gray", extent=[0, 64, 64, 0])

    # Draw fovea path
    if len(st.session_state.sim.fovea_path) > 1:
        path_x = [p[0] for p in st.session_state.sim.fovea_path]
        path_y = [p[1] for p in st.session_state.sim.fovea_path]
        ax_scene.plot(path_x, path_y, "r-", alpha=0.7, linewidth=2, label="Fovea path")
        ax_scene.scatter(
            path_x[-1], path_y[-1], c="red", s=100, alpha=0.8, label="Current fovea"
        )

    # Add terminal detection overlays using live activations
    terminal_positions = {
        "t_mean": (32, 32),
        "t_vert": (45, 32),
        "t_horz": (20, 32),
    }
    for term_name, (x, y) in terminal_positions.items():
        if term_name in st.session_state.sim.graph.units:
            term_unit = st.session_state.sim.graph.units[term_name]
            term_value = float(getattr(term_unit, "a", 0.0))
            color = "green" if term_value > 0.3 else "red"
            size = 50 + term_value * 100
            ax_scene.scatter(x, y, c=color, s=size, alpha=0.7, label=f"{term_name} ({term_value:.2f})")

    ax_scene.set_xlim(0, 64)
    ax_scene.set_ylim(64, 0)  # Flip y-axis for image coordinates
    ax_scene.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False, fontsize=8
    )
    ax_scene.set_title("Scene with Attention Path & Terminal Detections")

    # Confidence bars for script units
    script_units = ["u_root", "u_roof", "u_body", "u_door"]
    activations = []
    labels = []

    for script_id in script_units:
        if script_id in st.session_state.sim.graph.units:
            script_unit = st.session_state.sim.graph.units[script_id]
            activations.append(script_unit.a)
            labels.append(script_id.replace("u_", ""))

    if activations:
        bars = ax_bars.barh(
            labels, activations, color=["#2ca25f", "#6baed6", "#fdae6b", "#de2d26"]
        )
        ax_bars.set_xlim(0, 1)
        ax_bars.set_xlabel("Confidence")
        ax_bars.set_title("Part Confidence Levels")

        # Add value labels on bars
        for bar, activation in zip(bars, activations):
            ax_bars.text(
                activation + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{activation:.2f}",
                ha="left",
                va="center",
                fontsize=9,
            )

    ax_bars.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Scene info
    # Show live terminal activations from current graph
    terminals_live = [
        (uid, u.a)
        for uid, u in st.session_state.sim.graph.units.items()
        if u.kind == UnitType.TERMINAL
    ]
    if terminals_live:
        st.write("**Terminal Activations (Live):**")
        terminals_live.sort(key=lambda x: x[0])
        num_cols = min(3, len(terminals_live))
        cols = st.columns(num_cols)
        for i, (term_name, term_value) in enumerate(terminals_live):
            cols[i % num_cols].metric(term_name, f"{term_value:.3f}")

with col_graph:
    st.subheader("🕸️ Network Graph & Messages")

    # Create enhanced graph visualization
    fig, (ax_graph, ax_msgs) = plt.subplots(2, 1, figsize=(8, 10), height_ratios=[2, 1])

    # Build NetworkX graph for visualization
    G = nx.DiGraph()
    state_colors = {
        "INACTIVE": "#cccccc",
        "REQUESTED": "#6baed6",
        "WAITING": "#fdae6b",
        "ACTIVE": "#9ecae1",
        "TRUE": "#31a354",
        "CONFIRMED": "#2ca25f",
        "FAILED": "#de2d26",
        "SUPPRESSED": "#756bb1",
    }
    state_sizes = {
        "INACTIVE": 1800,
        "REQUESTED": 2400,
        "WAITING": 2100,
        "ACTIVE": 2700,
        "TRUE": 3000,
        "CONFIRMED": 3300,
        "FAILED": 2100,
        "SUPPRESSED": 1900,
    }

    # Node layout positions (hierarchical flow: root -> scripts -> terminals)
    pos = {
        "u_root": (0, 3),
        "u_roof": (-1, 2),
        "u_body": (0, 2),
        "u_door": (1, 2),
    }
    # Place terminal nodes grouped by their parent script units
    terminal_positions = {
        "t_horz": (-1.0, 1.0),  # near u_roof
        "t_vert": (1.0, 1.0),   # near u_door
        "t_mean": (0.0, 1.0),   # near u_body
    }

    # Position known terminals first
    graph = st.session_state.sim.graph
    units = graph.units
    out_edges = graph.out_edges
    for term_name, position in terminal_positions.items():
        if term_name in units:
            pos[term_name] = position

    # Position remaining terminals under their actual parent scripts via SUB links
    all_terminals = [uid for uid, u in units.items() if u.kind == UnitType.TERMINAL]
    remaining_terminals = [t for t in all_terminals if t not in terminal_positions]
    parent_to_terms = {}
    for term_id in remaining_terminals:
        parent_id = next((e.dst for e in out_edges.get(term_id, []) if e.type == LinkType.SUB), None)
        parent_to_terms.setdefault(parent_id, []).append(term_id)

    for parent_id, terms in parent_to_terms.items():
        if not terms:
            continue
        if parent_id in pos:
            parent_x, _ = pos[parent_id]
            xs = np.linspace(-0.8, 0.8, num=len(terms)) if len(terms) > 1 else [0.0]
            for dx, n in zip(xs, terms):
                pos[n] = (float(parent_x + dx), 0.9)
        else:
            # Place unparented terms along bottom
            xs = np.linspace(-1.5, 1.5, num=len(terms))
            for x, n in zip(xs, terms):
                pos[n] = (float(x), 0.8)

    # Add nodes with enhanced styling
    for node_id, graph_unit in st.session_state.sim.graph.units.items():
        state_name = graph_unit.state.name
        G.add_node(
            node_id,
            color=state_colors[state_name],
            size=state_sizes[state_name],
            activation=round(graph_unit.a, 2),
            state=state_name,
        )

    # Add edges with styling
    edge_styles = {
        "SUB": {"color": "#2ca25f", "style": "solid", "alpha": 0.7},
        "SUR": {"color": "#de2d26", "style": "solid", "alpha": 0.7},
        "POR": {"color": "#756bb1", "style": "dashed", "alpha": 0.5},
        "RET": {"color": "#fdae6b", "style": "dotted", "alpha": 0.5},
    }

    for src_id, edges in st.session_state.sim.graph.out_edges.items():
        for e in edges:
            style = edge_styles[e.type.name]
            G.add_edge(
                e.src,
                e.dst,
                color=style["color"],
                style=style["style"],
                alpha=style["alpha"],
                weight=e.w,
            )

    # Draw the graph
    node_colors = [G.nodes[n]["color"] for n in G.nodes()]
    node_sizes = [G.nodes[n]["size"] for n in G.nodes()]

    # Draw edges by type
    for link_type, style in edge_styles.items():
        edges_of_type = [
            (u, v) for u, v, d in G.edges(data=True) if d.get("style") == style["style"]
        ]
        if edges_of_type:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges_of_type,
                edge_color=style["color"],
                style=style["style"],
                alpha=style["alpha"],
                arrows=True,
                arrowsize=15,
                ax=ax_graph,
            )

    # Draw nodes and labels with selection highlighting
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes, ax=ax_graph
    )

    node_sizes = [G.nodes[n]["size"] for n in G.nodes()]
    # Highlight selected node
    if selected_unit in pos:
        sel_x, sel_y = pos[selected_unit]
        ax_graph.scatter(
            sel_x,
            sel_y,
            s=node_sizes[list(G.nodes()).index(selected_unit)] * 1.5,
            c="yellow",
            alpha=0.8,
            edgecolors="black",
            linewidth=3,
        )

    nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold", ax=ax_graph)

    # Add edge labels
    edge_labels = {}
    for u, v in G.edges():
        w = G.edges[(u, v)].get("weight")
        edge_labels[(u, v)] = f"{w:.1f}" if isinstance(w, (int, float)) else ""
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax_graph)

    ax_graph.set_title(
        f"Network State (t={current_snap['t']}) - Yellow outline shows selected unit"
    )
    ax_graph.axis("off")

    # Animated message arrows on the graph
    # Show messages that led into this state (previous step -> current)
    if st.session_state.sim.message_history:
        msg_index = max(0, min(len(st.session_state.sim.message_history) - 1, max(0, timeline_idx - 1)))
        messages = st.session_state.sim.message_history[msg_index]

        # Define arrow styles for different message types
        arrow_styles = {
            "REQUEST": {
                "color": "#6baed6",
                "width": 1,
                "style": "-",
                "alpha": 0.8,
                "label": "→",
            },
            "CONFIRM": {
                "color": "#31a354",
                "width": 3,
                "style": "-",
                "alpha": 0.9,
                "label": "⇒",
            },
            "WAIT": {
                "color": "#fdae6b",
                "width": 2,
                "style": "--",
                "alpha": 0.7,
                "label": "⇄",
            },
            "INHIBIT_REQUEST": {
                "color": "#de2d26",
                "width": 2,
                "style": ":",
                "alpha": 0.8,
                "label": "↔",
            },
            "INHIBIT_CONFIRM": {
                "color": "#756bb1",
                "width": 2,
                "style": "-.",
                "alpha": 0.8,
                "label": "⇔",
            },
        }

        # Draw arrows for each message
        for sender, receiver, msg in messages[
            -8:
        ]:  # Show last 8 messages to avoid clutter
            if sender in pos and receiver in pos:
                style = arrow_styles.get(
                    msg.name,
                    {
                        "color": "#666666",
                        "width": 1,
                        "style": "-",
                        "alpha": 0.5,
                        "label": "?",
                    },
                )

                # Calculate arrow positions with slight offset to avoid overlapping nodes
                sender_pos = pos[sender]
                receiver_pos = pos[receiver]

                # Add small offset for multiple messages between same units
                offset = (
                    messages.index((sender, receiver, msg)) * 0.05
                )  # Small offset for each message
                dx = receiver_pos[0] - sender_pos[0]
                dy = receiver_pos[1] - sender_pos[1]
                length = (dx**2 + dy**2) ** 0.5

                if length > 0:
                    # Perpendicular offset for parallel arrows
                    perp_x = -dy / length * offset
                    perp_y = dx / length * offset

                    start_x = sender_pos[0] + perp_x
                    start_y = sender_pos[1] + perp_y
                    end_x = receiver_pos[0] + perp_x
                    end_y = receiver_pos[1] + perp_y

                    # Draw the message arrow
                    ax_graph.annotate(
                        "",
                        xy=(end_x, end_y),
                        xytext=(start_x, start_y),
                        arrowprops=dict(
                            arrowstyle="->",
                            color=style["color"],
                            linewidth=style["width"],
                            linestyle=style["style"],
                            alpha=style["alpha"],
                            shrinkA=25,  # Shrink from start point
                            shrinkB=25,  # Shrink from end point
                        ),
                        zorder=10,
                    )  # Draw on top

                    # Add message type label near the arrow
                    mid_x = (start_x + end_x) / 2 + perp_x * 2
                    mid_y = (start_y + end_y) / 2 + perp_y * 2
                    ax_graph.text(
                        mid_x,
                        mid_y,
                        style["label"],
                        fontsize=8,
                        ha="center",
                        va="center",
                        bbox=dict(
                            boxstyle="round,pad=0.1",
                            facecolor=style["color"],
                            alpha=0.7,
                        ),
                        color="white",
                        fontweight="bold",
                        zorder=11,
                    )

    # Message summary panel
    try:
        t_now = int(current_snap["t"]) if isinstance(current_snap.get("t", 0), (int, float)) else 0
    except Exception:
        t_now = 0
    step_title = f"Message Summary (Step t={max(0, t_now-1)} → t={t_now})"
    ax_msgs.set_title(step_title)
    ax_msgs.set_xlim(0, 10)
    ax_msgs.set_ylim(0, 10)
    ax_msgs.axis("off")

    # Show message counts and recent activity for the same previous step
    if st.session_state.sim.message_history:
        msg_index = max(0, min(len(st.session_state.sim.message_history) - 1, max(0, timeline_idx - 1)))
        messages = st.session_state.sim.message_history[msg_index]

        # Message type counts
        msg_counts = {}
        for _, _, msg in messages:
            msg_counts[msg.name] = msg_counts.get(msg.name, 0) + 1

        y_pos = 9
        ax_msgs.text(
            0.5,
            y_pos,
            f"📨 Total Messages: {len(messages)}",
            fontsize=10,
            fontweight="bold",
        )
        y_pos -= 1

        # Show counts for each message type
        msg_colors = {
            "REQUEST": "#6baed6",
            "CONFIRM": "#31a354",
            "WAIT": "#fdae6b",
            "INHIBIT_REQUEST": "#de2d26",
            "INHIBIT_CONFIRM": "#756bb1",
        }

        for msg_type, count in msg_counts.items():
            if y_pos > 2:  # Leave space for recent messages
                color = msg_colors.get(msg_type, "#666666")
                ax_msgs.text(
                    0.5,
                    y_pos,
                    f"{msg_type}: {count}",
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                )
                y_pos -= 0.8

        # Show recent individual messages (last 3)
        if messages:
            y_pos = 2
            ax_msgs.text(0.5, y_pos, "Recent Messages:", fontsize=8, style="italic")
            y_pos -= 0.7

            for sender, receiver, msg in messages[-3:]:
                if y_pos > 0:
                    color = msg_colors.get(msg.name, "#666666")
                    ax_msgs.text(
                        0.5,
                        y_pos,
                        f"{sender}→{receiver}: {msg.name}",
                        fontsize=7,
                        color=color,
                        alpha=0.8,
                    )
                    y_pos -= 0.6

    st.pyplot(fig, use_container_width=True)

# Unit Details and Hover Information
st.subheader("📊 Unit Details & Hover Information")

st.info(
    "💡 **Interactive Inspection**: Use the sidebar selector above to inspect any unit in detail (equivalent to 'hover' functionality in the graph)"
)

# Show selected unit details
if selected_unit:
    selected_unit_obj = st.session_state.sim.graph.units[selected_unit]
    unit_snap = current_snap["units"][selected_unit]

    # Unit overview
    col_name, col_state, col_activation = st.columns(3)
    with col_name:
        st.metric("Unit", selected_unit)
    with col_state:
        state_color = {
            "INACTIVE": "🟢",
            "REQUESTED": "🔵",
            "WAITING": "🟡",
            "ACTIVE": "🟣",
            "TRUE": "🟢",
            "CONFIRMED": "🟦",
            "FAILED": "🔴",
            "SUPPRESSED": "⚫",
        }.get(unit_snap["state"], "⚪")
        st.metric("State", f"{state_color} {unit_snap['state']}")
    with col_activation:
        st.metric("Activation", f"{unit_snap['a']:.3f}")

    # Detailed information
    st.write(f"**📋 Details for {selected_unit}:**")

    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:
        st.write("**Message Queues:**")
        st.info(f"📥 Inbox: {unit_snap['inbox_size']} messages")
        st.info(f"📤 Outbox: {unit_snap['outbox_size']} messages")

        if selected_unit_obj.inbox:
            st.write("**Recent Inbox Messages:**")
            for i, (sender, msg) in enumerate(
                selected_unit_obj.inbox[-3:]
            ):  # Show last 3
                st.caption(f"• {sender} → {msg.name}")

    with detail_col2:
        st.write("**🔗 Connections:**")

        # Child relationships
        sub_children = st.session_state.sim.graph.sub_children(selected_unit)
        if sub_children:
            st.write("**Children (SUB links):**")
            for child in sub_children:
                child_unit = st.session_state.sim.graph.units[child]
                status = (
                    "🟢" if child_unit.state in [State.TRUE, State.CONFIRMED] else "⚪"
                )
                st.caption(
                    f"• {child}: {status} {child_unit.state.name} (a={child_unit.a:.2f})"
                )

        # Parent relationships
        sur_children = st.session_state.sim.graph.sur_children(selected_unit)
        if sur_children:
            st.write("**Requests to (SUR links):**")
            for child in sur_children:
                child_unit = st.session_state.sim.graph.units[child]
                status = "🔵" if child_unit.state == State.REQUESTED else "⚪"
                st.caption(f"• {child}: {status} {child_unit.state.name}")

        # Sequence relationships
        por_successors = st.session_state.sim.graph.por_successors(selected_unit)
        if por_successors:
            st.write("**Sequenced after (POR links):**")
            for succ in por_successors:
                succ_unit = st.session_state.sim.graph.units[succ]
                status = "🟡" if succ_unit.state == State.CONFIRMED else "⚪"
                st.caption(f"• {succ}: {status} {succ_unit.state.name}")

# Summary table for all units
st.write("**📈 All Units Overview:**")
unit_data = []
for summary_unit_id, unit_data_dict in current_snap["units"].items():
    summary_unit = st.session_state.sim.graph.units[summary_unit_id]
    unit_data.append(
        {
            "Unit": summary_unit_id,
            "Type": unit_data_dict["kind"],
            "State": unit_data_dict["state"],
            "Activation": round(unit_data_dict["a"], 3),
            "Inbox": unit_data_dict["inbox_size"],
            "Outbox": unit_data_dict["outbox_size"],
        }
    )

# Stable order for readability
unit_data.sort(key=lambda r: r["Unit"])
st.dataframe(unit_data, use_container_width=True)

# Status summary
col_status1, col_status2, col_status3 = st.columns(3)
with col_status1:
    active_units = sum(
        1
        for u in current_snap["units"].values()
        if u["state"] not in ["INACTIVE", "SUPPRESSED"]
    )
    st.metric("Active Units", active_units)

with col_status2:
    confirmed_units = sum(
        1 for u in current_snap["units"].values() if u["state"] == "CONFIRMED"
    )
    st.metric("Confirmed Units", confirmed_units)

with col_status3:
    total_messages = sum(
        u["inbox_size"] + u["outbox_size"] for u in current_snap["units"].values()
    )
    st.metric("Pending Messages", total_messages)
