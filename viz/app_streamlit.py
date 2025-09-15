"""
Streamlit visualization interface for ReCoN networks.

This module provides an interactive web-based demonstration of Request Confirmation
Network (ReCoN) dynamics with enhanced visualization features. Users can generate
synthetic scenes, watch the network process them through multiple time steps, and
visualize both the scene and the network's internal state in real-time.

Minimal interface includes:
- Clean two-panel layout: Scene and Network
- Essential controls only: Generate, Step, Run/Pause, Reset
- Dark mode by default with high-contrast charts
- Compact status bar; advanced details moved to expanders
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

from perception.terminals import sample_scene_and_terminals
from recon_core.engine import Engine
from recon_core.enums import LinkType, State, UnitType
from recon_core.graph import Edge, Graph, Unit

st.set_page_config(layout="wide", page_title="ReCoN Demo", initial_sidebar_state="collapsed")

# Global visual styles (dark mode)
plt.style.use("dark_background")
st.markdown(
    """
    <style>
    :root {
        --bg: #0b1220;
        --panel: #0f172a;
        --muted: #94a3b8;
        --text: #e5e7eb;
        --accent: #60a5fa;
        --border: #1f2937;
    }
    html, body, .block-container { background-color: var(--bg) !important; color: var(--text) !important; }
    .block-container { padding-top: 0.6rem; padding-bottom: 1.25rem; }
    [data-testid="stSidebar"] { background-color: var(--panel) !important; color: var(--text) !important; }
    .stButton>button { background: #111827; color: var(--text); border: 1px solid var(--border); border-radius: 10px; }
    .stButton>button:hover { border-color: var(--accent); }
    .stMetric { background: rgba(255,255,255,0.04); border: 1px solid var(--border); padding: 0.35rem 0.5rem; border-radius: 8px; }
    .stDataFrame { background: var(--panel); }
    h1,h2,h3,h4 { color: var(--text); }
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
        self.is_running = False
        self.max_history = 100

    def init_graph(self):
        """Initialize the house recognition network topology."""
        g = Graph()
        # script units
        for init_unit_id in ["u_root", "u_roof", "u_body", "u_door"]:
            g.add_unit(Unit(init_unit_id, UnitType.SCRIPT, state=State.INACTIVE, a=0.0))
        # terminals
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

    def generate_scene(self):
        """Generate a new synthetic scene and initialize terminals."""
        img, tvals = sample_scene_and_terminals()  # img is the scene image
        # seed terminals
        for term_id, term_val in tvals.items():
            self.graph.units[term_id].a = float(term_val)
            self.graph.units[term_id].state = (
                State.REQUESTED if term_val > 0.1 else State.INACTIVE
            )
        # energize root to start requests
        self.graph.units["u_root"].a = 1.0
        self.graph.units["u_root"].state = State.ACTIVE
        self.engine.t = 0
        self.history = []
        self.message_history = []
        self.fovea_path = [(32, 32)]  # Start at center
        return img, tvals

    def step_simulation(self, n_steps=1):
        """Step the simulation forward and record history."""
        # Capture messages before stepping
        messages_this_step = []
        for graph_unit in self.graph.units.values():
            for receiver_id, message in graph_unit.outbox:
                messages_this_step.append((graph_unit.id, receiver_id, message))

        # Step the simulation
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


# Initialize simulation
if "sim" not in st.session_state:
    st.session_state.sim = ReCoNSimulation()

st.title("🖼️ ReCoN — Minimal Interactive Demo")

"""Top control bar with core actions"""
ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)
with ctrl_col1:
    if st.button("🎲 Generate Scene", use_container_width=True):
        img, terminal_vals = st.session_state.sim.generate_scene()
        st.session_state.img = img
        st.session_state.tvals = terminal_vals
        st.session_state.snap = st.session_state.sim.engine.snapshot()
        st.rerun()
with ctrl_col2:
    if st.button("⏭️ Step", use_container_width=True):
        st.session_state.snap = st.session_state.sim.step_simulation(1)
        st.rerun()
with ctrl_col3:
    run_label = "▶️ Run" if not st.session_state.sim.is_running else "⏸️ Pause"
    if st.button(run_label, type="primary", use_container_width=True):
        st.session_state.sim.is_running = not st.session_state.sim.is_running
        st.rerun()
with ctrl_col4:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.snap = st.session_state.sim.reset_simulation()
        st.rerun()

# Auto-run logic (minimal)
if st.session_state.sim.is_running:
    st.session_state.snap = st.session_state.sim.step_simulation(1)
    time.sleep(0.35)
    st.rerun()

# Current snapshot
current_snap = st.session_state.get("snap", st.session_state.sim.engine.snapshot())

# Persist selected unit in session for highlighting
unit_options = list(st.session_state.sim.graph.units.keys())
if "unit_select" not in st.session_state:
    st.session_state.unit_select = (
        "u_root" if "u_root" in unit_options else (unit_options[0] if unit_options else "")
    )
selected_unit = st.session_state.unit_select

# Main display
col_scene, col_graph = st.columns([1, 1.2])

with col_scene:
    st.subheader("🏠 Scene")

    # Get current scene
    current_img = st.session_state.get("img", np.zeros((64, 64), dtype=np.float32))

    # Create minimal visualization with fovea path and overlays
    fig, ax_scene = plt.subplots(1, 1, figsize=(6, 6))
    ax_scene.imshow(current_img, cmap="gray", extent=[0, 64, 64, 0])

    # Draw fovea path
    if len(st.session_state.sim.fovea_path) > 1:
        path_x = [p[0] for p in st.session_state.sim.fovea_path]
        path_y = [p[1] for p in st.session_state.sim.fovea_path]
        ax_scene.plot(path_x, path_y, "r-", alpha=0.7, linewidth=2, label="Fovea path")
        ax_scene.scatter(
            path_x[-1], path_y[-1], c="red", s=100, alpha=0.8, label="Current fovea"
        )

    # Add terminal detection overlays
    if "tvals" in st.session_state:
        # Simple visualization of where terminals are "detecting"
        # Green dots for active terminals, red for inactive
        terminal_positions = {
            "t_mean": (32, 32),  # center
            "t_vert": (45, 32),  # right side
            "t_horz": (20, 32),  # left side
        }

        for term_name, term_value in st.session_state.tvals.items():
            if term_name in terminal_positions:
                x, y = terminal_positions[term_name]
                color = "green" if term_value > 0.3 else "red"
                size = 50 + term_value * 100  # Size based on activation
                ax_scene.scatter(
                    x,
                    y,
                    c=color,
                    s=size,
                    alpha=0.7,
                    label=f"{term_name} ({term_value:.2f})",
                )

    ax_scene.set_xlim(0, 64)
    ax_scene.set_ylim(64, 0)  # Flip y-axis for image coordinates
    ax_scene.set_title("Scene with Attention & Detections", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Scene info
    if "tvals" in st.session_state:
        st.write("**Terminal Activations:**")
        cols = st.columns(3)
        for i, (term_name, term_value) in enumerate(st.session_state.tvals.items()):
            cols[i].metric(term_name, f"{term_value:.3f}")

with col_graph:
    st.subheader("🕸️ Network")

    # Create minimal graph visualization
    fig, ax_graph = plt.subplots(1, 1, figsize=(8, 8))

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
        "INACTIVE": 300,
        "REQUESTED": 500,
        "WAITING": 400,
        "ACTIVE": 600,
        "TRUE": 700,
        "CONFIRMED": 800,
        "FAILED": 400,
        "SUPPRESSED": 350,
    }

    # Node layout positions (fixed for consistency)
    pos = {
        "u_root": (0, 0),
        "u_roof": (-1, 1),
        "u_body": (0, 1),
        "u_door": (1, 1),
        "t_mean": (0, 2),
        "t_vert": (-1, 2),
        "t_horz": (1, 2),
    }

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
                arrowsize=12,
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

    ax_graph.set_title(f"Network State (t={current_snap['t']})")
    ax_graph.axis("off")
    st.pyplot(fig, use_container_width=True)

st.subheader("📊 Inspect")
with st.expander("Unit details", expanded=False):
    unit_options = list(st.session_state.sim.graph.units.keys())
    default_index = unit_options.index("u_root") if "u_root" in unit_options else 0
    selected_unit = st.selectbox(
        "Unit",
        unit_options,
        index=default_index,
        key="unit_select",
    )
    selected_unit_obj = st.session_state.sim.graph.units[selected_unit]
    unit_snap = current_snap["units"][selected_unit]

    # Unit overview
    col_name, col_state, col_activation = st.columns(3)
    with col_name:
        st.metric("Unit", selected_unit)
    with col_state:
        state_icon = {
            "INACTIVE": "⚪",
            "REQUESTED": "🔵",
            "WAITING": "🟡",
            "ACTIVE": "🟣",
            "TRUE": "🟢",
            "CONFIRMED": "🟢",
            "FAILED": "🔴",
            "SUPPRESSED": "⚫",
        }.get(unit_snap["state"], "⚪")
        st.metric("State", f"{state_icon} {unit_snap['state']}")
    with col_activation:
        st.metric("Activation", f"{unit_snap['a']:.3f}")

    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.write("**Message Queues:**")
        st.info(f"📥 Inbox: {unit_snap['inbox_size']}  |  📤 Outbox: {unit_snap['outbox_size']}")
    with detail_col2:
        st.write("**Connections:**")
        sub_children = st.session_state.sim.graph.sub_children(selected_unit)
        if sub_children:
            st.caption("Children (SUB): " + ", ".join(sub_children))
        sur_children = st.session_state.sim.graph.sur_children(selected_unit)
        if sur_children:
            st.caption("Requests to (SUR): " + ", ".join(sur_children))
        por_successors = st.session_state.sim.graph.por_successors(selected_unit)
        if por_successors:
            st.caption("Sequenced after (POR): " + ", ".join(por_successors))

with st.expander("All units (table)", expanded=False):
    unit_data = []
    for summary_unit_id, unit_data_dict in current_snap["units"].items():
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
    st.dataframe(unit_data, use_container_width=True)

st.divider()
col_status1, col_status2, col_status3 = st.columns(3)
with col_status1:
    active_units = sum(
        1 for u in current_snap["units"].values() if u["state"] not in ["INACTIVE", "SUPPRESSED"]
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
