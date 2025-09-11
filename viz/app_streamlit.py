"""
Streamlit visualization interface for ReCoN networks.

This module provides an interactive web-based demonstration of Request Confirmation
Network (ReCoN) dynamics. Users can generate synthetic scenes, watch the network
process them through multiple time steps, and visualize both the scene and the
network's internal state in real-time.

The interface includes:
- Scene generation with synthetic geometric shapes
- Network graph visualization with state-based coloring
- Step-by-step simulation control
- Real-time display of unit activations and states
"""

import streamlit as st
import networkx as nx
import numpy as np
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from recon_core.enums import UnitType, LinkType, State
from recon_core.graph import Graph, Unit, Edge
from recon_core.engine import Engine
from perception.terminals import sample_scene_and_terminals

st.set_page_config(layout="wide", page_title="ReCoN Demo")


@st.cache_resource
def init_graph():
    """
    Initialize the house recognition network topology.

    Creates a ReCoN network designed to recognize house-like structures with:
    - Root script unit coordinating the recognition
    - Body, roof, and door script units for parts recognition
    - Terminal units for basic visual features (mean intensity, vertical/horizontal edges)

    The network uses hierarchical relationships:
    - SUB links: Evidence flows from terminals/scripts to their parents
    - SUR links: Requests flow from parents to children (top-down)
    - POR links: Temporal sequencing (roof → body → door)

    Returns:
        Graph: Initialized ReCoN network ready for simulation
    """
    g = Graph()
    # script units
    for uid in ['u_root','u_roof','u_body','u_door']:
        g.add_unit(Unit(uid, UnitType.SCRIPT, state=State.INACTIVE, a=0.0))
    # terminals
    for tid in ['t_mean','t_vert','t_horz']:
        g.add_unit(Unit(tid, UnitType.TERMINAL, state=State.INACTIVE, a=0.0, thresh=0.5))
    # hierarchy: terminals -> scripts via SUB; parent -> child via SUR
    # roof depends on t_horz; body on t_mean; door on (t_vert OR t_mean)
    g.add_edge(Edge('t_horz','u_roof', LinkType.SUB, w=1.0))
    g.add_edge(Edge('t_mean','u_body', LinkType.SUB, w=1.0))
    g.add_edge(Edge('t_vert','u_door', LinkType.SUB, w=1.0))
    g.add_edge(Edge('t_mean','u_door', LinkType.SUB, w=0.6))

    for c in ['u_roof','u_body','u_door']:
        g.add_edge(Edge('u_root', c, LinkType.SUR, w=1.0))
        g.add_edge(Edge(c, 'u_root', LinkType.SUB, w=1.0))

    # sequence (roof -> body -> door)
    g.add_edge(Edge('u_roof','u_body', LinkType.POR, w=1.0))
    g.add_edge(Edge('u_body','u_door', LinkType.POR, w=1.0))
    return g

g = init_graph()
engine = Engine(g)

st.title("Request Confirmation Network — Demo")
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Scene")
    if st.button("Generate Scene"):
        img, tvals = sample_scene_and_terminals()
        st.session_state['img'] = img
        st.session_state['tvals'] = tvals
        # seed terminals
        for tid, val in tvals.items():
            g.units[tid].a = float(val)
            g.units[tid].state = State.REQUESTED if val > 0.1 else State.INACTIVE
        # energize root to start requests
        g.units['u_root'].a = 1.0
        g.units['u_root'].state = State.ACTIVE
        engine.t = 0

    img = st.session_state.get('img', np.zeros((64,64), dtype=np.float32))
    st.image(img, caption="Synthetic Scene", width=320, clamp=True)

    steps = st.number_input("Steps", 1, 200, 1)
    if st.button("Run"):
        snap = engine.step(int(steps))
        st.session_state['snap'] = snap
    if st.button("Reset"):
        engine.reset()
        st.session_state['snap'] = engine.snapshot()

with col2:
    st.subheader("Graph")
    snap = st.session_state.get('snap', engine.snapshot())
    # Build a small NX graph for drawing
    G = nx.DiGraph()
    color_map = {
        'INACTIVE': '#aaaaaa', 'REQUESTED': '#6baed6', 'WAITING': '#fdae6b', 'ACTIVE': '#9ecae1',
        'TRUE': '#31a354', 'CONFIRMED': '#2ca25f', 'FAILED': '#de2d26', 'SUPPRESSED': '#756bb1'
    }
    for uid,u in g.units.items():
        G.add_node(uid, color=color_map[u.state.name], a=round(u.a,2))
    for src, edges in g.out_edges.items():
        for e in edges:
            style = 'solid' if e.type in (LinkType.SUB, LinkType.SUR) else 'dashed'
            G.add_edge(e.src, e.dst, type=e.type.name, w=e.w, style=style)

    pos = nx.spring_layout(G, seed=42)
    # Simple drawing with pyplot for Streamlit
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    node_colors = [G.nodes[n]['color'] for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle='->')
    edge_labels = {(u,v): G.edges[(u,v)]['type'] for (u,v) in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax)
    ax.axis('off')
    st.pyplot(fig)

    st.caption(f"t={snap['t']}  — Hover not supported in Streamlit's static matplotlib. Use labels/colors for state; activation shown below.")
    st.json(snap['units'])
