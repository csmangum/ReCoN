"""
Manim Community animation: Step-by-step ReCoN activation on the synthetic house scene.

This script renders a clean, polished walkthrough of the ReCoN network during
active perception over the synthetic house image. It uses your existing ReCoN
engine implementation for truthful state sequencing and overlays a minimal
graph viz showing SUR/SUB/POR transitions.

Usage examples:
  manim -pqh scripts/manim_recon_house.py HouseWalkthrough

Requires manim-community installed. See README at bottom of file for basics.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREEN,
    RED,
    UP,
    WHITE,
    YELLOW,
    FadeIn,
    FadeOut,
    Group,
    ImageMobject,
    Text,
)
from manim import rate_functions as rf

# Ensure project root import for recon_core and perception
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure scripts dir import for local manim utilities
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


from manim_common import BaseReconScene, NodeViz, edge_arrow, gradient_edge_arrow

from perception.dataset import make_castle_scene, make_house_scene
from perception.terminals import terminals_from_image
from recon_core.engine import Engine, EngineConfig
from recon_core.enums import LinkType, State, UnitType
from recon_core.graph import Edge, Graph, Unit

# ---------- Helpers to construct the demo graph and run steps ----------


def build_house_graph() -> Graph:
    g = Graph()
    # Scripts
    for uid in ["u_root", "u_roof", "u_body", "u_door"]:
        g.add_unit(Unit(uid, UnitType.SCRIPT))
    # Terminals
    for tid in ["t_mean", "t_vert", "t_horz"]:
        g.add_unit(Unit(tid, UnitType.TERMINAL, thresh=0.5))

    # Evidence links (SUB)
    g.add_edge(Edge("t_horz", "u_roof", LinkType.SUB, w=1.0))
    g.add_edge(Edge("t_mean", "u_body", LinkType.SUB, w=1.0))
    g.add_edge(Edge("t_vert", "u_door", LinkType.SUB, w=1.0))
    g.add_edge(Edge("t_mean", "u_door", LinkType.SUB, w=0.6))

    # Hierarchy SUR and child SUB back to root
    for child in ["u_roof", "u_body", "u_door"]:
        g.add_edge(Edge("u_root", child, LinkType.SUR, w=1.0))
        g.add_edge(Edge(child, "u_root", LinkType.SUB, w=1.0))

    # Sequence POR: roof -> body -> door
    g.add_edge(Edge("u_roof", "u_body", LinkType.POR, w=1.0))
    g.add_edge(Edge("u_body", "u_door", LinkType.POR, w=1.0))
    return g


def build_castle_graph() -> Graph:
    """Build a castle graph based on castle.yaml configuration."""
    g = Graph()
    
    # Root castle script
    g.add_unit(Unit("u_root", UnitType.SCRIPT))
    
    # Castle component scripts
    for uid in ["u_wall", "u_crenellations", "u_towers", "u_gate"]:
        g.add_unit(Unit(uid, UnitType.SCRIPT))
    
    # Terminals (same as house for compatibility)
    for tid in ["t_mean", "t_vert", "t_horz"]:
        g.add_unit(Unit(tid, UnitType.TERMINAL, thresh=0.5))

    # Evidence links (SUB) - based on castle.yaml parts
    g.add_edge(Edge("t_mean", "u_wall", LinkType.SUB, w=1.0))
    g.add_edge(Edge("t_horz", "u_crenellations", LinkType.SUB, w=1.0))
    g.add_edge(Edge("t_vert", "u_towers", LinkType.SUB, w=1.0))
    # Gate uses OR logic with both mean and vert
    g.add_edge(Edge("t_mean", "u_gate", LinkType.SUB, w=0.7))
    g.add_edge(Edge("t_vert", "u_gate", LinkType.SUB, w=0.7))

    # Hierarchy SUR and child SUB back to root
    for child in ["u_wall", "u_crenellations", "u_towers", "u_gate"]:
        g.add_edge(Edge("u_root", child, LinkType.SUR, w=1.0))
        g.add_edge(Edge(child, "u_root", LinkType.SUB, w=1.0))

    # Sequence POR: wall -> towers -> crenellations -> gate
    g.add_edge(Edge("u_wall", "u_towers", LinkType.POR, w=1.0))
    g.add_edge(Edge("u_towers", "u_crenellations", LinkType.POR, w=1.0))
    g.add_edge(Edge("u_crenellations", "u_gate", LinkType.POR, w=1.0))
    
    return g


def init_terminals_from_image(g: Graph, img: np.ndarray) -> Dict[str, float]:
    feats = terminals_from_image(img)
    for tid, val in feats.items():
        if tid in g.units:
            u = g.units[tid]
            u.a = float(val)
            u.state = State.REQUESTED if val > 0.1 else State.INACTIVE
    # Kickstart root
    g.units["u_root"].a = 1.0
    g.units["u_root"].state = State.ACTIVE
    return feats


# ---------- Scenes ----------


def _simulate_and_capture_messages(g: Graph, steps: int = 8) -> List[dict]:
    """Run the engine for a few steps and capture delivered messages with timestamps."""
    captured: List[dict] = []

    class RecordingEngine(Engine):
        def send_message(self, sender_id: str, receiver_id: str, message):
            msg_name = getattr(message, "name", str(message))
            captured.append(
                {
                    "t": self.t,
                    "sender": sender_id,
                    "receiver": receiver_id,
                    "message": msg_name,
                }
            )
            super().send_message(sender_id, receiver_id, message)

    eng = RecordingEngine(g, EngineConfig())
    eng.step(steps)
    return captured


def _group_messages_by_time(captured: List[dict]) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = {}
    for m in captured:
        try:
            t = int(m.get("t", 0))
        except (ValueError, TypeError):
            t = 0
        grouped.setdefault(t, []).append(m)
    return dict(sorted(grouped.items(), key=lambda kv: kv[0]))


def _color_for_message(msg_name: str):
    if msg_name == "REQUEST":
        return YELLOW
    if msg_name == "CONFIRM":
        return GREEN
    if msg_name in ("INHIBIT_REQUEST", "INHIBIT_CONFIRM"):
        return RED
    return WHITE


class RootActivationScene(BaseReconScene):
    """Scene showing root node activation and initial message propagation to children."""

    def construct(self):
        # Title card
        title = Text("Request Confirmation Network", font_size=32)
        subtitle = Text("Active Perception on a Synthetic House", font_size=22)
        subtitle.next_to(title, DOWN)
        self.play(FadeIn(title), FadeIn(subtitle), run_time=0.6)
        self.wait(1.0)
        self.play(FadeOut(title), FadeOut(subtitle), run_time=0.6)

        # 1) House image on the left
        img = make_house_scene(size=64, noise=0.05)
        pil_arr = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        house = ImageMobject(pil_arr).scale(5.5)
        house.move_to([-4.5, -0.5, 0])
        self.play(FadeIn(house), run_time=1.0)

        # Add house title
        house_title = Text("Synthetic House", font_size=24, color=WHITE)
        house_title.next_to(house, UP, buff=0.3)
        self.play(FadeIn(house_title), run_time=0.5)

        # Add descriptive text above the scene
        description_text = Text(
            "Initializing network structure...", font_size=20, color=WHITE
        )
        description_text.to_edge(UP, buff=0.5)
        self.play(FadeIn(description_text), run_time=0.5)

        # 2) Build network viz on the right
        g = build_house_graph()

        # Node layout positions (centered on right two-thirds)
        right_x = 2.0
        y_root = 1.5
        spread = 3.0
        node_positions = {
            "u_root": (right_x, y_root, 0),
            "u_roof": (right_x - spread, y_root - 2.0, 0),
            "u_body": (right_x, y_root - 2.0, 0),
            "u_door": (right_x + spread, y_root - 2.0, 0),
            "t_horz": (right_x - spread, y_root - 4.0, 0),
            "t_mean": (right_x, y_root - 4.0, 0),
            "t_vert": (right_x + spread, y_root - 4.0, 0),
        }

        nodes, node_group = self.build_nodes(g, node_positions)
        self.play(FadeIn(node_group), run_time=2.0, rate_func=rf.ease_in_out_sine)

        # Add "House Hypothesis" title above the root node
        root_hypothesis_title = Text("House Hypothesis", font_size=20, color=WHITE)
        root_hypothesis_title.next_to(nodes["u_root"].shape, UP, buff=0.4)
        self.play(FadeIn(root_hypothesis_title), run_time=0.5)

        # Update description text
        new_description = Text(
            "Network nodes created - Root, Scripts, and Terminals",
            font_size=20,
            color=WHITE,
        )
        new_description.to_edge(UP, buff=0.5)
        self.play(FadeOut(description_text), FadeIn(new_description), run_time=0.5)
        description_text = new_description

        background_edges, sur_edges, sub_edges, por_edges = self.compute_edges(g, nodes)

        # Draw background edges first (behind nodes)
        for edge in background_edges:
            edge.set_z_index(-1)
        self.play(*[FadeIn(m) for m in background_edges], lag_ratio=0.02, run_time=2.0)

        # Update description text
        new_description = Text(
            "Network connections established - SUR, SUB, and POR links",
            font_size=20,
            color=WHITE,
        )
        new_description.to_edge(UP, buff=0.5)
        self.play(FadeOut(description_text), FadeIn(new_description), run_time=0.5)
        description_text = new_description

        # Wait a moment to show the network
        self.wait(1.0)

        # 3) Show root node activation
        root_node = nodes["u_root"]

        # Update description text
        new_description = Text(
            "Root node activation - Starting the perception process",
            font_size=20,
            color=YELLOW,
        )
        new_description.to_edge(UP, buff=0.5)
        self.play(FadeOut(description_text), FadeIn(new_description), run_time=0.5)
        description_text = new_description

        # Highlight the root node activation with color change
        root_highlight = self.create_highlight_shape(root_node, YELLOW, stroke_width=4)
        root_highlight.move_to(root_node.shape.get_center())

        self.play(
            FadeIn(root_highlight),
            root_node.shape.animate.set_fill(YELLOW, opacity=0.3),
            run_time=1.0,
        )

        # Add activation text
        activation_text = Text("ACTIVATED", font_size=16, color=YELLOW)
        activation_text.next_to(root_node.shape, UP, buff=0.2)
        self.play(FadeIn(activation_text), run_time=0.5)

        self.wait(0.5)

        # 4) Animate REQUEST messages moving to children
        # The SUR edges are: u_root -> u_body, u_root -> u_door

        # Update description text
        new_description = Text(
            "Sending REQUEST messages to child scripts", font_size=20, color=YELLOW
        )
        new_description.to_edge(UP, buff=0.5)
        self.play(FadeOut(description_text), FadeIn(new_description), run_time=0.5)
        description_text = new_description

        message_animations = []
        for src_id, dst_id in sur_edges:
            if src_id == "u_root":
                anim = self.animate_message_between_nodes(
                    nodes[src_id], nodes[dst_id], "REQUEST", YELLOW, 1.2
                )
                message_animations.append(anim)

        # Play all message animations in parallel
        self.play(*message_animations, run_time=2.5)

        # 5) Show child scripts becoming ACTIVE and sending requests to terminals
        self.wait(1.0)

        # Update description text
        new_description = Text(
            "Child scripts become ACTIVE and request terminal data",
            font_size=20,
            color=BLUE,
        )
        new_description.to_edge(UP, buff=0.5)
        self.play(FadeOut(description_text), FadeIn(new_description), run_time=0.5)
        description_text = new_description

        # Child scripts transition to ACTIVE (roof, body, door)
        active_animations = []
        for script_id in ["u_roof", "u_body", "u_door"]:
            script_node = nodes[script_id]
            # Change color to indicate ACTIVE state
            active_animations.append(
                script_node.shape.animate.set_fill(BLUE, opacity=0.3)
            )

        self.play(*active_animations, run_time=0.8)

        # Add ACTIVE labels
        active_labels = []
        for script_id in ["u_roof", "u_body", "u_door"]:
            script_node = nodes[script_id]
            active_label = Text("ACTIVE", font_size=12, color=BLUE)
            active_label.next_to(script_node.shape, UP, buff=0.1)
            active_labels.append(active_label)

        self.play(*[FadeIn(label) for label in active_labels], run_time=0.5)

        # 6) Send REQUEST messages from scripts to their terminals
        self.wait(0.5)

        terminal_request_animations = []

        # Roof script -> horizontal terminal
        roof_to_horz = self.animate_message_between_nodes(
            nodes["u_roof"], nodes["t_horz"], "REQUEST", YELLOW, 1.2
        )
        terminal_request_animations.append(roof_to_horz)

        # Body script -> mean terminal
        body_to_mean = self.animate_message_between_nodes(
            nodes["u_body"], nodes["t_mean"], "REQUEST", YELLOW, 1.2
        )
        terminal_request_animations.append(body_to_mean)

        # Door script -> vertical terminal
        door_to_vert = self.animate_message_between_nodes(
            nodes["u_door"], nodes["t_vert"], "REQUEST", YELLOW, 1.2
        )
        terminal_request_animations.append(door_to_vert)

        # Door script -> mean terminal (shared)
        door_to_mean = self.animate_message_between_nodes(
            nodes["u_door"], nodes["t_mean"], "REQUEST", YELLOW, 1.2
        )
        terminal_request_animations.append(door_to_mean)

        self.play(*terminal_request_animations, run_time=2.0)

        # 7) Terminals detect features and send CONFIRM messages back
        self.wait(1.0)

        # Update description text
        new_description = Text(
            "Terminals detect features and become TRUE", font_size=20, color=GREEN
        )
        new_description.to_edge(UP, buff=0.5)
        self.play(FadeOut(description_text), FadeIn(new_description), run_time=0.5)
        description_text = new_description

        # Terminals become TRUE (detect features)
        terminal_true_animations = []
        for term_id in ["t_horz", "t_mean", "t_vert"]:
            term_node = nodes[term_id]
            terminal_true_animations.append(
                term_node.shape.animate.set_fill(GREEN, opacity=0.4)
            )

        self.play(*terminal_true_animations, run_time=0.8)

        # Add TRUE labels to terminals
        true_labels = []
        for term_id in ["t_horz", "t_mean", "t_vert"]:
            term_node = nodes[term_id]
            true_label = Text("TRUE", font_size=10, color=GREEN)
            true_label.next_to(term_node.shape, DOWN, buff=0.1)
            true_labels.append(true_label)

        self.play(*[FadeIn(label) for label in true_labels], run_time=0.5)

        # Send CONFIRM messages back to scripts
        self.wait(0.5)

        # Update description text
        new_description = Text(
            "Terminals send CONFIRM messages back to scripts", font_size=20, color=GREEN
        )
        new_description.to_edge(UP, buff=0.5)
        self.play(FadeOut(description_text), FadeIn(new_description), run_time=0.5)
        description_text = new_description

        confirm_animations = []

        # Horizontal terminal -> roof script
        horz_to_roof = self.animate_message_between_nodes(
            nodes["t_horz"], nodes["u_roof"], "CONFIRM", GREEN, 1.0
        )
        confirm_animations.append(horz_to_roof)

        # Mean terminal -> body script
        mean_to_body = self.animate_message_between_nodes(
            nodes["t_mean"], nodes["u_body"], "CONFIRM", GREEN, 1.0
        )
        confirm_animations.append(mean_to_body)

        # Vertical terminal -> door script
        vert_to_door = self.animate_message_between_nodes(
            nodes["t_vert"], nodes["u_door"], "CONFIRM", GREEN, 1.0
        )
        confirm_animations.append(vert_to_door)

        # Mean terminal -> door script (confirming the shared connection)
        mean_to_door = self.animate_message_between_nodes(
            nodes["t_mean"], nodes["u_door"], "CONFIRM", GREEN, 1.0
        )
        confirm_animations.append(mean_to_door)

        self.play(*confirm_animations, run_time=1.8)

        # 8) Scripts become CONFIRMED
        self.wait(0.8)

        # Update description text
        new_description = Text(
            "Scripts become CONFIRMED after receiving terminal data",
            font_size=20,
            color=GREEN,
        )
        new_description.to_edge(UP, buff=0.5)
        self.play(FadeOut(description_text), FadeIn(new_description), run_time=0.5)
        description_text = new_description

        confirmed_animations = []
        for script_id in ["u_roof", "u_body", "u_door"]:
            script_node = nodes[script_id]
            confirmed_animations.append(
                script_node.shape.animate.set_fill(GREEN, opacity=0.5)
            )

        self.play(*confirmed_animations, run_time=0.8)

        # Update labels to CONFIRMED
        confirmed_labels = []
        for i, script_id in enumerate(["u_roof", "u_body", "u_door"]):
            old_label = active_labels[i]
            confirmed_label = Text("CONFIRMED", font_size=10, color=GREEN)
            confirmed_label.move_to(old_label.get_center())
            confirmed_labels.append(confirmed_label)

        self.play(
            *[FadeOut(label) for label in active_labels],
            *[FadeIn(label) for label in confirmed_labels],
            run_time=0.6,
        )

        # 9) Scripts send CONFIRM messages back to root
        self.wait(1.0)

        # Update description text
        new_description = Text(
            "Scripts send CONFIRM messages back to root", font_size=20, color=GREEN
        )
        new_description.to_edge(UP, buff=0.5)
        self.play(FadeOut(description_text), FadeIn(new_description), run_time=0.5)
        description_text = new_description

        # Animate all confirmations simultaneously
        root_confirm_animations = []

        # Roof script -> root
        roof_to_root = self.animate_message_to_root(
            nodes["u_roof"], nodes["u_root"], "CONFIRM", GREEN, 1.5
        )
        root_confirm_animations.append(roof_to_root)

        # Body script -> root
        body_to_root = self.animate_message_to_root(
            nodes["u_body"], nodes["u_root"], "CONFIRM", GREEN, 1.5
        )
        root_confirm_animations.append(body_to_root)

        # Door script -> root
        door_to_root = self.animate_message_to_root(
            nodes["u_door"], nodes["u_root"], "CONFIRM", GREEN, 1.5
        )
        root_confirm_animations.append(door_to_root)

        # Play all confirmations simultaneously
        self.play(*root_confirm_animations, run_time=1.5)

        # 10) Root becomes CONFIRMED after receiving all confirmations
        self.wait(0.8)

        # Update description text
        new_description = Text(
            "Root becomes CONFIRMED - Perception process complete",
            font_size=20,
            color=GREEN,
        )
        new_description.to_edge(UP, buff=0.5)
        self.play(FadeOut(description_text), FadeIn(new_description), run_time=0.5)
        description_text = new_description

        # Highlight the root node to show it received all confirmations
        root_highlight = self.create_highlight_shape(
            nodes["u_root"], GREEN, stroke_width=4
        )
        root_highlight.move_to(nodes["u_root"].shape.get_center())

        # Root transitions to CONFIRMED state
        root_confirmed_fill = nodes["u_root"].shape.animate.set_fill(GREEN, opacity=0.6)
        root_confirmed_label = Text("CONFIRMED", font_size=12, color=GREEN)
        root_confirmed_label.move_to(activation_text.get_center())

        self.play(
            FadeIn(root_highlight),
            FadeOut(activation_text),
            FadeIn(root_confirmed_label),
            root_confirmed_fill,
            run_time=0.8,
        )

        # Hold the final state
        self.wait(2.0)


class HouseWalkthrough(BaseReconScene):
    """Single, concise walkthrough scene."""

    def construct(self):
        # 1) House image on the left
        img = make_house_scene(size=64, noise=0.05)
        # Normalize to 0..255 grayscale
        pil_arr = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        house = ImageMobject(pil_arr).scale(5.5)
        house.move_to([-4.5, -0.5, 0])  # Center on left third of screen, adjusted

        # Title card
        title = Text("Request Confirmation Network", font_size=32)
        subtitle = Text("Active Perception on a Synthetic House", font_size=22)
        subtitle.next_to(title, DOWN)
        self.play(FadeIn(title), FadeIn(subtitle), run_time=1.5)
        self.wait(1.0)
        self.play(FadeOut(title), FadeOut(subtitle), run_time=1.5)

        self.play(FadeIn(house), run_time=2.0)

        # 2) Build network viz on the right
        g = build_house_graph()
        init_terminals_from_image(g, img)

        # Node layout positions (centered on right two-thirds)
        right_x = 2.0  # Center of right two-thirds
        y_root = 1.5  # Adjusted for square terminals
        spread = 3.0  # Increased spread for better spacing
        node_positions = {
            "u_root": (right_x, y_root, 0),
            "u_roof": (right_x - spread, y_root - 2.0, 0),
            "u_body": (right_x, y_root - 2.0, 0),
            "u_door": (right_x + spread, y_root - 2.0, 0),
            "t_horz": (right_x - spread, y_root - 4.0, 0),
            "t_mean": (right_x, y_root - 4.0, 0),
            "t_vert": (right_x + spread, y_root - 4.0, 0),
        }

        nodes, node_group = self.build_nodes(g, node_positions)
        self.play(FadeIn(node_group), run_time=3.0, rate_func=rf.ease_in_out_sine)

        background_edges, sur_edges, sub_edges, por_edges = self.compute_edges(g, nodes)

        # Draw background edges first (behind nodes)
        for edge in background_edges:
            edge.set_z_index(-1)  # Behind nodes
        self.play(*[FadeIn(m) for m in background_edges], lag_ratio=0.02, run_time=2.5)

        # Graph is now built and displayed
        self.wait(2.0)


class HouseHypothesisOnCastle(BaseReconScene):
    """Accurate message animation by simulating the engine on a castle image."""

    def construct(self):
        # 1) Castle image on the left
        img = make_castle_scene(size=64, noise=0.05)
        pil_arr = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        castle = ImageMobject(pil_arr).scale(5.5)
        castle.move_to([-4.5, -0.5, 0])

        # Title
        title = Text("House Hypothesis on Castle", font_size=30)
        subtitle = Text(
            "Testing hypothesis on different image - True engine messages", font_size=20
        )
        subtitle.next_to(title, DOWN)
        self.play(FadeIn(title), FadeIn(subtitle), run_time=1.0)
        self.wait(0.4)
        self.play(FadeOut(title), FadeOut(subtitle), run_time=0.6)
        self.play(FadeIn(castle), run_time=1.0)

        # 2) House graph on the right, initialized from castle terminals
        g = build_house_graph()
        init_terminals_from_image(g, img)

        right_x = 2.0
        y_root = 1.5
        spread = 3.0
        node_positions = {
            "u_root": (right_x, y_root, 0),
            "u_roof": (right_x - spread, y_root - 2.0, 0),
            "u_body": (right_x, y_root - 2.0, 0),
            "u_door": (right_x + spread, y_root - 2.0, 0),
            "t_horz": (right_x - spread, y_root - 4.0, 0),
            "t_mean": (right_x, y_root - 4.0, 0),
            "t_vert": (right_x + spread, y_root - 4.0, 0),
        }

        nodes, node_group = self.build_nodes(g, node_positions)
        self.play(FadeIn(node_group), run_time=1.2)

        background_edges, sur_edges, sub_edges, por_edges = self.compute_edges(g, nodes)
        for edge in background_edges:
            edge.set_z_index(-1)
        self.play(*[FadeIn(m) for m in background_edges], lag_ratio=0.02, run_time=1.0)

        # 3) Simulate engine and capture real messages
        captured = _simulate_and_capture_messages(g, steps=8)
        grouped = _group_messages_by_time(captured)

        # 4) Animate per-step message batches
        for _, msgs in grouped.items():
            anims = []
            for m in msgs:
                s = m.get("sender")
                r = m.get("receiver")
                msg_name = str(m.get("message"))
                if s in nodes and r in nodes:
                    color = _color_for_message(msg_name)
                    anims.append(
                        self.animate_message_between_nodes(
                            nodes[s], nodes[r], msg_name, color, 0.9
                        )
                    )
            if anims:
                self.play(*anims, run_time=1.4)
                self.wait(0.1)

        # Freeze last frame
        self.wait(1.6)


class CastleActivationScene(BaseReconScene):
    """Scene showing castle root node activation and message propagation."""

    def construct(self):
        # Title card
        title = Text("Request Confirmation Network", font_size=32)
        subtitle = Text("Active Perception on a Synthetic Castle", font_size=22)
        subtitle.next_to(title, DOWN)
        self.play(FadeIn(title), FadeIn(subtitle), run_time=0.6)
        self.wait(1.0)
        self.play(FadeOut(title), FadeOut(subtitle), run_time=0.6)

        # 1) Castle image on the left
        img = make_castle_scene(size=64, noise=0.05)
        pil_arr = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        castle = ImageMobject(pil_arr).scale(5.5)
        castle.move_to([-4.5, -0.5, 0])
        self.play(FadeIn(castle), run_time=1.0)

        # Add castle title
        castle_title = Text("Synthetic Castle", font_size=24, color=WHITE)
        castle_title.next_to(castle, UP, buff=0.3)
        self.play(FadeIn(castle_title), run_time=0.5)

        # Add descriptive text above the scene
        description_text = Text(
            "Initializing castle network structure...", font_size=20, color=WHITE
        )
        description_text.to_edge(UP, buff=0.5)
        self.play(FadeIn(description_text), run_time=0.5)

        # 2) Build castle network viz on the right
        g = build_castle_graph()

        # Node layout positions (centered on right two-thirds)
        right_x = 2.0
        y_root = 1.5
        spread = 2.5
        node_positions = {
            "u_root": (right_x, y_root, 0),
            "u_wall": (right_x - spread, y_root - 1.5, 0),
            "u_towers": (right_x - spread/2, y_root - 1.5, 0),
            "u_crenellations": (right_x + spread/2, y_root - 1.5, 0),
            "u_gate": (right_x + spread, y_root - 1.5, 0),
            "t_horz": (right_x - spread, y_root - 3.5, 0),
            "t_mean": (right_x - spread/2, y_root - 3.5, 0),
            "t_vert": (right_x + spread/2, y_root - 3.5, 0),
        }

        nodes, node_group = self.build_nodes(g, node_positions)
        self.play(FadeIn(node_group), run_time=2.0, rate_func=rf.ease_in_out_sine)

        # Add "Castle Hypothesis" title above the root node
        root_hypothesis_title = Text("Castle Hypothesis", font_size=20, color=WHITE)
        root_hypothesis_title.next_to(nodes["u_root"].shape, UP, buff=0.4)
        self.play(FadeIn(root_hypothesis_title), run_time=0.5)

        # Update description text
        new_description = Text(
            "Castle network nodes created - Root, Components, and Terminals",
            font_size=20,
            color=WHITE,
        )
        new_description.to_edge(UP, buff=0.5)
        self.play(FadeOut(description_text), FadeIn(new_description), run_time=0.5)
        description_text = new_description

        background_edges, sur_edges, sub_edges, por_edges = self.compute_edges(g, nodes)

        # Draw background edges first (behind nodes)
        for edge in background_edges:
            edge.set_z_index(-1)
        self.play(*[FadeIn(m) for m in background_edges], lag_ratio=0.02, run_time=2.0)

        # Update description text
        new_description = Text(
            "Network connections established - SUR, SUB, and POR links",
            font_size=20,
            color=WHITE,
        )
        new_description.to_edge(UP, buff=0.5)
        self.play(FadeOut(description_text), FadeIn(new_description), run_time=0.5)
        description_text = new_description

        # Wait a moment to show the network
        self.wait(1.0)

        # 3) Show root node activation
        root_node = nodes["u_root"]

        # Update description text
        new_description = Text(
            "Root node activation - Starting castle perception process",
            font_size=20,
            color=YELLOW,
        )
        new_description.to_edge(UP, buff=0.5)
        self.play(FadeOut(description_text), FadeIn(new_description), run_time=0.5)
        description_text = new_description

        # Highlight the root node activation with color change
        root_highlight = self.create_highlight_shape(root_node, YELLOW, stroke_width=4)
        root_highlight.move_to(root_node.shape.get_center())

        self.play(
            FadeIn(root_highlight),
            root_node.shape.animate.set_fill(YELLOW, opacity=0.3),
            run_time=1.0,
        )

        # Add activation text
        activation_text = Text("ACTIVATED", font_size=16, color=YELLOW)
        activation_text.next_to(root_node.shape, UP, buff=0.2)
        self.play(FadeIn(activation_text), run_time=0.5)

        self.wait(0.5)

        # 4) Animate REQUEST messages moving to children
        # The SUR edges are: u_root -> u_wall, u_root -> u_towers, u_root -> u_crenellations, u_root -> u_gate

        # Update description text
        new_description = Text(
            "Sending REQUEST messages to castle components", font_size=20, color=YELLOW
        )
        new_description.to_edge(UP, buff=0.5)
        self.play(FadeOut(description_text), FadeIn(new_description), run_time=0.5)
        description_text = new_description

        message_animations = []
        for src_id, dst_id in sur_edges:
            if src_id == "u_root":
                anim = self.animate_message_between_nodes(
                    nodes[src_id], nodes[dst_id], "REQUEST", YELLOW, 1.2
                )
                message_animations.append(anim)

        # Play all message animations in parallel
        self.play(*message_animations, run_time=2.5)

        # 5) Show castle components becoming ACTIVE and sending requests to terminals
        self.wait(1.0)

        # Update description text
        new_description = Text(
            "Castle components become ACTIVE and request terminal data",
            font_size=20,
            color=BLUE,
        )
        new_description.to_edge(UP, buff=0.5)
        self.play(FadeOut(description_text), FadeIn(new_description), run_time=0.5)
        description_text = new_description

        # Castle components transition to ACTIVE
        active_animations = []
        for script_id in ["u_wall", "u_towers", "u_crenellations", "u_gate"]:
            script_node = nodes[script_id]
            # Change color to indicate ACTIVE state
            active_animations.append(
                script_node.shape.animate.set_fill(BLUE, opacity=0.3)
            )

        self.play(*active_animations, run_time=0.8)

        # Add ACTIVE labels
        active_labels = []
        for script_id in ["u_wall", "u_towers", "u_crenellations", "u_gate"]:
            script_node = nodes[script_id]
            active_label = Text("ACTIVE", font_size=10, color=BLUE)
            active_label.next_to(script_node.shape, UP, buff=0.1)
            active_labels.append(active_label)

        self.play(*[FadeIn(label) for label in active_labels], run_time=0.5)

        # 6) Send REQUEST messages from components to their terminals
        self.wait(0.5)

        terminal_request_animations = []

        # Wall script -> mean terminal
        wall_to_mean = self.animate_message_between_nodes(
            nodes["u_wall"], nodes["t_mean"], "REQUEST", YELLOW, 1.2
        )
        terminal_request_animations.append(wall_to_mean)

        # Towers script -> vertical terminal
        towers_to_vert = self.animate_message_between_nodes(
            nodes["u_towers"], nodes["t_vert"], "REQUEST", YELLOW, 1.2
        )
        terminal_request_animations.append(towers_to_vert)

        # Crenellations script -> horizontal terminal
        crenellations_to_horz = self.animate_message_between_nodes(
            nodes["u_crenellations"], nodes["t_horz"], "REQUEST", YELLOW, 1.2
        )
        terminal_request_animations.append(crenellations_to_horz)

        # Gate script -> both mean and vertical terminals (OR logic)
        gate_to_mean = self.animate_message_between_nodes(
            nodes["u_gate"], nodes["t_mean"], "REQUEST", YELLOW, 1.2
        )
        terminal_request_animations.append(gate_to_mean)
        
        gate_to_vert = self.animate_message_between_nodes(
            nodes["u_gate"], nodes["t_vert"], "REQUEST", YELLOW, 1.2
        )
        terminal_request_animations.append(gate_to_vert)

        self.play(*terminal_request_animations, run_time=2.0)

        # 7) Terminals detect features and send CONFIRM messages back
        self.wait(1.0)

        # Update description text
        new_description = Text(
            "Terminals detect features and become TRUE", font_size=20, color=GREEN
        )
        new_description.to_edge(UP, buff=0.5)
        self.play(FadeOut(description_text), FadeIn(new_description), run_time=0.5)
        description_text = new_description

        # Terminals become TRUE (detect features)
        terminal_true_animations = []
        for term_id in ["t_horz", "t_mean", "t_vert"]:
            term_node = nodes[term_id]
            terminal_true_animations.append(
                term_node.shape.animate.set_fill(GREEN, opacity=0.4)
            )

        self.play(*terminal_true_animations, run_time=0.8)

        # Add TRUE labels to terminals
        true_labels = []
        for term_id in ["t_horz", "t_mean", "t_vert"]:
            term_node = nodes[term_id]
            true_label = Text("TRUE", font_size=10, color=GREEN)
            true_label.next_to(term_node.shape, DOWN, buff=0.1)
            true_labels.append(true_label)

        self.play(*[FadeIn(label) for label in true_labels], run_time=0.5)

        # Send CONFIRM messages back to scripts
        self.wait(0.5)

        # Update description text
        new_description = Text(
            "Terminals send CONFIRM messages back to components", font_size=20, color=GREEN
        )
        new_description.to_edge(UP, buff=0.5)
        self.play(FadeOut(description_text), FadeIn(new_description), run_time=0.5)
        description_text = new_description

        confirm_animations = []

        # Mean terminal -> wall script
        mean_to_wall = self.animate_message_between_nodes(
            nodes["t_mean"], nodes["u_wall"], "CONFIRM", GREEN, 1.0
        )
        confirm_animations.append(mean_to_wall)

        # Vertical terminal -> towers script
        vert_to_towers = self.animate_message_between_nodes(
            nodes["t_vert"], nodes["u_towers"], "CONFIRM", GREEN, 1.0
        )
        confirm_animations.append(vert_to_towers)

        # Horizontal terminal -> crenellations script
        horz_to_crenellations = self.animate_message_between_nodes(
            nodes["t_horz"], nodes["u_crenellations"], "CONFIRM", GREEN, 1.0
        )
        confirm_animations.append(horz_to_crenellations)

        # Both terminals -> gate script (confirming the OR connection)
        mean_to_gate = self.animate_message_between_nodes(
            nodes["t_mean"], nodes["u_gate"], "CONFIRM", GREEN, 1.0
        )
        confirm_animations.append(mean_to_gate)
        
        vert_to_gate = self.animate_message_between_nodes(
            nodes["t_vert"], nodes["u_gate"], "CONFIRM", GREEN, 1.0
        )
        confirm_animations.append(vert_to_gate)

        self.play(*confirm_animations, run_time=1.8)

        # 8) Components become CONFIRMED
        self.wait(0.8)

        # Update description text
        new_description = Text(
            "Castle components become CONFIRMED after receiving terminal data",
            font_size=20,
            color=GREEN,
        )
        new_description.to_edge(UP, buff=0.5)
        self.play(FadeOut(description_text), FadeIn(new_description), run_time=0.5)
        description_text = new_description

        confirmed_animations = []
        for script_id in ["u_wall", "u_towers", "u_crenellations", "u_gate"]:
            script_node = nodes[script_id]
            confirmed_animations.append(
                script_node.shape.animate.set_fill(GREEN, opacity=0.5)
            )

        self.play(*confirmed_animations, run_time=0.8)

        # Update labels to CONFIRMED
        confirmed_labels = []
        for i, script_id in enumerate(["u_wall", "u_towers", "u_crenellations", "u_gate"]):
            old_label = active_labels[i]
            confirmed_label = Text("CONFIRMED", font_size=8, color=GREEN)
            confirmed_label.move_to(old_label.get_center())
            confirmed_labels.append(confirmed_label)

        self.play(
            *[FadeOut(label) for label in active_labels],
            *[FadeIn(label) for label in confirmed_labels],
            run_time=0.6,
        )

        # 9) Components send CONFIRM messages back to root
        self.wait(1.0)

        # Update description text
        new_description = Text(
            "Components send CONFIRM messages back to root", font_size=20, color=GREEN
        )
        new_description.to_edge(UP, buff=0.5)
        self.play(FadeOut(description_text), FadeIn(new_description), run_time=0.5)
        description_text = new_description

        # Animate all confirmations simultaneously
        root_confirm_animations = []

        # All components -> root
        for component_id in ["u_wall", "u_towers", "u_crenellations", "u_gate"]:
            component_to_root = self.animate_message_to_root(
                nodes[component_id], nodes["u_root"], "CONFIRM", GREEN, 1.5
            )
            root_confirm_animations.append(component_to_root)

        # Play all confirmations simultaneously
        self.play(*root_confirm_animations, run_time=1.5)

        # 10) Root becomes CONFIRMED after receiving all confirmations
        self.wait(0.8)

        # Update description text
        new_description = Text(
            "Root becomes CONFIRMED - Castle perception process complete",
            font_size=20,
            color=GREEN,
        )
        new_description.to_edge(UP, buff=0.5)
        self.play(FadeOut(description_text), FadeIn(new_description), run_time=0.5)
        description_text = new_description

        # Highlight the root node to show it received all confirmations
        root_highlight = self.create_highlight_shape(
            nodes["u_root"], GREEN, stroke_width=4
        )
        root_highlight.move_to(nodes["u_root"].shape.get_center())

        # Root transitions to CONFIRMED state
        root_confirmed_fill = nodes["u_root"].shape.animate.set_fill(GREEN, opacity=0.6)
        root_confirmed_label = Text("CONFIRMED", font_size=12, color=GREEN)
        root_confirmed_label.move_to(activation_text.get_center())

        self.play(
            FadeIn(root_highlight),
            FadeOut(activation_text),
            FadeIn(root_confirmed_label),
            root_confirmed_fill,
            run_time=0.8,
        )

        # Hold the final state
        self.wait(2.0)


class CastleWalkthrough(BaseReconScene):
    """Single, concise castle walkthrough scene."""

    def construct(self):
        # 1) Castle image on the left
        img = make_castle_scene(size=64, noise=0.05)
        # Normalize to 0..255 grayscale
        pil_arr = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        castle = ImageMobject(pil_arr).scale(5.5)
        castle.move_to([-4.5, -0.5, 0])  # Center on left third of screen, adjusted

        # Title card
        title = Text("Request Confirmation Network", font_size=32)
        subtitle = Text("Active Perception on a Synthetic Castle", font_size=22)
        subtitle.next_to(title, DOWN)
        self.play(FadeIn(title), FadeIn(subtitle), run_time=1.5)
        self.wait(1.0)
        self.play(FadeOut(title), FadeOut(subtitle), run_time=1.5)

        self.play(FadeIn(castle), run_time=2.0)

        # 2) Build castle network viz on the right
        g = build_castle_graph()
        init_terminals_from_image(g, img)

        # Node layout positions (centered on right two-thirds)
        right_x = 2.0  # Center of right two-thirds
        y_root = 1.5  # Adjusted for square terminals
        spread = 2.5  # Spread for castle components
        node_positions = {
            "u_root": (right_x, y_root, 0),
            "u_wall": (right_x - spread, y_root - 1.5, 0),
            "u_towers": (right_x - spread/2, y_root - 1.5, 0),
            "u_crenellations": (right_x + spread/2, y_root - 1.5, 0),
            "u_gate": (right_x + spread, y_root - 1.5, 0),
            "t_horz": (right_x - spread, y_root - 3.5, 0),
            "t_mean": (right_x - spread/2, y_root - 3.5, 0),
            "t_vert": (right_x + spread/2, y_root - 3.5, 0),
        }

        nodes, node_group = self.build_nodes(g, node_positions)
        self.play(FadeIn(node_group), run_time=3.0, rate_func=rf.ease_in_out_sine)

        background_edges, sur_edges, sub_edges, por_edges = self.compute_edges(g, nodes)

        # Draw background edges first (behind nodes)
        for edge in background_edges:
            edge.set_z_index(-1)  # Behind nodes
        self.play(*[FadeIn(m) for m in background_edges], lag_ratio=0.02, run_time=2.5)

        # Graph is now built and displayed
        self.wait(2.0)


# README (rendering notes):
# - Install Manim Community (and Cairo/ffmpeg per platform):
#     pip install manim
# - Render a quick, low-res preview:
#     make render-fast
#     # Or manually: manim -ql --fps 30 --media_dir output/media -o recon_house_walkthrough_preview scripts/manim_recon_house.py HouseWalkthrough
# - For higher quality (e.g., 60 fps, 1080p), use:
#     make render
#     # Or manually: manim -qp --fps 60 -r 1920,1080 --media_dir output/media -o recon_house_walkthrough scripts/manim_recon_house.py HouseWalkthrough
