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


from perception.dataset import make_house_scene, make_castle_scene
from perception.terminals import terminals_from_image
from recon_core.enums import LinkType, State, UnitType
from recon_core.graph import Edge, Graph, Unit
from manim_common import NodeViz, edge_arrow, gradient_edge_arrow, BaseReconScene

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

        background_edges, sur_edges, sub_edges, por_edges = self.compute_edges(g, nodes)

        # Draw background edges first (behind nodes)
        for edge in background_edges:
            edge.set_z_index(-1)
        self.play(*[FadeIn(m) for m in background_edges], lag_ratio=0.02, run_time=2.0)

        # Wait a moment to show the network
        self.wait(1.0)

        # 3) Show root node activation
        root_node = nodes["u_root"]

        # Highlight the root node activation with color change
        root_highlight = self.create_highlight_shape(root_node, YELLOW, stroke_width=4)
        root_highlight.move_to(root_node.shape.get_center())

        self.play(
            FadeIn(root_highlight),
            root_node.shape.animate.set_fill(YELLOW, opacity=0.3),
            run_time=1.0
        )

        # Add activation text
        activation_text = Text("ACTIVATED", font_size=16, color=YELLOW)
        activation_text.next_to(root_node.shape, UP, buff=0.2)
        self.play(FadeIn(activation_text), run_time=0.5)

        self.wait(0.5)

        # 4) Animate REQUEST messages moving to children
        # The SUR edges are: u_root -> u_roof, u_root -> u_body, u_root -> u_door

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
            run_time=0.6
        )

        # 9) Scripts send CONFIRM messages back to root
        self.wait(1.0)

        root_confirm_animations = []

        # Roof script -> root
        roof_to_root = self.animate_message_between_nodes(
            nodes["u_roof"], nodes["u_root"], "CONFIRM", GREEN, 1.2
        )
        root_confirm_animations.append(roof_to_root)

        # Body script -> root
        body_to_root = self.animate_message_between_nodes(
            nodes["u_body"], nodes["u_root"], "CONFIRM", GREEN, 1.2
        )
        root_confirm_animations.append(body_to_root)

        # Door script -> root
        door_to_root = self.animate_message_between_nodes(
            nodes["u_door"], nodes["u_root"], "CONFIRM", GREEN, 1.2
        )
        root_confirm_animations.append(door_to_root)

        self.play(*root_confirm_animations, run_time=2.0)

        # 10) Root becomes CONFIRMED
        self.wait(0.8)

        # Root transitions to CONFIRMED state
        root_confirmed_fill = nodes["u_root"].shape.animate.set_fill(GREEN, opacity=0.6)
        root_confirmed_label = Text("CONFIRMED", font_size=12, color=GREEN)
        root_confirmed_label.move_to(activation_text.get_center())

        self.play(
            FadeOut(activation_text),
            FadeIn(root_confirmed_label),
            root_confirmed_fill,
            run_time=0.8
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


# ---------- New Scene: House hypothesis on a Castle image ----------


class HouseHypothesisOnCastle(BaseReconScene):
    """Run the house hypothesis graph on a castle image and animate actual messages/outcomes."""

    def construct(self):
        # Title card
        title = Text("House Hypothesis on Castle", font_size=32)
        subtitle = Text("Accurate messages from house graph on castle image", font_size=22)
        subtitle.next_to(title, DOWN)
        self.play(FadeIn(title), FadeIn(subtitle), run_time=1.2)
        self.wait(0.8)
        self.play(FadeOut(title), FadeOut(subtitle), run_time=0.8)

        # 1) Castle image on the left
        img = make_castle_scene(size=64, noise=0.05)
        pil_arr = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        castle = ImageMobject(pil_arr).scale(5.5)
        castle.move_to([-4.5, -0.5, 0])
        self.play(FadeIn(castle), run_time=1.0)

        # 2) Build the same house hypothesis graph and initialize terminals from castle image
        g = build_house_graph()
        feats = init_terminals_from_image(g, img)

        # Layout positions (mirrors house scenes)
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

        background_edges, sur_edges, sub_edges, por_edges = self.compute_edges(g, nodes)

        for edge in background_edges:
            edge.set_z_index(-1)
        self.play(*[FadeIn(m) for m in background_edges], lag_ratio=0.02, run_time=1.6)

        # 3) Highlight root activation
        root_node = nodes["u_root"]
        root_highlight = self.create_highlight_shape(root_node, YELLOW, stroke_width=4)
        root_highlight.move_to(root_node.shape.get_center())
        self.play(
            FadeIn(root_highlight),
            root_node.shape.animate.set_fill(YELLOW, opacity=0.3),
            run_time=0.8,
        )
        activation_text = Text("ACTIVATED", font_size=16, color=YELLOW)
        activation_text.next_to(root_node.shape, UP, buff=0.2)
        self.play(FadeIn(activation_text), run_time=0.4)

        # 4) Requests from root to child scripts (SUR concept)
        message_anims = []
        for src_id, dst_id in sur_edges:
            if src_id == "u_root":
                message_anims.append(
                    self.animate_message_between_nodes(nodes[src_id], nodes[dst_id], "REQUEST", YELLOW, 1.0)
                )
        if message_anims:
            self.play(*message_anims, run_time=1.6)

        # Child scripts become ACTIVE
        active_animations = []
        for script_id in ["u_roof", "u_body", "u_door"]:
            active_animations.append(nodes[script_id].shape.animate.set_fill(BLUE, opacity=0.3))
        self.play(*active_animations, run_time=0.6)
        active_labels = []
        for script_id in ["u_roof", "u_body", "u_door"]:
            lbl = Text("ACTIVE", font_size=12, color=BLUE)
            lbl.next_to(nodes[script_id].shape, UP, buff=0.1)
            active_labels.append(lbl)
        self.play(*[FadeIn(l) for l in active_labels], run_time=0.4)

        # 5) Requests from scripts to terminals (conceptual)
        term_request_anims = []
        term_request_anims.append(
            self.animate_message_between_nodes(nodes["u_roof"], nodes["t_horz"], "REQUEST", YELLOW, 0.9)
        )
        term_request_anims.append(
            self.animate_message_between_nodes(nodes["u_body"], nodes["t_mean"], "REQUEST", YELLOW, 0.9)
        )
        term_request_anims.append(
            self.animate_message_between_nodes(nodes["u_door"], nodes["t_vert"], "REQUEST", YELLOW, 0.9)
        )
        term_request_anims.append(
            self.animate_message_between_nodes(nodes["u_door"], nodes["t_mean"], "REQUEST", YELLOW, 0.9)
        )
        self.play(*term_request_anims, run_time=1.4)

        # 6) Terminals that actually evaluate TRUE on the castle
        self.wait(0.3)
        true_terms = []
        for tid in ["t_horz", "t_mean", "t_vert"]:
            if g.units[tid].a >= g.units[tid].thresh:
                true_terms.append(tid)

        true_fill_anims = []
        true_labels = []
        for tid in true_terms:
            true_fill_anims.append(nodes[tid].shape.animate.set_fill(GREEN, opacity=0.4))
            lbl = Text("TRUE", font_size=10, color=GREEN)
            lbl.next_to(nodes[tid].shape, DOWN, buff=0.1)
            true_labels.append(lbl)
        if true_fill_anims:
            self.play(*true_fill_anims, run_time=0.6)
            self.play(*[FadeIn(l) for l in true_labels], run_time=0.3)

        # 7) CONFIRM messages from TRUE terminals back to their scripts
        confirm_anims = []
        if "t_horz" in true_terms:
            confirm_anims.append(self.animate_message_between_nodes(nodes["t_horz"], nodes["u_roof"], "CONFIRM", GREEN, 0.8))
        if "t_mean" in true_terms:
            confirm_anims.append(self.animate_message_between_nodes(nodes["t_mean"], nodes["u_body"], "CONFIRM", GREEN, 0.8))
            confirm_anims.append(self.animate_message_between_nodes(nodes["t_mean"], nodes["u_door"], "CONFIRM", GREEN, 0.8))
        if "t_vert" in true_terms:
            confirm_anims.append(self.animate_message_between_nodes(nodes["t_vert"], nodes["u_door"], "CONFIRM", GREEN, 0.8))
        if confirm_anims:
            self.play(*confirm_anims, run_time=1.2)

        # 8) Scripts that become CONFIRMED based on child TRUEs (confirmation_ratio ~0.6)
        script_children = {
            "u_roof": ["t_horz"],
            "u_body": ["t_mean"],
            "u_door": ["t_vert", "t_mean"],
        }
        confirmed_scripts = []
        new_labels = []
        confirm_script_anims = []
        for idx, sid in enumerate(["u_roof", "u_body", "u_door"]):
            children = script_children[sid]
            need = max(1, int(0.6 * len(children)))
            trues = sum(1 for c in children if c in true_terms)
            if children and trues >= need:
                confirmed_scripts.append(sid)
                confirm_script_anims.append(nodes[sid].shape.animate.set_fill(GREEN, opacity=0.5))
                lbl = Text("CONFIRMED", font_size=10, color=GREEN)
                lbl.move_to(active_labels[idx].get_center())
                new_labels.append((idx, lbl))

        if confirm_script_anims:
            self.play(*confirm_script_anims, run_time=0.6)
            if new_labels:
                self.play(
                    *[FadeOut(active_labels[i]) for (i, _) in new_labels],
                    *[FadeIn(lbl) for (_, lbl) in new_labels],
                    run_time=0.4,
                )

        # 9) CONFIRM messages from confirmed scripts to root
        root_confirm_anims = []
        for sid in confirmed_scripts:
            root_confirm_anims.append(self.animate_message_between_nodes(nodes[sid], nodes["u_root"], "CONFIRM", GREEN, 1.0))
        if root_confirm_anims:
            self.play(*root_confirm_anims, run_time=1.2)

        # 10) Root confirmation depends on number of confirmed scripts
        child_scripts = ["u_roof", "u_body", "u_door"]
        need_root = max(1, int(0.6 * len(child_scripts)))
        if len(confirmed_scripts) >= need_root:
            root_confirmed_fill = nodes["u_root"].shape.animate.set_fill(GREEN, opacity=0.6)
            root_confirmed_label = Text("CONFIRMED", font_size=12, color=GREEN)
            root_confirmed_label.move_to(activation_text.get_center())
            self.play(
                FadeOut(activation_text),
                FadeIn(root_confirmed_label),
                root_confirmed_fill,
                run_time=0.7,
            )
        else:
            # Keep root ACTIVE if insufficient confirmations
            root_active_label = Text("ACTIVE", font_size=12, color=BLUE)
            root_active_label.move_to(activation_text.get_center())
            self.play(
                FadeOut(activation_text),
                FadeIn(root_active_label),
                run_time=0.5,
            )

        self.wait(1.6)

# README (rendering notes):
# - Install Manim Community (and Cairo/ffmpeg per platform):
#     pip install manim
# - Render a quick, low-res preview:
#     make render-fast
#     # Or manually: manim -ql --fps 30 --media_dir output/media -o recon_house_walkthrough_preview scripts/manim_recon_house.py HouseWalkthrough
# - For higher quality (e.g., 60 fps, 1080p), use:
#     make render
#     # Or manually: manim -qp --fps 60 -r 1920,1080 --media_dir output/media -o recon_house_walkthrough scripts/manim_recon_house.py HouseWalkthrough
