import math
import os
import sys

import pytest

pytest.importorskip("manim")


def _ensure_scripts_on_path():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    scripts_dir = os.path.join(repo_root, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


def _skip_if_text_unavailable():
    from manim import Text

    try:
        _ = Text("ok", font_size=12)
    except (
        ImportError,
        RuntimeError,
        OSError,
    ) as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Skipping Manim Text-based tests: {exc}")


def test_nodeviz_label_and_shape_types():
    _ensure_scripts_on_path()
    from manim import Circle, Square
    from manim_common import (
        NodeViz,
    )  # Import added to path by _ensure_scripts_on_path()

    from recon_core.enums import UnitType

    _skip_if_text_unavailable()

    script_node = NodeViz("u_root", unit_type=UnitType.SCRIPT)
    term_node = NodeViz("t_mean", unit_type=UnitType.TERMINAL)

    # Labels trimmed
    assert script_node.label == "root"
    assert term_node.label == "mean"

    # Shapes
    assert isinstance(script_node.shape, Circle)
    assert isinstance(term_node.shape, Square)


def test_activation_meter_arc_properties():
    _ensure_scripts_on_path()
    from manim import Arc
    from manim_common import (
        NodeViz,
    )  # Import added to path by _ensure_scripts_on_path()

    from recon_core.enums import UnitType

    _skip_if_text_unavailable()

    node = NodeViz("u_body", unit_type=UnitType.SCRIPT, radius=0.5)

    # No meter when activation is zero
    node.set_activation_meter(0.0)
    assert node.meter is None

    # Partial arc when activation is non-zero
    node.set_activation_meter(0.5)
    assert isinstance(node.meter, Arc)

    # Angle approximately pi (2*pi*0.5)
    angle = getattr(node.meter, "angle", None)
    assert angle is not None
    assert math.isclose(angle, math.pi, rel_tol=1e-5, abs_tol=1e-5)


def test_edge_arrow_type_selection():
    _ensure_scripts_on_path()
    from manim import Arrow, DashedLine
    from manim_common import (  # Import added to path by _ensure_scripts_on_path()
        NodeViz,
        edge_arrow,
    )

    from recon_core.enums import UnitType

    _skip_if_text_unavailable()

    a = NodeViz("u_a", unit_type=UnitType.SCRIPT).move_to((-1, 0, 0))
    b = NodeViz("u_b", unit_type=UnitType.SCRIPT).move_to((1, 0, 0))

    solid = edge_arrow(a, b, dashed=False)
    dashed = edge_arrow(a, b, dashed=True)

    assert isinstance(solid, Arrow)
    assert isinstance(dashed, DashedLine)


def test_base_scene_helpers_without_init():
    _ensure_scripts_on_path()
    from manim import WHITE, AnimationGroup
    from manim_common import (  # Import added to path by _ensure_scripts_on_path()
        BaseReconScene,
        NodeViz,
    )

    from recon_core.enums import UnitType

    _skip_if_text_unavailable()

    # Create instance without running Scene.__init__ (pure helper usage)
    scene = BaseReconScene.__new__(BaseReconScene)

    src = NodeViz("u_src", unit_type=UnitType.SCRIPT).move_to((0, 0, 0))
    dst = NodeViz("u_dst", unit_type=UnitType.SCRIPT).move_to((2, 0, 0))

    anim = scene.animate_message_between_nodes(
        src, dst, "MSG", color=WHITE, duration=0.5
    )
    assert isinstance(anim, AnimationGroup)

    # Highlight shape matches node type
    square_node = NodeViz("t_vert", unit_type=UnitType.TERMINAL)
    circle_node = NodeViz("u_root", unit_type=UnitType.SCRIPT)
    sq = scene.create_highlight_shape(square_node, color=WHITE)
    cr = scene.create_highlight_shape(circle_node, color=WHITE)

    # Type names are stable across manim versions
    assert sq.__class__.__name__ == "Square"
    assert cr.__class__.__name__ == "Circle"
