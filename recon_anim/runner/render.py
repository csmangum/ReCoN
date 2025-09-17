from __future__ import annotations

import argparse
import os
import sys
from typing import Type

from manim import config as manim_config

from recon_core.compiler import compile_from_file
from recon_core.engine import EngineConfig

from recon_anim.adapters.live import EngineStepper
from recon_anim.scenes.activation_graph import ActivationGraphScene


def render_scene(scene_cls: Type[ActivationGraphScene], yaml_graph_path: str, quality: str = "ql", preview: bool = True, time_scale: float = 1.0, seed: int = 0):
    g = compile_from_file(yaml_graph_path)
    stepper = EngineStepper(g, EngineConfig())

    # Configure manim
    if quality == "ql":
        manim_config.quality = "low_quality"
    elif quality == "qh":
        manim_config.quality = "high_quality"
    else:
        manim_config.quality = quality

    manim_config.frame_rate = int(30 * time_scale)

    # Instantiate scene and attach stepper
    scene = scene_cls()
    setattr(scene, "_engine_stepper", stepper)
    scene.render()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render ReCoN graph activation scene")
    parser.add_argument("--scene", default="ActivationGraphScene")
    parser.add_argument("--graph", required=True, help="YAML graph spec (scripts/*.yaml)")
    parser.add_argument("--quality", default="ql", help="manim quality: ql/qh")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--time-scale", type=float, default=1.0)
    args = parser.parse_args(argv)

    scene_map = {
        "ActivationGraphScene": ActivationGraphScene,
    }

    scene_cls = scene_map.get(args.scene)
    if scene_cls is None:
        raise SystemExit(f"Unknown scene: {args.scene}")

    render_scene(scene_cls, args.graph, quality=args.quality, preview=args.preview, time_scale=args.time_scale)


if __name__ == "__main__":  # pragma: no cover
    main()

