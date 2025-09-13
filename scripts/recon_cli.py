#!/usr/bin/env python3
"""
Simple CLI to compile a YAML ReCoN script, run the engine for N steps,
and dump snapshots/metrics to stdout or a JSON file.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from recon_core import Engine
from recon_core.config import EngineConfig
from recon_core.compiler import compile_from_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ReCoN from YAML and dump snapshot/metrics")
    p.add_argument("yaml", help="Path to YAML script (e.g., scripts/house.yaml)")
    p.add_argument("--steps", type=int, default=5, help="Number of steps to run (default: 5)")
    p.add_argument("--out", type=str, default="", help="Optional output JSON file path")

    # Engine config overrides
    p.add_argument("--sur", type=float, default=None, help="SUR positive gate value (default 0.3)")
    p.add_argument("--por", type=float, default=None, help="POR positive gate value (default 0.5)")
    p.add_argument("--ret", type=float, default=None, help="RET positive gate value (default 0.2)")
    p.add_argument("--sub", type=float, default=None, help="SUB positive gate value (default 1.0)")
    p.add_argument("--confirm-ratio", type=float, default=None, help="Script confirmation ratio (default 0.6)")
    p.add_argument("--deterministic", action="store_true", help="Force deterministic processing order")
    p.add_argument("--ret-feedback", action="store_true", help="Enable RET-driven feedback in state updates")

    return p.parse_args()


def build_config(args: argparse.Namespace) -> EngineConfig:
    cfg = EngineConfig()
    if args.sur is not None:
        cfg.sur_positive = float(args.sur)
    if args.por is not None:
        cfg.por_positive = float(args.por)
    if args.ret is not None:
        cfg.ret_positive = float(args.ret)
    if args.sub is not None:
        cfg.sub_positive = float(args.sub)
    if args.confirm_ratio is not None:
        cfg.confirmation_ratio = float(args.confirm_ratio)
    if args.deterministic:
        cfg.deterministic_order = True
    cfg.ret_feedback_enabled = args.ret_feedback
    return cfg


def main() -> int:
    args = parse_args()
    cfg = build_config(args)

    g = compile_from_file(args.yaml)
    engine = Engine(g, config=cfg)

    snap = engine.step(args.steps)
    # Include a compact summary in addition to raw snapshot
    summary: Dict[str, Any] = {
        "t": snap["t"],
        "units": {uid: {"state": u["state"], "a": u["a"], "kind": u["kind"]} for uid, u in snap["units"].items()},
        "stats": snap.get("stats", {}),
    }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    else:
        print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

