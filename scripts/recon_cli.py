#!/usr/bin/env python3
"""
ReCoN CLI

Usage modes:
- Default run: compile YAML, step engine, print snapshot summary or write JSON
- Validation: check graph integrity and rules, print summary
- Stats: compute validation + performance stats
- Export: write GraphML for external tools
- Utility: list sample scenes, show version, dry-run compile only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

from recon_core import Engine  # type: ignore
from recon_core.config import EngineConfig  # type: ignore
from recon_core.compiler import compile_from_file  # type: ignore


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ReCoN from YAML and dump snapshot/metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Utilities / meta
    p.add_argument("--version", action="store_true", help="Print version and exit")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)")
    p.add_argument("--list-scenes", action="store_true", help="List bundled sample YAML scenes and exit")

    # Primary input
    p.add_argument("yaml", nargs="?", help="Path to YAML script (e.g., scripts/house.yaml)")

    # Execution
    p.add_argument("--steps", type=int, default=5, help="Number of steps to run")
    p.add_argument("--dry-run", action="store_true", help="Compile only; do not step the engine")
    p.add_argument("--out", type=str, default="", help="Optional output JSON file path")

    # Engine config overrides
    p.add_argument("--sur", type=float, default=None, help="SUR positive gate value")
    p.add_argument("--por", type=float, default=None, help="POR positive gate value")
    p.add_argument("--ret", type=float, default=None, help="RET positive gate value")
    p.add_argument("--sub", type=float, default=None, help="SUB positive gate value")
    p.add_argument("--confirm-ratio", type=float, default=None, help="Script confirmation ratio")
    p.add_argument("--deterministic", action="store_true", help="Force deterministic processing order")
    p.add_argument("--ret-feedback", action="store_true", help="Enable RET-driven feedback in state updates")

    # Analysis / export
    p.add_argument("--validate", action="store_true", help="Run comprehensive graph validation")
    p.add_argument("--strict-activation", action="store_true", help="Treat activation/threshold bounds as errors during validation")
    p.add_argument("--stats", action="store_true", help="Print graph statistics and health score")
    p.add_argument("--export-graphml", type=str, default="", help="Export compiled graph to GraphML at given path")

    return p.parse_args(argv)


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


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def find_sample_scenes() -> List[str]:
    # Search relative to repo root and this script location
    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    candidates = []
    for base in [repo_root, here.parent]:
        candidates.extend(sorted(glob(str(base / "*.yaml"))))
        candidates.extend(sorted(glob(str(base / "scripts" / "*.yaml"))))
    # Deduplicate while preserving order
    seen = set()
    result = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def main(argv: List[str] | None = None) -> int:
    try:
        from recon_core import __version__ as recon_version  # type: ignore
    except Exception:
        recon_version = "unknown"

    args = parse_args(argv)
    setup_logging(args.verbose)

    if args.version:
        print(recon_version)
        return 0

    if args.list_scenes:
        scenes = find_sample_scenes()
        if not scenes:
            print("[]")
            return 0
        print(json.dumps(scenes, indent=2))
        return 0

    if not args.yaml:
        print("error: missing YAML path (try --list-scenes)", file=sys.stderr)
        return 2

    cfg = build_config(args)

    logging.info("Compiling graph from %s", args.yaml)
    g = compile_from_file(args.yaml)

    # Optional validation
    if args.validate or args.stats:
        results = g.validate_all(strict_activation=args.strict_activation)
        summary = g.get_validation_summary(results)
        logging.info("Validation issues: %d (errors=%d warnings=%d)", summary["total_issues"], summary["errors"], summary["warnings"])
        if args.validate and not args.stats:
            print(json.dumps({"summary": summary, "results": results}, indent=2))
            # Non-zero exit on errors
            return 1 if summary["errors"] > 0 else 0

    # Optional stats
    if args.stats:
        stats = g.get_graph_statistics()
        print(json.dumps(stats, indent=2))
        # Do not step engine if stats requested without explicit run intent and dry-run
        if args.dry_run:
            return 0

    # Optional export
    if args.export_graphml:
        out_path = args.export_graphml
        logging.info("Exporting GraphML to %s", out_path)
        g.export_graphml(out_path)

    if args.dry_run:
        # Provide a minimal graph summary
        minimal = {
            "units": len(g.units),
            "edges": sum(len(e) for e in g.out_edges.values()),
        }
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(minimal, f, indent=2)
        else:
            print(json.dumps(minimal, indent=2))
        return 0

    # Run the engine
    engine = Engine(g, config=cfg)
    snap = engine.step(args.steps)

    # Include a compact summary in addition to raw snapshot
    summary: Dict[str, Any] = {
        "t": snap["t"],
        "units": {
            uid: {"state": u["state"], "a": u["a"], "kind": u["kind"]}
            for uid, u in snap["units"].items()
        },
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

