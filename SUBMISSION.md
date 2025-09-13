# Request Confirmation Networks: Active Perception Implementation and Dynamic Visualization

## Abstract

Briefly summarize the problem, your approach, and key outcomes (2–4 sentences). Mention ReCoN implementation, interactive visualization, metrics, and main findings.

## Demo & Links

- Project repository: <ADD_REPO_URL>
- Demo video (2–3 min): <ADD_VIDEO_LINK>
- Interactive demo (Streamlit): see Quickstart below to run locally
- Original paper: [CoCoNIPS 2015 PDF](CoCoNIPS_2015_paper_6.pdf)

## Quickstart (Reproduce in ~2 minutes)

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt

# Interactive visualization
streamlit run viz/app_streamlit.py

# CLI example: run a YAML script and print a snapshot
python scripts/recon_cli.py scripts/house.yaml --steps 5 --deterministic --ret-feedback

# Alternative: dump to a file
python scripts/recon_cli.py scripts/house.yaml --steps 10 --out snapshot.json
```

## Problem Framing

- What representation problem was targeted and why ReCoN (active, selective perception) is a good fit.
- Constraints, scope, and success criteria you defined for the challenge.

## Approach Overview

- ReCoN recap: units (SCRIPT, TERMINAL), 8-state FSM, link types (SUB, SUR, POR, RET), and message alphabet (REQUEST, CONFIRM, WAIT, INHIBIT_*).
- How “request/confirm” enables active perception vs. passive classification.
- Design principles: modular core, configurable engine, YAML compiler, visualization.

## System Architecture

- Core modules and responsibilities:
  - `recon_core/engine.py` — update cycle, compact gate arithmetic, message passing
  - `recon_core/graph.py` — `Unit`, `Edge`, `Graph` data structures
  - `recon_core/enums.py` — states, messages, link types, unit types
  - `recon_core/compiler.py` — YAML → graph (hierarchy via SUB/SUR; sequence via POR/RET)
  - `recon_core/metrics.py` — runtime stats and convenience helpers
  - `perception/dataset.py` — synthetic scenes (house, barn, occlusion, variations)
  - `perception/terminals.py` — terminal features (filters, SIFT-like, autoencoder)
  - `viz/app_streamlit.py` — interactive visualization
- Optional small diagram of topology and data flow (screenshot acceptable).

## Implementation Details

- Compact gate arithmetic (typical contributions):
  - SUB: TRUE/CONFIRMED → +1.0; FAILED → −1.0
  - SUR: REQUESTED/ACTIVE → +0.3; FAILED → −0.3
  - POR: CONFIRMED → +0.5; FAILED → −0.5
  - RET: CONFIRMED → +0.2; FAILED → −0.5
- State machine semantics and message handling across the four-phase step.
- Engine configuration knobs (confirmation ratio, deterministic order, feedback).
- Script compilation from `scripts/*.yaml` into graph structure.
- Key engineering choices and trade-offs.

## Dataset & Terminals

- Synthetic scenes from `perception/dataset.py` (size, noise, occlusion, variations).
- Terminal features from `perception/terminals.py`:
  - Basic filters, SIFT-like features, blob/geometric features, optional autoencoder.
- Mapping from features to terminal units used in the demo(s).

## Visualization & UX

- What the Streamlit app shows: graph states, message flow, scene overlays, step/run controls.
- How the visualization reveals causality (requests, confirmations, failures, sequencing).

## Experiments & Results

- Qualitative traces: step-by-step sequences showing active requests and confirmations.
- Quantitative metrics (via `recon_core/metrics.py` and engine stats):
  - `terminal_request_count` and per-terminal counts
  - `steps_to_first_true`, `steps_to_first_confirm`
  - Precision/recall example using `binary_precision_recall`
- Any ablations/toggles (e.g., with/without POR; different thresholds).

## Evaluation vs. CIMC Criteria

- Implemented a novel theoretical idea → faithful ReCoN with tests and docs.
- Translation between representation and UI → interactive visualization + CLI.
- Active perception exhibited → selective terminal requests, temporal sequencing.
- Bonus (if applicable) → learning utilities, format conversion, metrics suite.

## Limitations & Future Work

- Current constraints (noise robustness, scaling, learned structure), and practical limitations.
- Next steps: richer perception (CNNs), learning SUB/SUR/POR/RET, multi-modal, real-time.

## How to Run Tests

```bash
python -m pytest tests/ -v
# Lightweight runner if pytest unavailable
python run_tests.py
```

## Configuration Reference

Typical engine configuration usage:

```python
from recon_core.config import EngineConfig
from recon_core.engine import Engine

cfg = EngineConfig(
    sur_positive=0.25,
    por_positive=0.6,
    confirmation_ratio=0.7,
    deterministic_order=True,
    ret_feedback_enabled=True,
)
engine = Engine(graph, config=cfg)
```

- Env flags: `RECON_TRAIN_AE=1` enables autoencoder training for terminals.

## Submission Artifacts

- Source code, `requirements.txt`, and this `SUBMISSION.md`.
- Demo video link and a few screenshots from the Streamlit UI.
- Example YAML scripts in `scripts/` (e.g., `house.yaml`, `barn.yaml`).
- Reproducibility notes (Python version, OS; no GPU required).

## Acknowledgements

- "Request Confirmation Networks for Active Object Recognition" (CoCoNIPS 2015). See `CoCoNIPS_2015_paper_6.pdf`.
- Libraries and tools used (Streamlit, NumPy, etc.).

## Statement of Originality

This submission reflects my original work. External sources are properly attributed. All code and assets included are licensed or created by me for this challenge.

