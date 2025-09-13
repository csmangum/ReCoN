# Request Confirmation Networks: Active Perception Implementation and Dynamic Visualization

## Abstract

This project implements a faithful, practical Request Confirmation Network (ReCoN) based on the CoCoNIPS 2015 paper. ReCoN is modeled as a spreading-activation graph in which scripted units actively request evidence from terminal units and confirm hypotheses through hierarchical composition and temporal sequencing. The implementation includes a modular Python core, a YAML→graph compiler for authoring object scripts, a lightweight perception pipeline for synthetic scenes (houses, barns, occlusions), and an interactive Streamlit visualization. The result demonstrates end‑to‑end active perception: selective, top‑down computation with interpretable control flow and a clear bridge from representation to user interface.

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

- Representation problem: recognize composite objects by actively confirming parts and relations rather than passively classifying pixels. ReCoN’s message-driven scripts express “what to look for next” and “when to stop,” which is central to active perception.
- Scope: focus on controlled 2D synthetic scenes (houses, barns, occlusions) to clearly demonstrate hierarchical parts (AND/OR structure) and ordered checks.
- Success criteria: a top-level hypothesis (e.g., “house”) drives selective SUR requests to relevant terminals, confirms parts in sequence, and yields a readable causal narrative in the UI.

## Approach Overview

- ReCoN recap: SCRIPT and TERMINAL units are tiny finite-state machines (INACTIVE, REQUESTED, WAITING, ACTIVE, TRUE, CONFIRMED, FAILED, SUPPRESSED) connected by typed links—SUB (evidence up), SUR (requests down), POR (temporal precedence), RET (temporal feedback)—and coordinate via REQUEST/CONFIRM/WAIT/INHIBIT messages.
- Active perception: scripts issue SUR requests only when needed; evidence flows upward via SUB; POR/RET impose ordering so predecessors unlock successors and failures feed back.
- Design: modular separation (graph, engine, compiler, perception, visualization), deterministic stepping for reproducibility, and a minimal YAML schema to author hierarchies and sequences compiled into graphs.

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

End-to-end flow: a YAML script (e.g., `scripts/house.yaml`) is compiled into a `Graph` of `Unit`s and `Edge`s. The `Engine` advances in discrete steps—propagating activation by link type, processing messages, updating states with soft activation dynamics, and delivering messages asynchronously. The Streamlit app renders unit states and messages as the engine steps through a scene, exposing the causal chain of requests and confirmations.

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
 
Four-phase step: (1) propagate activation deltas per gate, (2) process messages and update unit states with soft integration and clamping, (3) deliver messages from outboxes to inboxes, (4) process newly delivered messages to capture within-step effects.

State semantics: terminals become TRUE when requested and evidence exceeds threshold; scripts confirm when a sufficient fraction of children are TRUE/CONFIRMED and fail-fast on decisive negative evidence; POR unlocks successors upon confirmation, RET feeds completion/failure backward.

Configuration: confirmation ratio, deterministic iteration order, and temporal feedback are toggled via `EngineConfig` for predictable traces and exploration.

Compiler: SUB/SUR encode part–whole; POR/RET encode ordered checks; OR/AND relations are expressed structurally via multiple children and weights.

Trade-offs: lightweight perception for responsiveness; deterministic stepping for pedagogy over raw performance; emphasis on readability and testability.

## Dataset & Terminals

- Synthetic scenes from `perception/dataset.py` (size, noise, occlusion, variations).
- Terminal features from `perception/terminals.py`:
  - Basic filters, SIFT-like features, blob/geometric features, optional autoencoder.
- Mapping from features to terminal units used in the demo(s).
 
Rationale: synthetic glyph-like scenes make object structure explicit (e.g., roof above body; door inside body), highlighting ReCoN’s strengths in representing part relations and temporal checks. Terminals provide simple, explainable feature activations; an optional denoising autoencoder can supply compact learned features when enabled.

## Visualization & UX

- What the Streamlit app shows: graph states, message flow, scene overlays, step/run controls.
- How the visualization reveals causality (requests, confirmations, failures, sequencing).
 
The UI presents the network graph with nodes colored by state and animated message edges, alongside the current scene and overlays for detected terminals. Controls allow single-step, run/pause, and reset. This pairing makes it easy to see which requests were issued, which evidence arrived, and why a script confirmed or failed.

## Experiments & Results

- Qualitative traces: step-by-step sequences on house and barn scenes showing selective SUR requests, confirmations via SUB, and POR-driven sequencing (e.g., roof → body → door).
- Stressors: occlusion and added noise to illustrate failure cases and how inhibition/ordering affect behavior.
- Toggles: with/without POR; different confirmation ratios; deterministic vs. nondeterministic ordering.

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

