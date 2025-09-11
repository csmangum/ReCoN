# ReCoN Demo — Request Confirmation Networks

A minimal, faithful **Request Confirmation Network** (ReCoN) implementation with a dynamic Streamlit visualization and a toy perception stack for *active parts-of confirmation* in synthetic 2D scenes (e.g., a “house” made of roof/body/door).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run viz/app_streamlit.py
```

Then click **Generate Scene** and **Run** to watch requests/confirmations flow.

## Repo layout

```
recon_core/
  enums.py        # state/message/link enums
  graph.py        # Unit, Edge, Graph model
  engine.py       # propagation + state update rules
  learn.py        # (optional) tiny learning helpers for sur weights
perception/
  dataset.py      # synthetic 2D scenes (64x64) with simple shapes
  terminals.py    # terminal feature detectors (filters/AE stub)
scripts/
  house.yaml      # script → recon graph compiler input
viz/
  app_streamlit.py    # interactive visualization
tests/
  test_engine.py      # smoke tests for engine transitions
```

## Design sketch

- **Nodes:** `Unit(type=script|terminal, state, a)` with real-valued activation `a` and a finite **state** (`inactive, requested, waiting, active, true, confirmed, failed, suppressed`).
- **Links:** `Edge(u→v, type=sub|sur|por|ret, w)` with typed propagation.
- **Messages:** explicit message passing system with REQUEST, CONFIRM, INHIBIT_REQUEST, INHIBIT_CONFIRM, and WAIT messages; units have inbox/outbox queues for asynchronous communication.
- **Update:** discrete steps: (1) compact arithmetic propagation via per-gate functions, (2) process incoming messages, (3) apply node rules to update states and send messages, (4) deliver messages to recipients.
- **Visualization:** graph state coloring + animated stepper; scene pane shows detected terminals and scan “requests”.

## Notes

- This is a **clean-room skeleton**. It captures the *behavioral essence* of ReCoN’s typed propagation and stateful scripts and is structured for clarity + extension (learning, format conversion, AgentFarm hook).
- Extend `engine.py` to refine gating and inhibitory interactions; extend `perception/terminals.py` with stronger features; add more scripts in `scripts/`.
