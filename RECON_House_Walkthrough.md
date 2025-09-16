# ReCoN Synthetic House Walkthrough

This guide walks through the Request Confirmation Network (ReCoN) activation and messaging flow for the synthetic house example: how the graph is built from YAML, how the synthetic image seeds terminal activations, and how recognition proceeds step by step.

## 1) Graph from YAML

The house script (`scripts/house.yaml`) defines an object with three parts and a textual sequence:

```yaml
object: house
children:
  - id: roof
    type: AND
    parts: [t_horz]
  - id: body
    type: AND
    parts: [t_mean]
  - id: door
    type: OR
    parts: [t_vert, t_mean]
sequence:
  - scan_edges_left_to_right
  - verify_roof_above_body
  - verify_door_inside_body
```

The compiler (`recon_core/compiler.py`) builds:
- Scripts: `u_house` (root), `u_roof`, `u_body`, `u_door`
- Terminals: `t_horz`, `t_mean`, `t_vert`
- Links (weights default 1.0 unless noted):
  - SUB (evidence): terminals → their script; child scripts → root
  - SUR (requests): root → child scripts; child scripts → terminals
  - POR (sequence): `u_roof → u_body → u_door`

## 2) Image → Terminals

Synthetic image (`perception/dataset.py`): a rectangular body, triangular roof, and rectangular door with set intensities. Terminal extractors (`perception/terminals.py`) compute:
- `t_mean`: mean intensity
- `t_horz`: horizontal edge strength (scaled)
- `t_vert`: vertical edge strength (scaled)

These values initialize corresponding terminal unit activations.

## 3) Engine Dynamics (per step)

Each time step (`recon_core/engine.py`) runs:
1. Propagate deltas via compact gate functions (SUB/SUR/POR/RET)
2. Process messages and update states with soft integration
3. Deliver messages (outbox → inbox)
4. Process newly delivered messages again

Key gate behaviors (defaults from `EngineConfig`):
- SUB: TRUE/CONFIRMED → +1.0; FAILED → -1.0
- SUR: REQUESTED/ACTIVE → +0.3; FAILED → -0.3
- POR: CONFIRMED → +0.5; FAILED → -0.5
- Activation gain: 0.8; confirmation ratio: 0.6

## 4) Step-by-Step Recognition Flow

- Step 0 (init):
  - Set `u_house` ACTIVE with high activation to start top-down flow.
  - Terminals get activations from image features.

- Step 1 (requests fan out):
  - `u_house` emits SUR to `u_roof`, `u_body`, `u_door` → they become REQUESTED→ACTIVE.
  - Each child script emits one-time SUR requests to its terminals.
  - Terminals with activation ≥ threshold become TRUE and queue CONFIRM to their script parent.

- Step 2 (child confirms + POR):
  - SUB from TRUE terminals boosts `u_roof`, `u_body`, `u_door`.
  - Scripts confirm once enough children are TRUE (ratio ≥ 0.6):
    - `u_roof`: needs `t_horz` TRUE → CONFIRMED
    - `u_body`: needs `t_mean` TRUE → CONFIRMED
    - `u_door`: needs one of {`t_vert`, `t_mean`} TRUE → CONFIRMED
  - Confirmed scripts send CONFIRM up to `u_house` and REQUEST to POR successors (`u_roof → u_body`, `u_body → u_door`).

- Step 3 (root confirms):
  - SUB from confirmed child scripts boosts `u_house`.
  - With confirmation ratio 0.6, `u_house` becomes CONFIRMED.

- Later steps (stability/failure):
  - If terminals drop below failure threshold, scripts can demote/fail, sending inhibition upstream; optional RET feedback can demote confirmed predecessors.

## 5) What to Look For in Traces

- Initial SUR messages from `u_house` to child scripts
- SUR messages from child scripts to terminals
- Terminal CONFIRM messages to scripts when thresholds are met
- Script CONFIRM messages to parent and POR REQUESTs to successors
- Final `u_house` CONFIRMED when enough children confirm

## 6) Running a Live Demo

- Streamlit app (`viz/app_streamlit.py`) wires image features into terminals and animates messages/states.
- CLI (`scripts/recon_cli.py`) compiles YAML and steps the engine; to reproduce the full image-driven flow, compute terminals and assign to units before stepping (see the notebook described below).

---

This walkthrough is validated by the accompanying Jupyter notebook that generates a house image, computes terminal activations, steps the engine, and visualizes states/messages to confirm the described behavior.