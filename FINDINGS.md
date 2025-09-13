# ReCoN Recon, Fidelity Audit, and Recommendations

## Network Understanding (from recon/docs)
- **Units**: SCRIPT and TERMINAL; 8-state FSM: INACTIVE, REQUESTED, WAITING, ACTIVE, TRUE, CONFIRMED, FAILED, SUPPRESSED.
- **Links**: SUB (child→parent evidence), SUR (parent→child requests), POR (temporal precedence), RET (temporal feedback).
- **Messages**: REQUEST, CONFIRM, WAIT, INHIBIT_REQUEST, INHIBIT_CONFIRM.
- **Cycle**: Propagation (compact gate arithmetic) → State update (messages + soft activation) → Message delivery.
- **Demo topology**: House recognition (`u_root`→`u_roof/u_body/u_door`; terminals `t_horz/t_vert/t_mean`), POR sequence `roof→body→door`.

## Implementation Audit
- **Core types** (`recon_core/enums.py`): All enums align with docs (UnitType, LinkType, State, Message).
- **Graph model** (`recon_core/graph.py`): `Unit`, `Edge`, and `Graph` with helpers `sub_children`, `sur_children`, `por_successors` match spec.
- **Engine** (`recon_core/engine.py`):
  - Message handling supports all 5 messages with expected effects.
  - Gate functions implement compact arithmetic consistent with docs:
    - SUB: TRUE/CONFIRMED→+1.0; FAILED→-1.0
    - SUR: REQUESTED/ACTIVE→+0.3; FAILED→-0.3
    - POR: CONFIRMED→+0.5; FAILED→-0.5
    - RET: CONFIRMED→+0.2; FAILED→-0.5
  - Activation update uses soft integration and clamping.
  - Script logic: single-episode SUR requests with metrics; 60% child TRUE/CONFIRMED threshold; fail-fast on any FAILED child; POR requests successors on CONFIRMED.
  - Terminal logic: REQUESTED/threshold→TRUE; TRUE with a<0.1→FAILED; SUB emits CONFIRM/INHIBIT_CONFIRM appropriately.
- **Compiler** (`recon_core/compiler.py`): YAML→Graph; creates SUB/SUR edges; POR sequence from `sequence` entries; auto-creates terminals.
- **Metrics** (`recon_core/metrics.py`): Counters and timing accessors align with docs; used by engine `stats`.
- **Perception** (`perception/*.py`): Synthetic scenes and feature terminals; optional autoencoder with on-demand training.
- **Visualization** (`viz/app_streamlit.py`): Streamlit interface wiring to `Engine`, graph, and scenes.

## Tests and Validation
- Ran lightweight runner `run_tests.py`: 10/10 tests passed, covering engine, graph, messages, learning, compiler, metrics.
- Test modules (`tests/*.py`) further validate gate arithmetic, state machines, sequencing, and micro-graphs. Structure matches the documentation claims.

## Fidelity Verdict
- The codebase is an accurate and faithful implementation of the documented ReCoN specification and aligns with the cited CoCoNIPS 2015 model (as summarized in repo docs). Core behaviors—message semantics, gate arithmetic, sequencing, confirmation thresholding, and failure propagation—are implemented and validated.

## Recommendations
- **Parametrize thresholds/gains**: Expose gate constants (e.g., SUR=0.3, POR=0.5) and confirmation ratio (0.6) via Engine config.
- **RET usage in state updates**: Consider explicit handling of RET feedback in state transitions beyond arithmetic deltas (e.g., predecessor waiting/unblocking policies).
- **Deterministic ordering**: Ensure stable unit iteration order in `Engine._update_states` by sorting IDs to avoid nondeterminism across Python versions.
- **Learning extensions**: Generalize `online_sur_update` to learn SUB weights and POR/RET strengths based on task outcomes.
- **Perception dependency slimming**: Lazy-import heavy libs (matplotlib) and gate autoencoder training under a config flag for lighter environments.
- **CLI/Config**: Add a small CLI to run compiled scripts with steps/seed control and to dump snapshots/metrics as JSON.
- **CI tests**: Add a GitHub Action to run `run_tests.py` (and pytest where available) to prevent regressions.