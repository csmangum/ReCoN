import nbformat as nbf


def main():
	nb = nbf.v4.new_notebook()

	intro = nbf.v4.new_markdown_cell(
		"""# ReCoN Key Features — Interactive Proof Notebook

This notebook demonstrates the implemented key features:

- Active Perception (SUR requests to terminals)
- Hierarchical Recognition (SUB evidence confirms parents)
- Temporal Sequencing (POR ordering)
- Metrics accessors for runtime behavior
- Extensibility via YAML compiler

Run top-to-bottom; executes quickly on CPU."""
	)

	setup = nbf.v4.new_code_cell(
		"""# Setup
import os
import numpy as np
from recon_core.enums import UnitType, LinkType, State, Message
from recon_core.graph import Graph, Unit, Edge
from recon_core.engine import Engine
from recon_core.config import EngineConfig
from recon_core.metrics import (
    steps_to_first_true,
    steps_to_first_confirm,
    total_terminal_requests,
    terminal_request_counts_by_id,
)
from recon_core import compile_from_file
from perception.dataset import make_house_scene
from perception.terminals import (
    terminals_from_image,
    advanced_terminals_from_image,
)

try:
    import matplotlib.pyplot as plt
    %matplotlib inline
except Exception:
    plt = None

np.random.seed(0)
print("Environment ready.")"""
	)

	scene = nbf.v4.new_code_cell(
		"""# 1) Generate a scene and extract features
img = make_house_scene(size=64, noise=0.05)
basic_feats = terminals_from_image(img)
adv_feats = advanced_terminals_from_image(img)

print({k: round(v, 3) for k, v in basic_feats.items()})
if plt is not None:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(img, cmap="gray")
    ax.set_title("Synthetic House Scene")
    ax.axis("off")
    plt.show()"""
	)

	house_run = nbf.v4.new_code_cell(
		"""# 2) Compile house graph, seed terminals, and run engine
house_yaml = "/workspace/scripts/house.yaml"

g = compile_from_file(house_yaml)
engine = Engine(g, config=EngineConfig(deterministic_order=True, ret_feedback_enabled=True))

# Seed terminal activations from the scene features
for tid in ["t_mean", "t_vert", "t_horz"]:
    if tid in g.units:
        g.units[tid].a = float(basic_feats.get(tid, 0.0))
        g.units[tid].state = State.REQUESTED if g.units[tid].a > 0.1 else State.INACTIVE

# Activate root script to initiate requests
root_id = [uid for uid in g.units if uid.startswith("u_house")][0]
g.units[root_id].a = 1.0
g.units[root_id].state = State.ACTIVE

snap = engine.step(10)
print(f"t={snap['t']}, root={root_id}, state={snap['units'][root_id]['state']}")

# Show a compact unit state summary
summary = {uid: (u["state"], round(u["a"], 2)) for uid, u in snap["units"].items()}
summary"""
	)

	hier = nbf.v4.new_code_cell(
		"""# 3) Prove hierarchical recognition (root confirms)
root_state = engine.snapshot()['units'][root_id]['state']
print("Root:", root_id, "State:", root_state)
assert root_state == 'CONFIRMED', "Root script should confirm given seeded evidence"

# Show child states
for cid in ['u_roof','u_body','u_door']:
    if cid in g.units:
        print(cid, g.units[cid].state.name, round(g.units[cid].a, 2))"""
	)

	sur = nbf.v4.new_code_cell(
		"""# 4) Prove active perception (SUR requests)
engine.reset()
# Re-energize root
root = [uid for uid in g.units if uid.startswith('u_house')][0]
g.units[root].a = 1.0
g.units[root].state = State.ACTIVE

snap = engine.step(2)
req_total = total_terminal_requests(engine)
req_by_id = terminal_request_counts_by_id(engine)
print("Total terminal SUR requests:", req_total)
print("By terminal:", req_by_id)
assert req_total >= 1 and all(v >= 1 for v in req_by_id.values()), "Expected top-down requests to terminals"""
	)

	por = nbf.v4.new_code_cell(
		"""# 5) Prove temporal sequencing (POR ordering)
engine.reset()
for uid in g.units:
    g.units[uid].a = 0.0
    g.units[uid].inbox.clear()
    g.units[uid].outbox.clear()

# Seed minimal terminal evidence again
for tid in ["t_mean", "t_vert", "t_horz"]:
    if tid in g.units:
        g.units[tid].a = float(basic_feats.get(tid, 0.0))
        g.units[tid].state = State.REQUESTED if g.units[tid].a > 0.1 else State.INACTIVE

# Root active
root = [uid for uid in g.units if uid.startswith('u_house')][0]
g.units[root].a = 1.0
g.units[root].state = State.ACTIVE

engine.step(10)
ftc = engine.stats.get('first_confirm_step', {})
print("First confirm steps:", ftc)
order = [ftc.get(u, 9999) for u in ['u_roof','u_body','u_door'] if u in g.units]
print("POR order roof->body->door steps:", order)
assert order == sorted(order), "Expected POR ordering: roof before body before door"""
	)

	barn = nbf.v4.new_code_cell(
		"""# 6) Extensibility: compile an alternate script (barn) and run
barn_yaml = "/workspace/scripts/barn.yaml"

g2 = compile_from_file(barn_yaml)
engine2 = Engine(g2, config=EngineConfig(deterministic_order=True))

# Seed same features (not tailored) just to exercise the topology
for tid in ["t_mean", "t_vert", "t_horz"]:
    if tid in g2.units:
        g2.units[tid].a = float(basic_feats.get(tid, 0.0))
        g2.units[tid].state = State.REQUESTED if g2.units[tid].a > 0.1 else State.INACTIVE

root2 = [uid for uid in g2.units if uid.startswith("u_barn")][0]
g2.units[root2].a = 1.0
g2.units[root2].state = State.ACTIVE

snap2 = engine2.step(8)
print("Barn root:", root2, "state:", snap2['units'][root2]['state'])
{uid: (u['state'], round(u['a'], 2)) for uid, u in snap2['units'].items()}"""
	)

	nb.cells = [intro, setup, scene, house_run, hier, sur, por, barn]

	# Basic metadata for kernel
	nb.metadata["kernelspec"] = {
		"display_name": "Python 3",
		"language": "python",
		"name": "python3",
	}
	nb.metadata["language_info"] = {"name": "python", "version": "3"}

	with open("/workspace/ReCoN_Key_Features_Demo.ipynb", "w", encoding="utf-8") as f:
		nbf.write(nb, f)

	print("Notebook written to /workspace/ReCoN_Key_Features_Demo.ipynb")


if __name__ == "__main__":
	main()

