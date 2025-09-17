# Recon Activation Network - Manim Scene

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Render

```bash
manim -p -qh scenes/recon_activation_network.py ReconActivationNetwork
```

- `-p`: preview after render
- `-qh`: high quality, faster than production
- For production: `manim -p -qh scenes/recon_activation_network.py ReconActivationNetwork`

The output will be in `media/videos/scenes/1080p60/` by default.
