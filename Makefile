SHELL := /usr/bin/bash

.PHONY: help venv deps nb tests demo render render-fast clean

help:
	@echo "Targets:"
	@echo "  venv   - create virtualenv (.venv)"
	@echo "  deps   - install Python dependencies"
	@echo "  nb     - generate key-features notebook"
	@echo "  tests  - run full pytest suite"
	@echo "  demo   - run Streamlit demo"
	@echo "  render - render Manim video (1080p60 production)"
	@echo "  render-fast - quick Manim preview (low quality)"
	@echo "  clean  - remove caches and build artifacts"

venv:
	python3 -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip

deps: venv
	. .venv/bin/activate && pip install -r requirements.txt
	. .venv/bin/activate && pip install pytest nbformat

nb:
	. .venv/bin/activate && python scripts/create_key_features_notebook.py

tests:
	. .venv/bin/activate && python -m pytest -q tests

demo:
	. .venv/bin/activate && streamlit run viz/app_streamlit.py

render: deps
	. .venv/bin/activate && \
	manim -qp --fps 60 -r 1920,1080 \
	  --media_dir output/media \
	  -o recon_house_walkthrough \
	  scripts/manim_recon_house.py HouseWalkthrough

render-fast: deps
	. .venv/bin/activate && \
	manim -ql --fps 30 \
	  --media_dir output/media \
	  -o recon_house_walkthrough_preview \
	  scripts/manim_recon_house.py HouseWalkthrough

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
