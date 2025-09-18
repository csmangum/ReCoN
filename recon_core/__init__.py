"""
ReCoN Core Package.

This package contains the fundamental implementation of the Request Confirmation
Network (ReCoN) algorithm, including:

- Core data structures (Graph, Unit, Edge)
- State machines and message passing (Engine)
- Network topology definitions (enums)
- Learning utilities for weight adaptation

The ReCoN algorithm implements hierarchical, message-passing networks that can
learn to recognize complex patterns through coordinated activation dynamics.
"""

# ReCoN Core Package

__version__ = "0.1.0"

from .enums import UnitType, LinkType, State, Message
from .graph import Graph, Unit, Edge
from .engine import Engine

# Day 4: expose compiler utilities
try:
	from .compiler import compile_from_yaml, compile_from_file, compile_from_dict
except ImportError:  # pragma: no cover - optional import safety
	# Allow core to be imported even if optional deps (pyyaml) missing
	compile_from_yaml = None
	compile_from_file = None
	compile_from_dict = None

# expose metrics utilities
try:
	from .metrics import (
		binary_precision_recall,
		steps_to_first_confirm,
		steps_to_first_true,
		total_terminal_requests,
		terminal_request_counts_by_id,
	)
except ImportError:  # pragma: no cover
	# Metrics are optional; allow import without failing
	binary_precision_recall = None
	steps_to_first_confirm = None
	steps_to_first_true = None
	total_terminal_requests = None
	terminal_request_counts_by_id = None
