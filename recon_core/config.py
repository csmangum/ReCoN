"""
Configuration objects for the ReCoN engine.

Exposes tunable parameters for gate strengths, thresholds, and behavioral
policies, enabling experiments without editing core logic.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class EngineConfig:
    """
    Configuration for `Engine` behavior and arithmetic.

    Gate strengths and thresholds default to the values used in the current
    implementation to preserve backward compatibility.
    """

    # Gate outputs (compact arithmetic)
    sub_positive: float = 1.0
    sub_negative: float = -1.0

    sur_positive: float = 0.3
    sur_negative: float = -0.3

    por_positive: float = 0.5
    por_negative: float = -0.5

    ret_positive: float = 0.2
    ret_negative: float = -0.5

    # Activation integration
    activation_gain: float = 0.8

    # State thresholds
    script_request_activation_threshold: float = 0.1
    terminal_failure_threshold: float = 0.1
    confirmation_ratio: float = 0.6

    # Processing policies
    deterministic_order: bool = True

    # Optional RET feedback policy in state updates (beyond arithmetic deltas)
    # When enabled, a FAILED successor can demote a CONFIRMED predecessor to ACTIVE
    # via RET links; a CONFIRMED successor can stabilize predecessor (no-op here).
    ret_feedback_enabled: bool = False

    # Optional per-link-type minimal source activation thresholds for propagation.
    # If set > 0, gate outputs for that link type are suppressed unless the source
    # unit's activation is at least the specified value. Defaults preserve behavior.
    sub_min_source_activation: float = 0.0
    sur_min_source_activation: float = 0.0
    por_min_source_activation: float = 0.0
    ret_min_source_activation: float = 0.0
