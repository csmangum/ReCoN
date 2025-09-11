"""
Core enumerations for the Request Confirmation Network (ReCoN) system.

This module defines the fundamental types, states, and messages used throughout
the ReCoN implementation for representing network components and their interactions.
"""

from enum import Enum, auto


class UnitType(Enum):
    """
    Types of units in the ReCoN network.

    Units are the basic computational elements that can be either:
    - SCRIPT: Higher-level units that orchestrate and coordinate activities
    - TERMINAL: Leaf units that interface with the external world/perception
    """

    SCRIPT = auto()
    """Higher-level orchestration unit that coordinates child activities."""

    TERMINAL = auto()
    """Leaf unit that interfaces with sensory/perceptual inputs."""


class LinkType(Enum):
    """
    Types of directed links between units in the ReCoN network.

    Links define different types of relationships and information flow:
    - SUB: Evidence propagation from child to parent (bottom-up)
    - SUR: Request propagation from parent to child (top-down)
    - POR: Temporal precedence in sequential processes
    - RET: Temporal feedback from successor to predecessor
    """

    SUB = auto()
    """Child-to-parent link for evidence propagation (part-of relationships)."""

    SUR = auto()
    """Parent-to-child link for request propagation (top-down control)."""

    POR = auto()
    """Predecessor-to-successor link for temporal/sequential ordering."""

    RET = auto()
    """Successor-to-predecessor link for temporal feedback."""


class State(Enum):
    """
    Possible states of units in the ReCoN network.

    States represent the current status and activation level of each unit:
    - INACTIVE: Unit is dormant, no activity
    - REQUESTED: Unit has received a request for activation
    - WAITING: Unit is waiting for dependencies or external conditions
    - ACTIVE: Unit is actively processing/confirming
    - TRUE: Terminal unit has detected its feature/pattern
    - CONFIRMED: Script unit has sufficient confirmation from children
    - FAILED: Unit has failed validation or encountered error
    - SUPPRESSED: Unit has been inhibited by conflicting information
    """

    INACTIVE = auto()
    """Unit is dormant with no activation."""

    REQUESTED = auto()
    """Unit has been requested to activate."""

    WAITING = auto()
    """Unit is waiting for dependencies or conditions."""

    ACTIVE = auto()
    """Unit is actively processing."""

    TRUE = auto()
    """Terminal unit has detected its target feature."""

    CONFIRMED = auto()
    """Script unit has sufficient confirmation from children."""

    FAILED = auto()
    """Unit has failed or been invalidated."""

    SUPPRESSED = auto()
    """Unit has been suppressed by inhibitory signals."""


class Message(Enum):
    """
    Message types for inter-unit communication in ReCoN.

    Messages enable asynchronous communication between units:
    - REQUEST: Request a unit to activate
    - WAIT: Tell a unit to wait for further instructions
    - CONFIRM: Confirm successful completion/activation
    - INHIBIT_REQUEST: Block/suppress a request
    - INHIBIT_CONFIRM: Invalidate a confirmation
    """

    REQUEST = auto()
    """Request activation from another unit."""

    WAIT = auto()
    """Signal to wait for further processing."""

    CONFIRM = auto()
    """Confirm successful activation or completion."""

    INHIBIT_REQUEST = auto()
    """Suppress or block an activation request."""

    INHIBIT_CONFIRM = auto()
    """Invalidate or inhibit a confirmation."""
