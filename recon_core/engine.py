"""
Request Confirmation Network (ReCoN) Engine Implementation.

This module implements the core ReCoN algorithm for processing activation dynamics
and state transitions in hierarchical networks. The engine manages:

- Message passing between units via inbox/outbox queues
- Compact arithmetic propagation via per-gate functions
- State machine transitions for both SCRIPT and TERMINAL units
- Temporal sequencing through POR (predecessor/successor) links

Configuration: Engine behavior (gate strengths, thresholds, ordering, optional
RET-driven feedback) is tunable via `EngineConfig` in `recon_core.config`.

The ReCoN algorithm operates in discrete time steps with four phases:
1. Propagation: Calculate activation deltas using gate functions
2. State Update: Process messages and update unit states with soft integration
3. Message Delivery: Move messages from outboxes to inboxes
4. Second Message Processing: Process newly delivered messages in the same step
"""

from __future__ import annotations

from typing import Dict

from .config import EngineConfig
from .enums import LinkType, Message, State, UnitType
from .graph import Graph


class Engine:
    """
    Main engine for simulating Request Confirmation Network dynamics.

    The Engine class implements the core ReCoN algorithm, managing the simulation
    of network activation, message passing, and state transitions over discrete
    time steps. It coordinates between different types of units (SCRIPT and TERMINAL)
    and handles various link types (SUB, SUR, POR, RET).

    Attributes:
        g: The network graph containing units and edges
        t: Current simulation time step
    """

    def __init__(self, g: Graph, config: EngineConfig | None = None):
        """
        Initialize the ReCoN engine with a network graph.

        Args:
            g: Graph object containing the network topology (units and edges)
        """
        self.g = g
        self.t = 0
        self.config = config or EngineConfig()
        # Track which scripts have already sent their initial SUR requests
        self._sur_requested_parents = set()
        # Metrics/statistics for Day 6 analytics
        self.stats = {
            "terminal_request_count": 0,
            "terminal_request_counts_by_id": {},  # unit_id -> count
            "first_request_step": {},  # unit_id -> t when first became REQUESTED
            "first_active_step": {},  # unit_id -> t when first became ACTIVE
            "first_true_step": {},  # unit_id -> t when terminal first became TRUE
            "first_confirm_step": {},  # unit_id -> t when first became CONFIRMED
        }

    # ----- helpers -----
    def _get_units_iter(self):
        """Return iterable of units honoring deterministic ordering when enabled."""
        return (
            [self.g.units[uid] for uid in sorted(self.g.units)]
            if self.config.deterministic_order
            else list(self.g.units.values())
        )

    def _get_unit_ids(self):
        """Return list of unit IDs honoring deterministic ordering when enabled."""
        return (
            sorted(self.g.units)
            if self.config.deterministic_order
            else list(self.g.units.keys())
        )

    def reset(self):
        """
        Reset the network to its initial state.

        This method clears all activation levels, resets unit states to INACTIVE,
        and empties all message queues (inbox and outbox) for all units in the network.
        The simulation time is also reset to 0.
        """
        for u in self.g.units.values():
            u.state = State.INACTIVE
            u.a = 0.0
            u.inbox.clear()
            u.outbox.clear()
        self.t = 0
        self._sur_requested_parents.clear()
        # Reset metrics/stats
        self.stats = {
            "terminal_request_count": 0,
            "terminal_request_counts_by_id": {},
            "first_request_step": {},
            "first_active_step": {},
            "first_true_step": {},
            "first_confirm_step": {},
        }

    # ----- message passing -----
    def send_message(self, sender_id: str, receiver_id: str, message: Message):
        """
        Send a message from one unit to another.

        This method places a message in the receiver's inbox queue for later processing.
        Messages enable asynchronous communication between units in the network.

        Args:
            sender_id: ID of the unit sending the message
            receiver_id: ID of the unit receiving the message
            message: The message type to send (REQUEST, CONFIRM, etc.)
        """
        if receiver_id in self.g.units:
            self.g.units[receiver_id].inbox.append((sender_id, message))

    def _deliver_messages(self):
        """
        Deliver all pending messages from unit outboxes to recipient inboxes.

        This method processes the message delivery phase of the ReCoN algorithm,
        moving messages from sender outboxes to receiver inboxes. This enables
        the asynchronous message passing that coordinates network activity.
        """
        units_iter = self._get_units_iter()
        for u in units_iter:
            while u.outbox:
                receiver_id, message = u.outbox.pop(0)
                self.send_message(u.id, receiver_id, message)

    def _process_messages(self, unit_id: str):
        """
        Process all messages in a unit's inbox and update its state accordingly.

        This method implements the message processing logic for different message types:
        - REQUEST: Boosts activation and may change state to REQUESTED
        - CONFIRM: Increases activation level
        - WAIT: Changes state to WAITING
        - INHIBIT_REQUEST: Suppresses requests and reduces activation
        - INHIBIT_CONFIRM: Fails confirmed units and strongly reduces activation

        Args:
            unit_id: ID of the unit whose messages should be processed
        """
        u = self.g.units[unit_id]
        messages = u.inbox.copy()
        u.inbox.clear()

        for _, message in messages:
            if message == Message.REQUEST:
                if u.state == State.INACTIVE:
                    u.state = State.REQUESTED
                    u.a = max(u.a, 0.3)  # boost activation on request
            elif message == Message.CONFIRM:
                if u.state in (State.REQUESTED, State.ACTIVE):
                    u.a = min(u.a + 0.4, 1.0)  # boost activation on confirm
            elif message == Message.WAIT:
                if u.state in (State.REQUESTED, State.ACTIVE):
                    u.state = State.WAITING
            elif message == Message.INHIBIT_REQUEST:
                if u.state == State.REQUESTED:
                    u.state = State.SUPPRESSED
                    u.a = max(u.a - 0.3, 0.0)  # reduce activation
            elif message == Message.INHIBIT_CONFIRM:
                if u.state in (State.TRUE, State.CONFIRMED):
                    u.state = State.FAILED
                    u.a = max(u.a - 0.5, 0.0)  # strong inhibition

    # ----- compact arithmetic propagation (per-gate functions) -----
    def _propagate(self):
        """
        Implement compact arithmetic propagation via per-gate functions.

        This method calculates activation deltas for all units based on the ReCoN
        compact arithmetic propagation algorithm. Each edge type (SUB, SUR, POR, RET)
        has its own gate function that determines how activation flows between units.

        The propagation follows these principles:
        - SUB links: Evidence flows bottom-up from children to parents
        - SUR links: Requests flow top-down from parents to children
        - POR links: Temporal precedence enables sequential activation
        - RET links: Temporal feedback provides success/failure signals

        Returns:
            Dict[str, float]: Dictionary mapping unit IDs to activation deltas
        """
        # collect deltas by destination
        delta = {uid: 0.0 for uid in self.g.units}

        for src_id, edges in self.g.out_edges.items():
            src = self.g.units[src_id]
            for e in edges:
                # Compact per-gate functions as described in the paper
                gate_output = self._gate_function(src, e.type)
                if gate_output != 0.0:
                    delta[e.dst] += e.w * gate_output
        return delta

    def _gate_function(self, unit, link_type):
        """
        Compute the gate function output for a specific unit and link type.

        Gate functions implement the core logic of how activation propagates through
        different types of links in the ReCoN network. Each link type has different
        propagation rules based on the source unit's current state.

        Args:
            unit: Source unit whose activation is being propagated
            link_type: Type of link (SUB, SUR, POR, RET) determining propagation logic

        Returns:
            float: Gate output value (0.0 if no propagation, positive/negative otherwise)
        """
        state = unit.state
        a = getattr(unit, "a", 0.0)

        # Optional minimal activation thresholds per link type
        # If the source activation is below the configured minimum, suppress output
        if link_type == LinkType.SUB and a < self.config.sub_min_source_activation:
            return 0.0
        if link_type == LinkType.SUR and a < self.config.sur_min_source_activation:
            return 0.0
        if link_type == LinkType.POR and a < self.config.por_min_source_activation:
            return 0.0
        if link_type == LinkType.RET and a < self.config.ret_min_source_activation:
            return 0.0

        if link_type == LinkType.SUB:
            # SUB: child -> parent evidence propagation
            if state in (State.TRUE, State.CONFIRMED):
                return self.config.sub_positive  # positive evidence
            elif state == State.FAILED:
                return self.config.sub_negative  # negative evidence
            else:
                return 0.0

        elif link_type == LinkType.SUR:
            # SUR: parent -> child top-down request
            if state in (State.REQUESTED, State.ACTIVE):
                return self.config.sur_positive  # request signal
            elif state == State.FAILED:
                return self.config.sur_negative  # inhibition signal
            else:
                return 0.0

        elif link_type == LinkType.POR:
            # POR: predecessor -> successor temporal sequence
            if state == State.CONFIRMED:
                return self.config.por_positive  # enable successor
            elif state == State.FAILED:
                return self.config.por_negative  # inhibit successor
            else:
                return 0.0

        elif link_type == LinkType.RET:
            # RET: successor -> predecessor temporal feedback
            if state == State.FAILED:
                return self.config.ret_negative  # failure feedback
            elif state == State.CONFIRMED:
                return self.config.ret_positive  # success feedback
            else:
                return 0.0

        return 0.0

    def _update_states(self, delta: Dict[str, float]):
        """
        Update unit states and send messages based on current activation levels.

        This method implements the state machine logic for both SCRIPT and TERMINAL units:
        - TERMINAL units: Detect features and send confirmations when thresholds are met
        - SCRIPT units: Orchestrate children, request confirmations, and manage sequences

        The method processes messages first, then updates activations softly using the
        propagation deltas, and finally handles state transitions and message sending.

        Args:
            delta: Dictionary of activation deltas from propagation phase
        """
        # First, process incoming messages for all units
        unit_ids = self._get_unit_ids()
        for uid in unit_ids:
            self._process_messages(uid)

        # Update activations softly
        for uid in unit_ids:
            u = self.g.units[uid]
            u.a = max(0.0, min(1.0, u.a + self.config.activation_gain * delta[uid]))

        # State transitions and message sending
        # Process units in reverse order so children are processed before parents
        ordered_ids = unit_ids
        for uid in reversed(ordered_ids):
            u = self.g.units[uid]
            if u.kind == UnitType.TERMINAL:
                # Terminal state machine with message sending
                # Only process state transitions if no messages have changed the state
                if u.state == State.INACTIVE and u.a >= u.thresh:
                    u.state = State.TRUE
                    # Record first TRUE timing
                    if uid not in self.stats["first_true_step"]:
                        self.stats["first_true_step"][uid] = self.t
                    # Send CONFIRM to parent (via SUB links)
                    for edge in self.g.out_edges[uid]:
                        if edge.type == LinkType.SUB:
                            u.outbox.append((edge.dst, Message.CONFIRM))

                elif u.state == State.REQUESTED and u.a >= u.thresh:
                    u.state = State.TRUE
                    if uid not in self.stats["first_true_step"]:
                        self.stats["first_true_step"][uid] = self.t
                    # Send CONFIRM to parent (via SUB links)
                    for edge in self.g.out_edges[uid]:
                        if edge.type == LinkType.SUB:
                            u.outbox.append((edge.dst, Message.CONFIRM))

                elif (
                    u.state == State.TRUE
                    and u.a < self.config.terminal_failure_threshold
                ):
                    if u.state != State.FAILED:
                        u.state = State.FAILED
                        # Send INHIBIT_CONFIRM to parent
                        for edge in self.g.out_edges[uid]:
                            if edge.type == LinkType.SUB:
                                u.outbox.append((edge.dst, Message.INHIBIT_CONFIRM))

            else:  # SCRIPT
                # Script state machine with message sending
                if (
                    u.state == State.INACTIVE
                    and u.a > self.config.script_request_activation_threshold
                ):
                    u.state = State.REQUESTED
                    if uid not in self.stats["first_request_step"]:
                        self.stats["first_request_step"][uid] = self.t

                if u.state == State.REQUESTED:
                    u.state = State.ACTIVE
                    if uid not in self.stats["first_active_step"]:
                        self.stats["first_active_step"][uid] = self.t
                    # Children should have been requested when we became REQUESTED
                    # No need to request again

                # Ensure SUR requests are sent exactly once when a script is REQUESTED or ACTIVE
                if (
                    u.state in (State.REQUESTED, State.ACTIVE)
                    and uid not in self._sur_requested_parents
                ):
                    # Send REQUEST to children via SUR links
                    for child_id in self.g.sur_children(u.id):
                        u.outbox.append((child_id, Message.REQUEST))
                        # Count terminal queries (SUR to TERMINAL children)
                        child_unit = self.g.units.get(child_id)
                        if child_unit and child_unit.kind == UnitType.TERMINAL:
                            self.stats["terminal_request_count"] += 1
                            self.stats["terminal_request_counts_by_id"][child_id] = (
                                self.stats["terminal_request_counts_by_id"].get(
                                    child_id, 0
                                )
                                + 1
                            )
                    self._sur_requested_parents.add(uid)

                # Check if enough children are TRUE to confirm (for both REQUESTED->ACTIVE and existing ACTIVE/CONFIRMED)
                if u.state in (State.ACTIVE, State.CONFIRMED):
                    child_ids = self.g.sub_children(u.id)
                    trues = sum(
                        1
                        for c in child_ids
                        if self.g.units[c].state in (State.TRUE, State.CONFIRMED)
                    )
                    failed = any(
                        self.g.units[c].state == State.FAILED for c in child_ids
                    )
                    need = (
                        max(1, int(self.config.confirmation_ratio * len(child_ids)))
                        if child_ids
                        else 0
                    )

                    if (
                        child_ids
                        and trues >= need
                        and not failed
                        and u.state != State.CONFIRMED
                    ):
                        u.state = State.CONFIRMED
                        if uid not in self.stats["first_confirm_step"]:
                            self.stats["first_confirm_step"][uid] = self.t
                        # Send CONFIRM to parent (via SUB links)
                        for edge in self.g.out_edges[uid]:
                            if edge.type == LinkType.SUB:
                                u.outbox.append((edge.dst, Message.CONFIRM))
                    elif (
                        u.state == State.CONFIRMED
                        and child_ids
                        and (trues < need or failed)
                    ):
                        # If confirmed but no longer meets criteria, revert to ACTIVE
                        # Only check this for scripts that actually have children
                        u.state = State.ACTIVE

                # Handle POR succession for confirmed scripts
                if u.state == State.CONFIRMED:
                    # Send REQUEST to POR successors
                    for succ_id in self.g.por_successors(u.id):
                        u.outbox.append((succ_id, Message.REQUEST))

                # Optional RET feedback policy: a FAILED successor can demote a CONFIRMED predecessor
                if self.config.ret_feedback_enabled and u.state == State.CONFIRMED:
                    # Check incoming RET edges from successors
                    has_failed_successor = any(
                        e.type == LinkType.RET
                        and self.g.units.get(e.src)
                        and self.g.units[e.src].state == State.FAILED
                        for e in self.g.in_edges.get(uid, [])
                    )
                    if has_failed_successor:
                        u.state = State.ACTIVE

                # Handle inhibition from failed children - fail immediately if any child fails
                failed_children = [
                    c
                    for c in self.g.sub_children(u.id)
                    if self.g.units[c].state == State.FAILED
                ]
                if failed_children and u.state not in (State.FAILED,):
                    old_state = u.state
                    u.state = State.FAILED
                    # Propagate failure to parent only if we just failed
                    if old_state != State.FAILED:
                        for edge in self.g.out_edges[uid]:
                            if edge.type == LinkType.SUB:
                                u.outbox.append((edge.dst, Message.INHIBIT_CONFIRM))

    def step(self, n=1):
        """
        Advance the simulation by n time steps.

        Each time step consists of four phases:
        1. Propagation: Calculate activation deltas using gate functions
        2. State Update: Process messages and update unit states
        3. Message Delivery: Move messages from outboxes to inboxes
        4. Second Message Processing: Process newly delivered messages

        Args:
            n: Number of time steps to advance (default: 1)

        Returns:
            dict: Snapshot of the network state after stepping
        """
        for _ in range(n):
            delta = self._propagate()
            self._update_states(delta)
            self._deliver_messages()  # deliver messages after state updates
            # Process newly delivered messages in the same step
            for uid in self._get_unit_ids():
                self._process_messages(uid)
            self.t += 1
        return self.snapshot()

    def snapshot(self):
        """
        Create a snapshot of the current network state.

        This method captures the current state of all units including their
        activation levels, states, message queue sizes, and metadata. Used for
        visualization, debugging, and analysis.

        Returns:
            dict: Dictionary containing:
                - 't': Current simulation time
                - 'units': Dictionary of unit states with activation, state, etc.
        """
        return {
            "t": self.t,
            "units": {
                uid: {
                    "state": u.state.name,
                    "a": u.a,
                    "kind": u.kind.name if u.kind else "UNKNOWN",
                    "meta": u.meta,
                    "inbox_size": len(u.inbox),
                    "outbox_size": len(u.outbox),
                }
                for uid, u in self.g.units.items()
            },
            "stats": self.stats,
        }
