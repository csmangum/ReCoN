"""
Unit tests for Day 6 metrics utilities and engine instrumentation.
"""

import pytest

from recon_core.enums import UnitType, State, LinkType
from recon_core.graph import Graph, Unit, Edge
from recon_core.engine import Engine
from recon_core.metrics import (
    binary_precision_recall,
    steps_to_first_confirm,
    steps_to_first_true,
    total_terminal_requests,
    terminal_request_counts_by_id,
)


class TestBinaryMetrics:
    def test_precision_recall_basic(self):
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 1, 0, 0, 1]
        m = binary_precision_recall(y_true, y_pred)
        # tp=2 (idx 0,4), fp=1 (idx 1), fn=1 (idx 2), tn=1 (idx 3)
        assert pytest.approx(m["precision"], 1e-9) == 2/3
        assert pytest.approx(m["recall"], 1e-9) == 2/3
        assert m["tp"] == 2
        assert m["fp"] == 1
        assert m["tn"] == 1
        assert m["fn"] == 1

    def test_precision_recall_zero_divisions(self):
        # No positives predicted
        m1 = binary_precision_recall([1, 1, 0], [0, 0, 0])
        assert m1["precision"] == 0.0
        assert m1["recall"] == 0.0
        # No positives in truth
        m2 = binary_precision_recall([0, 0, 0], [1, 1, 1])
        assert m2["precision"] == 0.0  # tp=0
        assert m2["recall"] == 0.0     # tp+fn = 0 -> define as 0


class TestEngineMetrics:
    def build_simple_network(self):
        g = Graph()
        # Parent script requests terminal
        parent = Unit('parent', UnitType.SCRIPT)
        term = Unit('term', UnitType.TERMINAL, thresh=0.5)
        g.add_unit(parent)
        g.add_unit(term)
        g.add_edge(Edge('term', 'parent', LinkType.SUB))
        g.add_edge(Edge('parent', 'term', LinkType.SUR))
        return g, parent, term

    def test_terminal_request_counting(self):
        g, parent, term = self.build_simple_network()
        engine = Engine(g)
        # Activate parent to trigger SUR requests
        parent.a = 1.0
        parent.state = State.ACTIVE
        engine.step(1)
        # One SUR request to a terminal should have been counted
        assert total_terminal_requests(engine) >= 1
        counts = terminal_request_counts_by_id(engine)
        assert counts.get('term', 0) >= 1

    def test_steps_to_events(self):
        g, parent, term = self.build_simple_network()
        engine = Engine(g)
        # Preload terminal activation to exceed threshold
        term.a = 0.6
        # Activate parent
        parent.a = 1.0
        parent.state = State.ACTIVE
        # Run several steps to allow TRUE and CONFIRM
        engine.step(5)
        # Terminal TRUE timing
        t_true = steps_to_first_true(engine, 'term')
        assert t_true is not None
        # Parent CONFIRMED timing
        t_confirm = steps_to_first_confirm(engine, 'parent')
        assert t_confirm is not None

    def test_stats_reset_clears_metrics(self):
        g, parent, term = self.build_simple_network()
        engine = Engine(g)
        parent.a = 1.0
        parent.state = State.ACTIVE
        engine.step(2)
        # Ensure counters populated
        assert total_terminal_requests(engine) >= 1
        assert terminal_request_counts_by_id(engine).get('term', 0) >= 1
        # Reset and verify stats cleared
        engine.reset()
        assert total_terminal_requests(engine) == 0
        assert terminal_request_counts_by_id(engine) == {}
        assert engine.stats['first_true_step'] == {}
        assert engine.stats['first_confirm_step'] == {}

    def test_terminal_request_sent_once(self):
        g, parent, term = self.build_simple_network()
        engine = Engine(g)
        parent.a = 1.0
        parent.state = State.ACTIVE
        # Step multiple times; SUR requests should be sent once per parent
        engine.step(5)
        assert total_terminal_requests(engine) == 1
        assert terminal_request_counts_by_id(engine).get('term', 0) == 1

