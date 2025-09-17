"""
Unit tests for the generic online learning update.

These tests ensure `online_generic_update` identifies targets by link type and
updates edge weights toward the proper targets, with clamping to [0, 2].
"""

from recon_core.enums import UnitType, State, LinkType
from recon_core.graph import Graph, Unit, Edge
from recon_core.learn import online_generic_update


def _expected_weight(old_w: float, target: float, lr: float) -> float:
    return old_w + lr * (target - old_w)


class TestOnlineGenericUpdate:
    def _make_units(self):
        g = Graph()
        # Use scripts for generality; terminals appear where appropriate
        a = Unit('a', UnitType.SCRIPT, state=State.ACTIVE)
        b = Unit('b', UnitType.SCRIPT, state=State.REQUESTED)
        g.add_unit(a)
        g.add_unit(b)
        return g, a, b

    def test_sub_edge_targets(self):
        g, a, b = self._make_units()
        # SUB: child (src) TRUE/CONFIRMED -> target 1, FAILED -> target 0
        sub = Edge('a', 'b', LinkType.SUB, w=0.5)
        g.add_edge(sub)

        # src TRUE
        g.units['a'].state = State.TRUE
        online_generic_update(g, 'a', 'b', lr=0.2)
        assert abs(sub.w - _expected_weight(0.5, 1.0, 0.2)) < 1e-12

        # src FAILED
        g.units['a'].state = State.FAILED
        online_generic_update(g, 'a', 'b', lr=0.2)
        assert abs(sub.w - _expected_weight(0.6, 0.0, 0.2)) < 1e-12

    def test_sur_edge_targets(self):
        g, a, b = self._make_units()
        # SUR: target 1 if dst TRUE/CONFIRMED and src CONFIRMED, else 0 if dst FAILED
        sur = Edge('a', 'b', LinkType.SUR, w=0.5)
        g.add_edge(sur)

        # dst TRUE but src not CONFIRMED -> no change
        g.units['a'].state = State.ACTIVE
        g.units['b'].state = State.TRUE
        online_generic_update(g, 'a', 'b', lr=0.3)
        assert sur.w == 0.5

        # dst TRUE and src CONFIRMED -> target 1
        g.units['a'].state = State.CONFIRMED
        online_generic_update(g, 'a', 'b', lr=0.3)
        assert abs(sur.w - _expected_weight(0.5, 1.0, 0.3)) < 1e-12

        # dst FAILED -> target 0
        g.units['b'].state = State.FAILED
        online_generic_update(g, 'a', 'b', lr=0.3)
        # previous w after first update: 0.65 -> now toward 0
        assert abs(sur.w - _expected_weight(0.65, 0.0, 0.3)) < 1e-12

    def test_por_edge_targets(self):
        g, a, b = self._make_units()
        por = Edge('a', 'b', LinkType.POR, w=1.2)
        g.add_edge(por)

        # src CONFIRMED and dst progressed (REQUESTED/ACTIVE/CONFIRMED) -> 1
        g.units['a'].state = State.CONFIRMED
        g.units['b'].state = State.REQUESTED
        online_generic_update(g, 'a', 'b', lr=0.5)
        assert abs(por.w - _expected_weight(1.2, 1.0, 0.5)) < 1e-12

        # dst FAILED -> 0
        g.units['b'].state = State.FAILED
        online_generic_update(g, 'a', 'b', lr=0.5)
        # previous w after first update: 1.1 -> now toward 0
        assert abs(por.w - _expected_weight(1.1, 0.0, 0.5)) < 1e-12

    def test_ret_edge_targets(self):
        g, a, b = self._make_units()
        ret = Edge('a', 'b', LinkType.RET, w=0.3)
        g.add_edge(ret)

        # dst CONFIRMED and src CONFIRMED -> 1
        g.units['a'].state = State.CONFIRMED
        g.units['b'].state = State.CONFIRMED
        online_generic_update(g, 'a', 'b', lr=0.1)
        assert abs(ret.w - _expected_weight(0.3, 1.0, 0.1)) < 1e-12

        # dst FAILED -> 0
        g.units['b'].state = State.FAILED
        online_generic_update(g, 'a', 'b', lr=0.1)
        # previous w after first update: 0.37 -> now toward 0
        assert abs(ret.w - _expected_weight(0.37, 0.0, 0.1)) < 1e-12

    def test_no_edge_no_crash(self):
        g, a, b = self._make_units()
        # No edge between a and b, should return without error
        online_generic_update(g, 'a', 'b', lr=0.5)

    def test_clamping_bounds(self):
        g, a, b = self._make_units()
        # Start near bounds and push beyond
        e = Edge('a', 'b', LinkType.SUB, w=1.99)
        g.add_edge(e)
        g.units['a'].state = State.TRUE
        online_generic_update(g, 'a', 'b', lr=1.0)
        assert 0.0 <= e.w <= 2.0

        e2 = Edge('b', 'a', LinkType.SUB, w=0.01)
        g.add_edge(e2)
        g.units['b'].state = State.FAILED
        online_generic_update(g, 'b', 'a', lr=1.0)
        assert 0.0 <= e2.w <= 2.0

