"""
Unit tests for small helper functions in the viz module.
"""

import pytest


class TestGetSpeedLabelFromDelay:
    def test_exact_mapping(self):
        streamlit = pytest.importorskip("streamlit")
        from viz.app_streamlit import get_speed_label_from_delay  # noqa: WPS433
        assert get_speed_label_from_delay(0.8) == "Slow"
        assert get_speed_label_from_delay(0.5) == "Normal"
        assert get_speed_label_from_delay(0.2) == "Fast"

    def test_fallback_logic(self):
        streamlit = pytest.importorskip("streamlit")
        from viz.app_streamlit import get_speed_label_from_delay  # noqa: WPS433
        assert get_speed_label_from_delay(0.7) == "Slow"   # > 0.5
        assert get_speed_label_from_delay(0.3) == "Fast"   # < 0.5

