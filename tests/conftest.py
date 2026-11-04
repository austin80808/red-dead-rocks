"""Shared fixtures."""

from pathlib import Path

import pytest

from redrocks import cli


@pytest.fixture(autouse=True)
def no_scan_blend_by_default(monkeypatch):
    """Keep CLI builds hermetic: the real scan raster (assets/scan) only fits the
    real DEM, so tests opt in to a scan explicitly with --scan-heights."""
    monkeypatch.setattr(cli, "DEFAULT_SCAN_HEIGHTS", Path("/nonexistent/scan_heights.npz"))
