"""tools/fit_scan.py: transform maths and the search recover a known placement."""

import importlib.util
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("fit_scan", ROOT / "tools" / "fit_scan.py")
fit_scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit_scan)


def _terrain():
    xs = np.linspace(-300, 300, 121)
    xx, yy = np.meshgrid(xs, xs[::-1])
    grid = 40 * np.sin(xx / 70) + 30 * np.cos(yy / 55) + 0.1 * xx
    return grid, 300.0


def test_pattern_search_recovers_placement():
    grid, half = _terrain()
    sample = fit_scan.sampler(grid, half)
    rng = np.random.default_rng(1)
    xy = rng.uniform(-120, 120, (1500, 2))
    s, yaw, mirror, t, centre = 0.5, np.radians(40), -1, (10.0, -20.0), (0.0, 0.0)
    wx, wy = fit_scan.to_world(xy, s, yaw, mirror, t, centre)
    z = (sample(wx, wy) - 150.0) / s
    pts = np.column_stack([xy, z])

    class NoLandmarks:
        def errors(self, *a):
            return {"none": 0.0}

    cost = fit_scan.make_cost(pts, sample, NoLandmarks(), mirror, centre)
    c, x = fit_scan.pattern_search(cost, (0.51, 34.0, 5.0, -15.0), (0.01, 4.0, 8.0, 8.0))
    assert c < 0.5
    assert abs(x[1] - 40) < 4.0 and abs(x[2] - 10) < 5 and abs(x[3] + 20) < 5


def test_import_scan_constants_in_range():
    src = (ROOT / "unreal" / "import_scan.py").read_text()
    scale = float(re.search(r"^FIT_SCALE = ([\d.]+)", src, re.M).group(1))
    assert 47.5 * 0.99 <= scale <= 51.6 * 1.08  # m/unit 0.475..0.557, in cm
    assert re.search(r"^FIT_MIRROR_X = -?1$", src, re.M)
