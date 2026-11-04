"""redrocks.scan_blend: the terrain is pulled under the scan and eased back to the DEM."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import shapely

from redrocks.dem import Dem, square_bbox
from redrocks.scan_blend import (
    ScanHeights,
    _node_grid,
    conform_dem,
    fade_alpha,
    footprint,
    harmonic_fill,
    lower_envelope,
    mesh_height,
)
from redrocks.terrain_mesh import resample_to_grid

ROOT = Path(__file__).resolve().parent.parent
GRID = 61  # 10 m mesh cells over the 600 m test bbox


def flat_dem(z=1900.0, n=241):
    bbox = square_bbox(39.6655, -105.2055, 600.0)
    return Dem(data=np.full((n, n), z, dtype=np.float32), bbox=bbox)


def bumpy_scan(radius=100.0, base=1906.0):
    """A disc of scan floor, 6 m above the flat DEM, with 3 m ripples."""
    cell = 2.0
    xs = np.arange(-150.0, 150.0, cell) + cell / 2
    e, n = np.meshgrid(xs, xs)
    h = base + 3.0 * np.sin(e / 13.0) * np.cos(n / 9.0)
    h[np.hypot(e, n) > radius] = np.nan
    return ScanHeights(heights=h, x0=-150.0, y0=-150.0, cell=cell)


def test_from_points_keeps_the_lowest_point():
    s = ScanHeights.from_points([0.5, 1.5, 0.7], [0.5, 0.5, 0.6], [10.0, 7.0, 4.0], cell=2.0)
    assert s.heights.shape == (1, 1)
    assert s.heights[0, 0] == 4.0


def test_save_load_round_trip(tmp_path):
    s = bumpy_scan()
    s.save(tmp_path / "s.npz")
    t = ScanHeights.load(tmp_path / "s.npz")
    assert (t.x0, t.y0, t.cell) == (s.x0, s.y0, s.cell)
    np.testing.assert_allclose(t.heights, s.heights, atol=1e-3)


def test_footprint_closes_small_holes():
    s = bumpy_scan()
    s.heights[75, 75] = np.nan  # one missing cell in the middle
    assert footprint(s)[75, 75]
    assert not footprint(s)[0, 0]


def test_fade_alpha_is_zero_at_the_edge_and_one_inside():
    s = bumpy_scan(radius=100.0)
    a = fade_alpha(s, fade_m=30.0)
    e, n = s.centres()
    r = np.hypot(e, n)
    assert a[r > 101].max() == 0.0
    assert a[r < 60].min() == 1.0
    ring = (r > 80) & (r < 90)
    assert 0.0 < a[ring].mean() < 1.0


def test_harmonic_fill_is_linear_between_fixed_columns():
    v = np.zeros((5, 11))
    v[:, -1] = 10.0
    fixed = np.zeros_like(v, dtype=bool)
    fixed[:, 0] = fixed[:, -1] = True
    out = harmonic_fill(v, fixed, tol=1e-6, max_iter=20000)
    np.testing.assert_allclose(out[2], np.linspace(0.0, 10.0, 11), atol=0.01)


def test_lower_envelope_puts_every_point_under_the_mesh_margin():
    xs = np.linspace(0.0, 40.0, 5)
    ys = np.linspace(40.0, 0.0, 5)
    nodes = np.full((5, 5), 10.0)
    rng = np.random.default_rng(3)
    e, n = rng.uniform(0, 40, (2, 500))
    floor = 9.0 + rng.uniform(-2, 2, 500)
    out = lower_envelope(nodes, xs, ys, e, n, floor, margin_m=0.4)
    assert (mesh_height(out, xs, ys, e, n) <= floor - 0.4 + 1e-6).all()
    assert (out <= nodes).all()


def test_conform_dem_sits_under_the_scan_and_returns_to_the_dem():
    dem = flat_dem()
    scan = bumpy_scan()
    out = conform_dem(dem, scan, grid=GRID, falloff_m=60.0, margin_m=0.4)
    xs, ys = _node_grid(dem, GRID)
    mesh = resample_to_grid(out, GRID)
    e, n = scan.centres()
    ok = ~np.isnan(scan.heights)
    # The mesh, as triangulated, is under every scan point by about the margin.
    gap = scan.heights[ok] - mesh_height(mesh, xs, ys, e[ok], n[ok])
    assert gap.min() > 0.2
    assert np.median(gap) < 3.0  # follows the scan, not sunk far below it
    # Far outside the falloff the DEM is untouched; between, no overshoot.
    east, north = np.meshgrid(xs, ys)
    r = np.hypot(east, north)
    assert np.allclose(mesh[r > 175], 1900.0)
    band = (r > 105) & (r < 155)
    assert mesh[band].min() >= 1900.0 - 1e-3 and mesh[band].max() <= 1909.0


def test_keep_out_leaves_the_dem_alone():
    dem = flat_dem()
    keep = shapely.Point(0.0, 0.0).buffer(30.0)
    out = conform_dem(dem, bumpy_scan(), grid=GRID, keep_out=keep, keep_out_m=1.0, keep_out_taper_m=6.0)
    xs, ys = _node_grid(dem, GRID)
    east, north = np.meshgrid(xs, ys)
    mesh = resample_to_grid(out, GRID)
    assert np.allclose(mesh[np.hypot(east, north) < 30.0], 1900.0)
    assert mesh[(np.hypot(east, north) > 50) & (np.hypot(east, north) < 90)].min() > 1901.0


def test_cli_blends_only_with_a_scan_raster(monkeypatch, tmp_path):
    from redrocks import cli
    from redrocks.osm import OsmData

    def fake_dem(bbox, size=2017):
        data = np.full((65, 65), 1900.0, dtype=np.float32)
        data[0, 0] = 1890.0  # the minimum, far from the scan
        return Dem(data=data, bbox=bbox)

    monkeypatch.setattr(cli, "fetch_dem", fake_dem)
    monkeypatch.setattr(cli, "fetch_osm", lambda bbox: OsmData(nodes={}, ways=[]))
    monkeypatch.setattr(cli, "build_monoliths", lambda *a, **k: [])
    bumpy_scan().save(tmp_path / "scan.npz")
    args = ["build", "--out", str(tmp_path / "out"), "--landscape-size", "33", "--terrain-grid", "65"]
    assert cli.main(args + ["--scan-heights", str(tmp_path / "scan.npz")]) == 0
    import json

    info = json.loads((tmp_path / "out" / "manifest.json").read_text())["scan_blend"]
    assert info["changed_area_m2"] > 10000 and info["raised_max_m"] > 3.0
    assert cli.main(args + ["--scan-heights", ""]) == 0
    assert "scan_blend" not in json.loads((tmp_path / "out" / "manifest.json").read_text())


def test_scan_heights_tool_reads_the_fit_constants():
    spec = importlib.util.spec_from_file_location("scan_heights", ROOT / "tools" / "scan_heights.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    fit = tool.fit_constants((ROOT / "unreal" / "import_scan.py").read_text())
    assert 0.44 < fit["scale_m"] < 0.56
    assert fit["mirror"] in (-1, 1)
    assert len(fit["centre"]) == 2 and len(fit["offset_m"]) == 3


def test_committed_fade_mask_matches_its_placement():
    import json

    meta = json.loads((ROOT / "assets" / "scan" / "scan_fade.json").read_text())
    s = ScanHeights.load(ROOT / "assets" / "scan" / "scan_heights.npz")
    ny, nx = s.shape
    assert meta["texels"] == [nx, ny]
    assert meta["x0_cm"] == pytest.approx(s.x0 * 100.0, abs=1.0)
    assert meta["y1_cm"] == pytest.approx((s.y0 + ny * s.cell) * 100.0, abs=1.0)
