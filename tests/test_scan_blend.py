"""redrocks.scan_blend: the terrain is pulled under the scan and eased back to the DEM."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from redrocks.dem import Dem, square_bbox
from redrocks.scan_blend import (
    ScanGround,
    ScanHeights,
    _node_grid,
    conform_dem,
    fade_alpha,
    footprint,
    ground_surface,
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


def test_conform_dem_has_no_keep_out():
    """The benches and stage stand on the scan now, so no terrain is exempt."""
    import inspect

    assert "keep_out" not in inspect.signature(conform_dem).parameters


def sloped_points(seed=0, n_per_m2=11.0, roof=True):
    """Scan-like points on a 20 degree plane, with a roof 12 m over part of it."""
    rng = np.random.default_rng(seed)
    n = int(60 * 60 * n_per_m2)
    e = rng.uniform(-30.0, 30.0, n)
    north = rng.uniform(-30.0, 30.0, n)
    z = 1900.0 + 0.37 * e + 0.1 * north + rng.normal(0.0, 0.03, n)
    if roof:
        r = (np.abs(e) < 8) & (np.abs(north) < 8) & (rng.uniform(size=n) < 0.6)
        z[r] += 12.0
    return e, north, z


def test_ground_surface_follows_a_slope_where_the_floor_sits_low():
    e, n, z = sloped_points(roof=False)
    g = ground_surface(e, n, z)
    ce, cn = g.centres()
    true = 1900.0 + 0.37 * ce + 0.1 * cn
    inner = (np.abs(ce) < 28) & (np.abs(cn) < 28)
    err = g.heights[inner] - true[inner]
    assert np.abs(err).max() < 0.05
    # The floor raster (lowest point per cell) reads a slope well under its surface.
    floor = ScanHeights.from_points(e, n, z, cell=2.0)
    fe, fn = floor.centres()
    sag = floor.heights - (1900.0 + 0.37 * fe + 0.1 * fn)
    assert np.median(sag) < -0.25


def test_ground_surface_leaves_out_a_roof_over_the_ground():
    e, n, z = sloped_points()
    g = ground_surface(e, n, z)
    ce, cn = g.centres()
    under = (np.abs(ce) < 6) & (np.abs(cn) < 6)
    assert np.abs(g.heights[under] - (1900.0 + 0.37 * ce[under] + 0.1 * cn[under])).max() < 0.05


def test_ground_surface_is_nan_without_points():
    g = ground_surface([0.2, 0.4, 0.6], [0.2, 0.7, 0.4], [5.0, 5.0, 5.0])
    assert g.heights.shape == (1, 1) and g.heights[0, 0] == pytest.approx(5.0)
    e, n, z = sloped_points(roof=False)
    keep = (e < -10) | (e > 10)
    g = ground_surface(e[keep], n[keep], z[keep])
    ce, _ = g.centres()
    assert np.isnan(g.heights[np.abs(ce) < 8]).all()


def test_scan_ground_samples_the_raster_and_falls_back_off_it():
    e, n, z = sloped_points(roof=False)
    keep = (e < -10) | (e > 10)
    ground = ScanGround(ground_surface(e[keep], n[keep], z[keep]), fallback=lambda x, y: -1.0)
    assert ground(20.3, 4.7) == pytest.approx(1900.0 + 0.37 * 20.3 + 0.1 * 4.7, abs=0.05)
    assert ground(0.0, 0.0) == -1.0  # the hole in the middle
    assert ground(500.0, 0.0) == -1.0  # off the raster


def test_committed_ground_raster_matches_the_floor_raster():
    """scan_ground.npz is the same scan as scan_heights.npz: never under its floor by much."""
    floor = ScanHeights.load(ROOT / "assets" / "scan" / "scan_heights.npz")
    ground = ScanHeights.load(ROOT / "assets" / "scan" / "scan_ground.npz")
    assert ground.cell == 1.0
    ge, gn = ground.centres()
    ok = np.isfinite(ground.heights)
    f = ScanGround(floor, fallback=lambda x, y: np.nan)
    sample = np.flatnonzero(ok.ravel())[::997]
    diff = np.array([ground.heights.ravel()[i] - f(ge.ravel()[i], gn.ravel()[i]) for i in sample])
    diff = diff[np.isfinite(diff)]
    assert len(diff) > 100
    assert np.median(diff) > 0.0  # a plane through the points sits over their lowest


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
