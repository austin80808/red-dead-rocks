"""Curved rows (venue arcs about the stage) and benches that follow the ground."""

from collections import Counter

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from redrocks.benches import SEAT_HEIGHT_M, export_benches_obj, terrain_base_elevation
from redrocks.dem import BBox, Dem
from redrocks.terrain_mesh import TerrainSurface, export_terrain_obj
from redrocks.venue import (
    CAPACITY,
    Row,
    build_venue,
    slice_polygon_arcs,
    stage_centre,
)

BASE = 1750.0


class IdentityProjection:
    def to_local(self, lat, lon, h=0.0):
        return (lon, lat, h)


class RampDem:
    """Rises with east: the west end of a footprint is the top of the bowl."""

    def sample(self, lat, lon):
        return 1900.0 + 0.4 * lon


def fan(r_in=10.0, r_out=80.0, half_angle=50.0, n=60):
    """An annular sector about the origin, opening toward +x."""
    a = np.radians(np.linspace(-half_angle, half_angle, n))
    outer = [(r_out * np.cos(t), r_out * np.sin(t)) for t in a]
    inner = [(r_in * np.cos(t), r_in * np.sin(t)) for t in a[::-1]]
    return Polygon(outer + inner)


def load(path):
    verts, faces = [], []
    for line in path.read_text().splitlines():
        tok = line.split()
        if tok and tok[0] == "v":
            verts.append([float(x) for x in tok[1:4]])
        elif tok and tok[0] == "f":
            faces.append([int(t.split("/")[0]) for t in tok[1:]])
    return np.array(verts), faces


def arc_line(radius, half_angle_deg=30.0, n=40):
    a = np.radians(np.linspace(-half_angle_deg, half_angle_deg, n))
    return np.stack([radius * np.cos(a), radius * np.sin(a)], axis=1)


def one_row(elevation=1800.0):
    return [Row(1, 100, 30.0, elevation, 0.0, 0.0)]


def export(tmp_path, lines, rows=None, **kw):
    rows = rows or [Row(i + 1, 100, 30.0, 1800.0 + i, 0.0, 0.0) for i in range(len(lines))]
    out = tmp_path / "Benches.obj"
    mesh = export_benches_obj(rows, lines, out, BASE, **kw)
    verts, faces = load(out)
    return mesh, verts, faces


# ---- venue: arcs ----------------------------------------------------------


def test_arcs_are_circles_about_the_centre():
    arcs = slice_polygon_arcs(fan(), 30, (0.0, 0.0))
    assert len(arcs) == 30
    radii = []
    for arc in arcs:
        r = np.hypot(*np.array(arc.coords).T)
        assert r.max() - r.min() < 0.01  # one circle
        radii.append(float(r.mean()))
    assert radii == sorted(radii)  # stage first
    assert len(set(round(r, 3) for r in radii)) == 30


def test_arcs_curve_and_stay_inside_the_footprint():
    poly = fan()
    for arc in slice_polygon_arcs(poly, 20, (0.0, 0.0)):
        pts = np.array(arc.coords)
        chord = np.hypot(*(pts[-1] - pts[0]))
        assert arc.length > chord + 0.1  # bowed, not straight
        assert poly.buffer(1e-6).contains(arc)


def test_arcs_run_counter_clockwise_about_the_centre():
    for arc in slice_polygon_arcs(fan(), 10, (0.0, 0.0)):
        pts = np.array(arc.coords)
        a0, a1 = np.arctan2(pts[0, 1], pts[0, 0]), np.arctan2(pts[-1, 1], pts[-1, 0])
        assert a1 > a0


def test_arcs_drop_grazing_ends_but_return_requested_count():
    # A rectangle seen from a corner-ish centre: radius range starts and ends in slivers.
    poly = Polygon([(0, 0), (0, 30), (90, 30), (90, 0)])
    arcs = slice_polygon_arcs(poly, 40, (-20.0, 15.0))
    assert len(arcs) == 40
    assert min(a.length for a in arcs) >= 5.0


def test_arcs_follow_one_branch_of_a_concave_footprint():
    # A "C" opening to the east: circles about the origin cross both arms.
    c_shape = Polygon([(10, -40), (10, 40), (40, 40), (40, 25), (25, 25), (25, -25), (40, -25), (40, -40)])
    arcs = slice_polygon_arcs(c_shape, 25, (0.0, 0.0))
    assert arcs
    # Consecutive rows keep to the same arm: the midpoint never jumps across the gap.
    mids = np.array([a.interpolate(0.5, normalized=True).coords[0] for a in arcs])
    assert np.max(np.hypot(*np.diff(mids, axis=0).T)) < 12.0


def test_arcs_reject_empty_footprint():
    with pytest.raises(ValueError):
        slice_polygon_arcs(Polygon(), 10, (0.0, 0.0))


def test_stage_centre_is_pushed_beyond_the_low_end():
    poly = Polygon([(0, 0), (0, 30), (90, 30), (90, 0)])  # long axis east-west
    low_east = lambda e, n: 1900.0 + 0.4 * e   # stage at the west end
    low_west = lambda e, n: 1900.0 - 0.4 * e   # stage at the east end
    ce, cn = stage_centre(poly, low_east, setback_m=25.0)
    assert ce == pytest.approx(-25.0, abs=0.5) and cn == pytest.approx(15.0, abs=0.5)
    ce, cn = stage_centre(poly, low_west, setback_m=25.0)
    assert ce == pytest.approx(115.0, abs=0.5) and cn == pytest.approx(15.0, abs=0.5)
    ce, _ = stage_centre(poly, low_east, setback_m=0.0)
    assert ce == pytest.approx(0.0, abs=0.5)


# ---- venue: build_venue ----------------------------------------------------


def footprint():
    return [(-15, 0), (-15, 40), (15, 40), (15, 0), (-15, 0)]  # (lat, lon) as metres


def test_build_venue_rows_are_curved_arcs_with_exact_seats():
    venue = build_venue(footprint(), RampDem(), IdentityProjection(), 0.0, 0.0)
    assert venue.total_seats == CAPACITY
    assert len(venue.rows) == 70
    assert all(r.polyline is not None for r in venue.rows)
    for r in venue.rows:
        chord = np.hypot(*(r.polyline[-1] - r.polyline[0]))
        assert r.length_m > chord  # curved
        assert r.length_m == pytest.approx(float(np.hypot(*np.diff(r.polyline, axis=0).T).sum()))
    elev = [r.elevation_m for r in venue.rows]
    assert elev == sorted(elev)  # row 1 at the stage, rising


def test_build_venue_rows_are_equidistant_from_one_stage_centre():
    venue = build_venue(footprint(), RampDem(), IdentityProjection(), 0.0, 0.0)
    centre = stage_centre(
        Polygon([(lon, lat) for lat, lon in footprint()]), lambda e, n: 1900.0 + 0.4 * e
    )
    for r in venue.rows:
        d = np.hypot(r.polyline[:, 0] - centre[0], r.polyline[:, 1] - centre[1])
        assert d.max() - d.min() < 0.02


def test_rake_uses_plan_distance_not_just_north():
    from redrocks.venue import OFFICIAL_RISE_M, Venue
    import math

    rows = [
        Row(1, 1, 10.0, 1900.0, 0.0, 0.0),
        Row(2, 1, 10.0, 1900.0 + OFFICIAL_RISE_M, 40.0, 30.0),  # 50 m away in plan
    ]
    assert Venue(rows=rows).rake_degrees == pytest.approx(math.degrees(math.atan(OFFICIAL_RISE_M / 50.0)))


# ---- benches: curved polylines ---------------------------------------------


def test_curved_bench_is_closed_and_positive_volume(tmp_path):
    mesh, verts, faces = export(tmp_path, [arc_line(30.0)], simplify_m=0)
    directed = Counter()
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            directed[(a, b)] += 1
    assert all(c == 1 for c in directed.values())
    assert all(directed.get((b, a)) == 1 for a, b in directed)
    vol = sum(
        verts[f[0] - 1] @ np.cross(verts[f[1] - 1], verts[f[2] - 1]) / 6.0 for f in faces
    )
    assert vol > 0


def test_curved_bench_keeps_constant_depth_around_the_bend(tmp_path):
    """Mitres on an arc must give a strip of even width, not pinch or flare."""
    radius, depth = 25.0, 0.9
    _, verts, _ = export(tmp_path, [arc_line(radius)], simplify_m=0)
    r = np.hypot(verts[:, 0], verts[:, 1]) / 100.0
    inner, outer = radius - depth / 2, radius + depth / 2
    # Mitre corners of a polygonal arc land at radius / cos(half step): tiny.
    assert r.min() == pytest.approx(inner, abs=0.02)
    assert r.max() == pytest.approx(outer, abs=0.02)
    assert np.all((np.abs(r - inner) < 0.02) | (np.abs(r - outer) < 0.02))


def test_sharp_mitre_is_capped(tmp_path):
    spike = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 0.2]])  # nearly a U-turn
    _, verts, _ = export(tmp_path, [spike], simplify_m=0)
    assert np.all(np.abs(verts[:, :2]) < 2500)  # within 25 m, nothing flung out


def test_arc_polyline_is_simplified_but_stays_within_tolerance(tmp_path):
    line = arc_line(60.0, half_angle_deg=25.0, n=400)
    mesh, verts, _ = export(tmp_path, [line], simplify_m=0.1)
    assert mesh.vertices < 4 * 40  # far fewer than 400 stations
    r = np.hypot(verts[:, 0], verts[:, 1]) / 100.0
    assert r.min() > 60.0 - 0.45 - 0.12


# ---- benches: following the ground ------------------------------------------


class Tilted:
    """Rises 0.1 m per metre east and 0.2 m per metre north."""

    def __call__(self, east, north):
        return 1800.0 + 0.1 * east + 0.2 * north


def section_stations(verts, tris_per_bench=None):
    """Group a single bench's vertices by station: 4 corners each (LB, LT, RT, RB)."""
    return verts.reshape(-1, 4, 3) / 100.0


def test_bench_follows_the_ground_along_its_length(tmp_path):
    line = np.array([[-40.0, 100.0], [40.0, 100.0]])
    ground = Tilted()
    mesh, verts, _ = export(tmp_path, [line], ground=ground)
    st = section_stations(verts)
    assert len(st) > 30  # subdivided, not a two-station box
    for lb, lt, rt, rb in st:
        gc = ground(0.5 * (lt[0] + rt[0]), 0.5 * (lt[1] + rt[1]))
        assert lt[2] + BASE == pytest.approx(gc + SEAT_HEIGHT_M, abs=1e-6)
        assert rt[2] + BASE == pytest.approx(gc + SEAT_HEIGHT_M, abs=1e-6)
    tops = np.array([lt[2] for _, lt, _, _ in st])
    assert tops.max() - tops.min() == pytest.approx(0.1 * 80.0, abs=0.01)  # rises with the slope


def test_ends_neither_float_nor_sink(tmp_path):
    line = np.array([[-40.0, 100.0], [40.0, 100.0]])
    ground = Tilted()
    _, verts, _ = export(tmp_path, [line], ground=ground)
    for lb, lt, rt, rb in section_stations(verts):
        under = [ground(c[0], c[1]) for c in (lb, rb)]
        centre = ground(0.5 * (lt[0] + rt[0]), 0.5 * (lt[1] + rt[1]))
        lowest = min(under + [centre])
        assert lb[2] + BASE == pytest.approx(lowest, abs=1e-6)  # base on the lowest ground
        assert rb[2] + BASE == pytest.approx(lowest, abs=1e-6)
        assert min(under) >= lb[2] + BASE - 1e-6  # no corner of the bench floats above the ground


def test_curved_bench_follows_the_ground_too(tmp_path):
    ground = Tilted()
    _, verts, _ = export(tmp_path, [arc_line(30.0)], ground=ground)
    st = section_stations(verts)
    for lb, lt, rt, rb in st:
        centre = ground(0.5 * (lt[0] + rt[0]), 0.5 * (lt[1] + rt[1]))
        assert lt[2] + BASE == pytest.approx(centre + SEAT_HEIGHT_M, abs=1e-4)
    assert np.ptp([lt[2] for _, lt, _, _ in st]) > 1.0


def test_flat_ground_gives_a_flat_45cm_bench(tmp_path):
    _, verts, _ = export(tmp_path, [arc_line(30.0)], ground=lambda e, n: 1800.0)
    z = verts[:, 2] / 100.0 + BASE
    assert z.min() == pytest.approx(1800.0, abs=1e-6)
    assert z.max() == pytest.approx(1800.0 + SEAT_HEIGHT_M, abs=1e-6)


def test_without_ground_benches_stay_flat_at_the_row_elevation(tmp_path):
    rows = one_row(1800.0)
    out = tmp_path / "b.obj"
    export_benches_obj(rows, [arc_line(30.0)], out, BASE)
    verts, _ = load(out)
    z = verts[:, 2] / 100.0 + BASE
    assert z.min() == pytest.approx(1800.0, abs=1e-6)
    assert z.max() == pytest.approx(1800.0 + SEAT_HEIGHT_M, abs=1e-6)


def test_ground_step_bounds_station_spacing(tmp_path):
    line = np.array([[-40.0, 100.0], [40.0, 100.0]])
    _, verts, _ = export(tmp_path, [line], ground=Tilted(), ground_step_m=2.5)
    xs = np.unique(np.round(verts[:, 0] / 100.0, 6))
    assert np.max(np.diff(xs)) <= 2.5 + 1e-6


def test_row_gap_is_measured_between_arcs_not_between_midpoints(tmp_path):
    # Three concentric arcs 0.5 m apart whose midpoints are nowhere near 0.5 m apart.
    lines = [arc_line(30.0 + 0.5 * i, half_angle_deg=10.0 + 15 * i, n=30) for i in range(3)]
    rows = [Row(i + 1, 1, 10.0, 1800.0, 30.0 * np.cos(i), 40.0 * i) for i in range(3)]
    out = tmp_path / "b.obj"
    export_benches_obj(rows, lines, out, BASE, simplify_m=0)
    verts, _ = load(out)
    first = verts[: 4 * 30]
    r = np.hypot(first[:, 0], first[:, 1]) / 100.0
    assert r.max() - r.min() == pytest.approx(0.95 * 0.5, abs=0.01)  # depth clamped to 95% of 0.5 m


# ---- TerrainSurface ----------------------------------------------------------


def make_dem():
    rng = np.random.default_rng(3)
    bbox = BBox(west=-105.22, south=39.65, east=-105.19, north=39.68)
    return Dem(data=(1900 + 40 * rng.random((60, 60))).astype(np.float32), bbox=bbox)


def test_terrain_surface_matches_the_exported_mesh(tmp_path):
    dem, grid = make_dem(), 9
    out = tmp_path / "t.obj"
    export_terrain_obj(dem, out, grid=grid)
    verts, faces = load(out)
    surface = TerrainSurface(dem, grid)
    assert surface.base_elevation_m == pytest.approx(terrain_base_elevation(dem, grid))
    base = surface.base_elevation_m
    for x, y, z in verts:  # exact at every vertex
        assert surface(x / 100.0, y / 100.0) == pytest.approx(z / 100.0 + base, abs=1e-3)
    for f in faces:  # planar across each triangle, so it follows the mesh's diagonals
        tri = np.array([verts[i - 1] for i in f[:3]])
        c = tri.mean(axis=0)
        assert surface(c[0] / 100.0, c[1] / 100.0) == pytest.approx(c[2] / 100.0 + base, abs=1e-3)


def test_terrain_surface_clamps_outside_the_extent():
    surface = TerrainSurface(make_dem(), 9)
    assert surface(1e6, 1e6) == pytest.approx(surface(1e5, 1e5))
    with pytest.raises(ValueError):
        TerrainSurface(make_dem(), 1)
