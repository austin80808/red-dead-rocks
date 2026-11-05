"""Curved rows: venue arcs about the stage, and the terrain surface the seating stands on."""

import numpy as np
import pytest
from shapely.geometry import Polygon

from redrocks.benches import terrain_base_elevation
from redrocks.dem import BBox, Dem
from redrocks.terrain_mesh import TerrainSurface, export_terrain_obj
from redrocks.venue import (
    CAPACITY,
    Row,
    build_venue,
    slice_polygon_arcs,
    stage_centre,
)


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


# ---- benches: following the ground ------------------------------------------


class Tilted:
    """Rises 0.1 m per metre east and 0.2 m per metre north."""

    def __call__(self, east, north):
        return 1800.0 + 0.1 * east + 0.2 * north


def section_stations(verts, tris_per_bench=None):
    """Group a single bench's vertices by station: 4 corners each (LB, LT, RT, RB)."""
    return verts.reshape(-1, 4, 3) / 100.0


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


