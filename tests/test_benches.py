from collections import Counter

import numpy as np
import pytest

from redrocks.benches import (
    SEAT_DEPTH_M,
    SEAT_HEIGHT_M,
    export_benches_obj,
    row_polylines,
    terrain_base_elevation,
)
from redrocks.dem import BBox, Dem
from redrocks.geo import LocalProjection
from redrocks.venue import Row, build_venue

BASE = 1750.0


def make_rows(n=68, spacing=1.0, rise=0.4355, half_width=40.0):
    rows, lines = [], []
    for i in range(n):
        north = 100.0 + i * spacing
        rows.append(
            Row(index=i + 1, seats=100, length_m=2 * half_width, elevation_m=1800.0 + i * rise,
                centre_east=0.0, centre_north=north)
        )
        lines.append(np.array([[-half_width, north], [half_width, north]]))
    return rows, lines


def load(path):
    verts, uvs, faces, objects = [], [], [], []
    for line in path.read_text().splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "v":
            verts.append([float(x) for x in tok[1:4]])
        elif tok[0] == "vt":
            uvs.append([float(x) for x in tok[1:3]])
        elif tok[0] == "o":
            objects.append(tok[1])
        elif tok[0] == "f":
            faces.append([tuple(int(i) for i in t.split("/")) for t in tok[1:]])
    return np.array(verts), np.array(uvs), faces, objects


def components(faces, n_verts):
    parent = list(range(n_verts))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for f in faces:
        for (a, _), (b, _) in zip(f, f[1:]):
            parent[find(a - 1)] = find(b - 1)
    return len({find(i) for i in range(n_verts)})


def export(tmp_path, rows=None, lines=None, **kw):
    if rows is None:
        rows, lines = make_rows()
    out = tmp_path / "Benches.obj"
    mesh = export_benches_obj(rows, lines, out, BASE, **kw)
    return mesh, load(out), rows


def test_single_object_and_row_count(tmp_path):
    mesh, (verts, uvs, faces, objects), rows = export(tmp_path)
    assert objects == ["Benches"]
    assert mesh.benches == len(rows) == 68
    assert components(faces, len(verts)) == 68
    assert mesh.triangles == len(faces)
    assert mesh.vertices == len(verts)


def test_watertight(tmp_path):
    _, (verts, _, faces, _), _ = export(tmp_path)
    directed = Counter()
    for f in faces:
        ids = [i for i, _ in f]
        for a, b in zip(ids, ids[1:] + ids[:1]):
            directed[(a, b)] += 1
    # Every directed edge appears once, and its reverse appears once.
    assert all(c == 1 for c in directed.values())
    assert all(directed.get((b, a)) == 1 for a, b in directed)


def test_normals_outward(tmp_path):
    _, (verts, _, faces, _), rows = export(tmp_path, *make_rows(5))
    # Straight benches are convex boxes: each face normal must point away from
    # the bench's centre.
    for bench in range(5):
        fs = [f for f in faces if (f[0][0] - 1) // 8 == bench]
        assert len(fs) == 12
        ids = sorted({i - 1 for f in fs for i, _ in f})
        centre = verts[ids].mean(axis=0)
        for f in fs:
            a, b, c = (verts[i - 1] for i, _ in f)
            n = np.cross(b - a, c - a)
            assert n @ (a - centre) > 0


def test_bent_bench_has_positive_volume(tmp_path):
    rows, _ = make_rows(1)
    for pts in (
        [[-20, 100], [0, 104], [20, 100]],  # bends one way
        [[20, 100], [0, 104], [-20, 100]],  # same, reversed travel
        [[-20, 100], [0, 96], [20, 100]],  # bends the other way
    ):
        mesh, (verts, _, faces, _), _ = export(tmp_path, rows, [np.array(pts, float)], simplify_m=0)
        v = 0.0
        for f in faces:
            a, b, c = (verts[i - 1] for i, _ in f)
            v += a @ np.cross(b, c) / 6.0
        assert v > 0
        assert mesh.triangles == 2 * 8 + 4


def test_every_face_has_valid_uv(tmp_path):
    _, (verts, uvs, faces, _), _ = export(tmp_path)
    assert len(uvs) > 0
    for f in faces:
        assert len(f) == 3
        for v, vt in f:
            assert 1 <= v <= len(verts)
            assert 1 <= vt <= len(uvs)


def test_uv_is_metres_along_row_and_across(tmp_path):
    rows, lines = make_rows(1, half_width=40.0)
    _, (_, uvs, faces, _), _ = export(tmp_path, rows, lines)
    assert uvs[:, 0].min() == pytest.approx(0.0)
    assert uvs[:, 0].max() == pytest.approx(80.0, abs=1e-3)  # row length in metres
    assert uvs[:, 1].max() == pytest.approx(SEAT_DEPTH_M, abs=1e-3)  # lone row keeps 0.9 m


def test_units_are_centimetres_and_aligned_to_terrain_frame(tmp_path):
    rows, lines = make_rows(1, half_width=40.0)
    _, (verts, _, _, _), _ = export(tmp_path, rows, lines)
    assert verts[:, 0].min() == pytest.approx(-4000.0, abs=0.01)  # -40 m east
    assert verts[:, 0].max() == pytest.approx(4000.0, abs=0.01)
    assert verts[:, 1].mean() == pytest.approx(10000.0, abs=0.01)  # 100 m north
    assert np.ptp(verts[:, 1]) == pytest.approx(SEAT_DEPTH_M * 100, abs=0.01)


def test_top_is_45cm_above_row_relative_to_base(tmp_path):
    rows, lines = make_rows(68)
    _, (verts, _, faces, _), _ = export(tmp_path, rows, lines)
    for bench, row in enumerate(rows):
        ids = sorted({i - 1 for f in faces for i, _ in f if (i - 1) // 8 == bench})
        z = verts[ids, 2]
        assert z.min() == pytest.approx((row.elevation_m - BASE) * 100, abs=0.01)
        assert z.max() == pytest.approx((row.elevation_m - BASE + SEAT_HEIGHT_M) * 100, abs=0.01)
        assert z.max() - z.min() == pytest.approx(45.0, abs=0.01)


def test_depth_clamped_so_neighbouring_benches_do_not_overlap(tmp_path):
    rows, lines = make_rows(3, spacing=0.5)
    _, (verts, _, _, _), _ = export(tmp_path, rows, lines)
    first = verts[:8]
    assert np.ptp(first[:, 1]) == pytest.approx(47.5, abs=0.01)  # 0.95 * 0.5 m in cm
    rows, lines = make_rows(3, spacing=0.5)
    _, (verts, _, _, _), _ = export(tmp_path, rows, lines, fit_to_row_gap=False)
    assert np.ptp(verts[:8, 1]) == pytest.approx(90.0, abs=0.01)


def test_realistic_bowl_is_small(tmp_path):
    mesh, _, _ = export(tmp_path)
    assert mesh.triangles < 200_000
    assert mesh.triangles == 68 * 12


def test_collinear_points_are_simplified(tmp_path):
    rows, _ = make_rows(1)
    line = np.array([[x, 100.0] for x in np.linspace(-30, 30, 25)])
    mesh, _, _ = export(tmp_path, rows, [line])
    assert mesh.triangles == 12


def test_mismatched_inputs_raise(tmp_path):
    rows, lines = make_rows(3)
    with pytest.raises(ValueError):
        export_benches_obj(rows, lines[:2], tmp_path / "b.obj", BASE)
    with pytest.raises(ValueError):
        export_benches_obj([], [], tmp_path / "b.obj", BASE)
    with pytest.raises(ValueError):
        export_benches_obj(rows[:1], [np.array([[1.0, 1.0], [1.0, 1.0]])], tmp_path / "b.obj", BASE)


def test_terrain_base_matches_terrain_min():
    bbox = BBox(west=-105.21, south=39.66, east=-105.20, north=39.67)
    data = np.linspace(1900.0, 1950.0, 64, dtype=np.float32).reshape(8, 8)
    assert terrain_base_elevation(Dem(data=data, bbox=bbox), grid=17) == pytest.approx(1900.0)


class _SlopeDem:
    def sample(self, lat, lon):
        return 1800.0 + (lat - 39.6650) * 11000.0


def test_row_polylines_match_venue_rows():
    lat0, lon0 = 39.6655, -105.2055
    proj = LocalProjection(lat0, lon0, 1800.0)
    # A ~120 m x 60 m footprint, the long side north-south.
    d = 0.0005
    ring = [
        (lat0 - 3 * d, lon0 - d), (lat0 - 3 * d, lon0 + d),
        (lat0 + 3 * d, lon0 + d), (lat0 + 3 * d, lon0 - d), (lat0 - 3 * d, lon0 - d),
    ]
    venue = build_venue(ring, _SlopeDem(), proj, lat0, lon0, n_rows=20, capacity=1000)
    lines = row_polylines(venue, ring, proj, n_rows=20)
    assert len(lines) == len(venue.rows) == 20
    for row, line in zip(venue.rows, lines):
        # build_venue's own (curved) arcs come back as they are.
        assert np.array_equal(line, row.polyline)
        assert float(np.hypot(*np.diff(line, axis=0).T).sum()) == pytest.approx(row.length_m, abs=0.01)
        if row.length_m > 20.0:
            assert row.length_m > np.hypot(*(line[-1] - line[0])) + 0.01  # bowed, not a chord


def test_row_polylines_falls_back_to_straight_chords_for_bare_rows():
    """Rows without a stored polyline are re-cut with the straight slicer."""
    from shapely.geometry import Polygon

    from redrocks.venue import slice_polygon

    lat0, lon0 = 39.6655, -105.2055
    proj = LocalProjection(lat0, lon0, 1800.0)
    d = 0.0005
    ring = [
        (lat0 - 3 * d, lon0 - d), (lat0 - 3 * d, lon0 + d),
        (lat0 + 3 * d, lon0 + d), (lat0 + 3 * d, lon0 - d), (lat0 - 3 * d, lon0 - d),
    ]
    poly = Polygon([proj.to_local(a, b)[:2] for a, b in ring])
    rows = []
    for i, chord in enumerate(slice_polygon(poly, 20)):
        m = chord.interpolate(0.5, normalized=True)
        rows.append(Row(i + 1, 50, chord.length, 1800.0 + i, m.x, m.y))  # no polyline

    class Bare:
        pass

    venue = Bare()
    venue.rows = rows
    lines = row_polylines(venue, ring, proj, n_rows=20)
    for row, line in zip(rows, lines):
        assert np.hypot(*(line[-1] - line[0])) == pytest.approx(row.length_m, abs=0.01)
    with pytest.raises(ValueError):
        row_polylines(venue, ring, proj, n_rows=7)
