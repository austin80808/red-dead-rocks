"""Terraced seating: tread levels, terrace, plank and bracket meshes, and the scan cut."""

from collections import Counter

import numpy as np
import pytest

from redrocks.scan_blend import ScanHeights
from redrocks.seating import (
    BRACKET_SPACING_M,
    BURY_M,
    CUT_TEXEL_M,
    MAX_RISE_M,
    MIN_RISE_M,
    PLANK_BACK_M,
    PLANK_OVERHANG_M,
    PLANK_TOP_M,
    export_seating,
    seating_fade,
    seating_layout,
    tread_levels,
)
from redrocks.venue import Row

BASE = 1750.0
R0, GAP, N = 30.0, 1.2, 40
HALF_ANGLE = 30.0


def arc(radius, half_angle=HALF_ANGLE, n=60):
    a = np.radians(np.linspace(-half_angle, half_angle, n))
    return np.stack([radius * np.cos(a), radius * np.sin(a)], axis=1)


def rows_and_lines(n=N, elevations=None):
    lines = [arc(R0 + i * GAP) for i in range(n)]
    if elevations is None:
        elevations = [1800.0 + 0.4 * i for i in range(n)]
    rows = [Row(i + 1, 100, float(np.hypot(*np.diff(l, axis=0).T).sum()), float(elevations[i]),
                float(l[len(l) // 2, 0]), float(l[len(l) // 2, 1]), polyline=l) for i, l in enumerate(lines)]
    return rows, lines


def profile(r):
    """Bowl ground: rises 0.30 m per metre out, a little steeper higher up."""
    return 1800.0 + 0.30 * (r - R0) + 0.002 * (r - R0) ** 2


def bowl(e, n):
    return profile(float(np.hypot(e, n)))


def floor(e, n):
    return bowl(e, n) - 0.9


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
            faces.append([tuple(int(i) for i in t.split("/")[:2]) for t in tok[1:]])
    return np.array(verts), np.array(uvs), faces, objects


def build(tmp_path, ground=bowl, clearance=0.45, **kw):
    rows, lines = rows_and_lines()
    layout = seating_layout(rows, lines, ground=ground, floor=floor if ground else None,
                            clearance_m=clearance, **kw)
    mesh = export_seating(layout, tmp_path, BASE, ground=ground, floor=floor if ground else None)
    return layout, mesh, rows, lines


def tri_normal(v, f):
    a, b, c = (v[i - 1] for i, _ in f)
    n = np.cross(b - a, c - a)
    return n / np.linalg.norm(n)


# ---- tread levels ------------------------------------------------------------

def test_levels_follow_the_bowl_profile_plus_clearance():
    rows, lines = rows_and_lines()
    levels, info = tread_levels(rows, lines, ground=bowl, clearance_m=0.45)
    expected = [profile(R0 + i * GAP + GAP / 2) + 0.45 for i in range(N)]
    assert np.allclose(levels, expected, atol=0.05)
    rises = np.diff(expected)
    assert info["rise_min_m"] == pytest.approx(rises.min(), abs=0.02)
    assert info["rise_max_m"] == pytest.approx(rises.max(), abs=0.02)


def test_a_bump_in_one_row_does_not_move_its_level():
    """The FOH booth or a tree in one row's band is not that row's tread."""
    rows, lines = rows_and_lines()

    def bumpy(e, n):
        r = np.hypot(e, n)
        return bowl(e, n) + (1.5 if R0 + 10 * GAP <= r < R0 + 11 * GAP else 0.0)

    levels, info = tread_levels(rows, lines, ground=bumpy)
    assert levels[10] == pytest.approx(profile(R0 + 10.5 * GAP), abs=0.05)
    assert 11 in info["rows_ignored"]


def test_levels_without_ground_smooth_the_row_elevations():
    noisy = [1800.0 + 0.4 * i + (0.15 if i % 2 else -0.15) for i in range(N)]
    rows, lines = rows_and_lines(elevations=noisy)
    levels, _ = tread_levels(rows, lines, ground=None, clearance_m=0.45)
    # Clearance is for the scan's collision only; no ground, no clearance.
    assert np.allclose(levels, [1800.0 + 0.4 * i for i in range(N)], atol=0.06)


def test_levels_always_rise_even_where_the_ground_flattens():
    flat_top = [1800.0 + 0.45 * min(i, 30) for i in range(N)]
    rows, lines = rows_and_lines(elevations=flat_top)
    levels, _ = tread_levels(rows, lines)
    assert np.diff(levels).min() >= MIN_RISE_M - 1e-9


def test_levels_need_matching_rows_and_lines():
    rows, lines = rows_and_lines()
    with pytest.raises(ValueError):
        tread_levels(rows, lines[:-1])


# ---- terraces ----------------------------------------------------------------

def test_writes_three_meshes_each_one_object(tmp_path):
    _, mesh, _, _ = build(tmp_path)
    for name in ("Benches", "Bench_Planks", "Bench_Brackets"):
        _, _, faces, objects = load(tmp_path / f"{name}.obj")
        assert objects == [name] and faces
    assert mesh.terraces == N and mesh.triangles["Benches"] > 0


def test_every_mesh_is_watertight_and_faces_outward(tmp_path):
    build(tmp_path)
    for name in ("Benches", "Bench_Planks", "Bench_Brackets"):
        v, _, faces, _ = load(tmp_path / f"{name}.obj")
        edges = Counter()
        for f in faces:
            ids = [i for i, _ in f]
            for a, b in zip(ids, ids[1:] + ids[:1]):
                edges[(a, b)] += 1
        # Every directed edge has its reverse exactly once: closed and consistently wound.
        assert all(edges[(b, a)] == 1 for (a, b) in edges), name
        vol = sum(np.dot(v[f[0][0] - 1], np.cross(v[f[1][0] - 1], v[f[2][0] - 1])) for f in faces) / 6.0
        assert vol > 0, name


def test_every_face_has_a_valid_uv(tmp_path):
    build(tmp_path)
    for name in ("Benches", "Bench_Planks", "Bench_Brackets"):
        _, uvs, faces, _ = load(tmp_path / f"{name}.obj")
        assert all(1 <= t <= len(uvs) for f in faces for _, t in f), name


def test_each_tread_is_flat_at_its_level_from_its_row_line_back(tmp_path):
    layout, _, rows, _ = build(tmp_path)
    v, _, faces, _ = load(tmp_path / "Benches.obj")
    m = v / 100.0
    tops = [f for f in faces if tri_normal(v, f)[2] > 0.99]
    for f in tops:
        corners = m[[i - 1 for i, _ in f]]
        r = np.hypot(*corners[:, :2].mean(axis=0))
        if r < R0:
            continue  # a front step
        k = int((r - R0) // GAP)
        assert np.allclose(corners[:, 2] + BASE, layout.levels[k], atol=1e-3)
    for k, level in enumerate(layout.levels):
        on = np.array([m[i - 1] for f in tops for i, _ in f if np.isclose(m[i - 1, 2] + BASE, level, atol=1e-3)])
        r = np.hypot(on[:, 0], on[:, 1])
        # The front edge (the riser) stands on the row's own line; the tread
        # runs back to the next row's riser and a little under it.
        assert r.min() == pytest.approx(R0 + k * GAP, abs=0.01)
        if k + 1 < len(layout.levels):
            assert R0 + (k + 1) * GAP <= r.max() <= R0 + (k + 1) * GAP + 0.1


def test_terraces_are_buried_below_the_floor(tmp_path):
    build(tmp_path)
    v, _, faces, _ = load(tmp_path / "Benches.obj")
    m = v / 100.0
    bottoms = {i - 1 for f in faces if tri_normal(v, f)[2] < -0.99 for i, _ in f}
    for e, n, z in m[sorted(bottoms)]:
        assert z + BASE <= floor(e, n) - BURY_M + 0.35  # stations every couple of metres


def test_risers_are_flat_faced_walls(tmp_path):
    build(tmp_path)
    v, _, faces, _ = load(tmp_path / "Benches.obj")
    normals = np.array([tri_normal(v, f) for f in faces])
    # Only tops, bottoms and vertical walls: nothing sloped to snag the player.
    nz = np.abs(normals[:, 2])
    assert np.all((nz > 0.99) | (nz < 1e-3))


# ---- front steps -------------------------------------------------------------

def test_front_steps_climb_from_the_apron_to_row_one(tmp_path):
    def sunk_apron(e, n):
        r = np.hypot(e, n)
        return bowl(e, n) - (1.2 if r < R0 else 0.0)

    layout, _, _, _ = build(tmp_path, ground=sunk_apron)
    steps = layout.front_steps
    assert len(steps) >= 2
    lows = [sunk_apron(*steps[0].front[len(steps[0].front) // 2]) - 0.0]
    heights = [lows[0]] + [s.level_m for s in steps] + [layout.levels[0]]
    assert max(np.diff(heights)) <= MAX_RISE_M + 1e-6
    assert all(np.hypot(*s.front[len(s.front) // 2]) < R0 for s in steps)


def test_no_front_steps_when_row_one_is_one_step_up(tmp_path):
    layout, _, _, _ = build(tmp_path, clearance=0.0)
    assert layout.front_steps == []


# ---- planks and brackets -----------------------------------------------------

def test_planks_sit_on_each_riser_edge(tmp_path):
    layout, _, _, _ = build(tmp_path)
    v, _, _, _ = load(tmp_path / "Bench_Planks.obj")
    m = v / 100.0
    r = np.hypot(m[:, 0], m[:, 1])
    for k, level in enumerate(layout.levels):
        row = m[(r > R0 + k * GAP - 0.5) & (r < R0 + k * GAP + 0.5)]
        rr = np.hypot(row[:, 0], row[:, 1])
        assert row[:, 2].max() + BASE == pytest.approx(level + PLANK_TOP_M, abs=1e-3)
        assert rr.min() == pytest.approx(R0 + k * GAP - PLANK_OVERHANG_M, abs=0.02)
        assert rr.max() == pytest.approx(R0 + k * GAP + PLANK_BACK_M, abs=0.02)


def test_brackets_line_up_across_the_rows(tmp_path):
    build(tmp_path)
    v, _, faces, _ = load(tmp_path / "Bench_Brackets.obj")
    m = v / 100.0
    # One bracket = 8 vertices (a box); group them and take bearings of the centres.
    centres = m.reshape(-1, 8, 3).mean(axis=1)
    r = np.hypot(centres[:, 0], centres[:, 1])
    bearing = np.degrees(np.arctan2(centres[:, 1], centres[:, 0]))
    low = np.round(bearing[np.abs(r - (R0 + 5 * GAP)) < 0.5], 2)
    high = np.round(bearing[np.abs(r - (R0 + 35 * GAP)) < 0.5], 2)
    assert len(low) >= 3 and set(low) <= set(high)
    # Spaced about BRACKET_SPACING_M apart along the middle row.
    mid = np.sort(bearing[np.abs(r - (R0 + 20 * GAP)) < 0.5])
    spacing = np.radians(np.diff(mid)) * (R0 + 20 * GAP)
    assert np.median(spacing) == pytest.approx(BRACKET_SPACING_M, rel=0.2)


# ---- scan cut ----------------------------------------------------------------

def fade_inputs():
    alpha = np.ones((60, 60))  # 2 m texels, row 0 = north
    place = {"x0_cm": -2000.0, "y1_cm": 6000.0, "width_cm": 12000.0, "height_cm": 12000.0,
             "texels": [60, 60], "fade_m": 30.0, "close_m": 6.0}
    return alpha, place


def texel(alpha, place, e, n):
    ny, nx = alpha.shape
    j = int((e * 100 - place["x0_cm"]) / place["width_cm"] * nx)
    i = int((place["y1_cm"] - n * 100) / place["height_cm"] * ny)
    return alpha[i, j]


def test_scan_is_cut_out_over_the_terraces_only():
    rows, lines = rows_and_lines()
    layout = seating_layout(rows, lines, ground=bowl, clearance_m=0.45)
    alpha, place = fade_inputs()
    out, meta = seating_fade(alpha, place, layout)
    assert meta["texels"] == [int(round(12000 / (CUT_TEXEL_M * 100)))] * 2
    assert meta["x0_cm"] == place["x0_cm"] and meta["width_cm"] == place["width_cm"]
    mid = R0 + 20 * GAP + GAP / 2
    assert texel(out, meta, mid, 0.0) == 0.0  # inside the bowl
    assert texel(out, meta, R0 - 6.0, 0.0) == 1.0  # the apron in front of row 1
    assert texel(out, meta, mid * np.cos(np.radians(45)), mid * np.sin(np.radians(45))) == 1.0  # past the row ends


def test_inside_the_seating_the_scan_is_always_cut():
    rows, lines = rows_and_lines()
    layout = seating_layout(rows, lines, ground=bowl, clearance_m=0.45)
    alpha, place = fade_inputs()
    out, meta = seating_fade(alpha, place, layout)
    mid = R0 + 20 * GAP + GAP / 2
    assert texel(out, meta, mid, 0.0) == 0.0 and "kept_tall_m2" not in meta["seating_cut"]


# ---- the build -----------------------------------------------------------------

def _cli_build(monkeypatch, tmp_path, with_fade=True):
    import json

    from PIL import Image

    from redrocks import cli
    from redrocks.dem import Dem
    from redrocks.osm import OsmData, Way

    def fake_dem(bbox, size=2017):
        data = np.linspace(1900.0, 1950.0, 16 * 16, dtype=np.float32).reshape(16, 16)
        return Dem(data=data, bbox=bbox)

    lat, lon = cli.RED_ROCKS_LAT, cli.RED_ROCKS_LON
    ring = [(lat - 0.0012, lon - 0.0008), (lat - 0.0012, lon + 0.0008),
            (lat + 0.0012, lon + 0.0008), (lat + 0.0012, lon - 0.0008)]
    # The sound booth's roof, a 6 x 4.5 m footprint in the middle of the bowl.
    booth = [(lat - 0.00002, lon - 0.000035), (lat - 0.00002, lon + 0.000035),
             (lat + 0.00002, lon + 0.000035), (lat + 0.00002, lon - 0.000035)]
    nodes = {i + 1: p for i, p in enumerate(ring + booth)}
    osm = OsmData(nodes=nodes, ways=[
        Way(id=1, node_ids=[1, 2, 3, 4, 1], tags={"amenity": "theatre"}),
        Way(id=229359709, node_ids=[5, 6, 7, 8, 5], tags={"building": "roof", "layer": "1"}),
    ])
    monkeypatch.setattr(cli, "fetch_dem", fake_dem)
    monkeypatch.setattr(cli, "fetch_osm", lambda bbox: osm)
    monkeypatch.setattr(cli, "build_monoliths", lambda *a, **k: [])

    xs = np.arange(-200.0, 200.0) + 0.5
    e, n = np.meshgrid(xs, xs)
    plane = 1955.0 + 0.05 * e - 0.2 * n  # rising south, like the fake DEM's rows
    ScanHeights(plane - 0.3, -200.0, -200.0, 1.0).save(tmp_path / "floor.npz")
    ScanHeights(plane, -200.0, -200.0, 1.0).save(tmp_path / "ground.npz")
    ScanHeights(plane + 0.2, -200.0, -200.0, 1.0).save(tmp_path / "top.npz")
    Image.fromarray(np.full((200, 200, 3), 255, np.uint8)).save(tmp_path / "fade.png")
    (tmp_path / "fade.json").write_text(json.dumps({
        "x0_cm": -20000.0, "y1_cm": 20000.0, "width_cm": 40000.0, "height_cm": 40000.0,
        "texels": [200, 200], "fade_m": 30.0, "close_m": 6.0}))
    out = tmp_path / "out"
    args = ["build", "--out", str(out), "--side-m", "2400", "--landscape-size", "33", "--terrain-grid", "65",
            "--scan-heights", str(tmp_path / "floor.npz"), "--scan-ground", str(tmp_path / "ground.npz"),
            "--scan-top", str(tmp_path / "top.npz"),
            "--scan-fade", str(tmp_path / "fade.png") if with_fade else ""]
    assert cli.main(args) == 0
    return out, json.loads((out / "manifest.json").read_text())


def test_build_writes_the_seating_meshes_and_their_manifest(monkeypatch, tmp_path):
    out, manifest = _cli_build(monkeypatch, tmp_path)
    b = manifest["benches"]
    for key, name in (("obj", "Benches"), ("planks_obj", "Bench_Planks"), ("brackets_obj", "Bench_Brackets")):
        assert b[key] == f"{name}.obj" and (out / b[key]).exists()
    assert b["rows"] == b["pieces"] == manifest["venue"]["rows"] == 70  # rows run unbroken
    assert len(b["tread_levels_m"]) == 70 and b["clearance_m"] == pytest.approx(0.45)
    assert b["rows_trimmed"] == 0  # nothing stands in this fake bowl's aisles
    assert np.all(np.diff(b["tread_levels_m"]) >= MIN_RISE_M - 1e-6)
    assert manifest["handrail"]["obj"] == "Handrail.obj" and (out / "Handrail.obj").exists()
    assert manifest["stairs"]["obj"] == "Stairs.obj" and manifest["stairs"]["parts"] == 3 * 2 * 70
    # The sound booth stands on its OSM roof footprint, with its roof in Booth.obj.
    assert manifest["booth"]["osm_way"] == 229359709 and (out / "Booth.obj").exists()
    assert manifest["booth"]["deck_elevation_m"] > b["tread_levels_m"][0]
    assert abs(manifest["booth"]["east_m"]) < 5.0 and abs(manifest["booth"]["north_m"]) < 5.0
    # The stage's walkway sits just under the top tread.
    assert manifest["stage"]["walkway_elevation_m"] == pytest.approx(b["tread_levels_m"][-1] - 0.02, abs=1e-3)


def test_build_cuts_the_seating_out_of_the_scan_fade(monkeypatch, tmp_path):
    import json

    from PIL import Image

    out, manifest = _cli_build(monkeypatch, tmp_path)
    meta = json.loads((out / "scan_fade.json").read_text())
    alpha = np.asarray(Image.open(out / "scan_fade.png").convert("L"))
    assert meta["texels"] == [alpha.shape[1], alpha.shape[0]] == [800, 800]
    assert (alpha == 0).sum() * CUT_TEXEL_M ** 2 == pytest.approx(meta["seating_cut"]["cut_m2"], rel=0.01)
    assert manifest["scan_cut"]["cut_m2"] > 1000
    # No fade mask given: no cut, and an old one is not left behind.
    out, manifest = _cli_build(monkeypatch, tmp_path, with_fade=False)
    assert not (out / "scan_fade.png").exists() and "scan_cut" not in manifest


# ---- rows end where the aisle has room ------------------------------------------

def test_rows_stop_short_of_an_obstacle_at_their_end():
    from shapely.geometry import Point, box

    from redrocks.seating import AISLE_CLEAR_M, AISLE_HALF_WIDTH_M, trim_row_ends

    rows, lines = rows_and_lines()
    # A tower over the start end of rows 10-20 (the arcs start at -30 degrees).
    start = lines[15][0]
    tower = box(start[0] - 3.0, start[1] - 6.0, start[0] + 3.0, start[1] + 6.0)
    trimmed = trim_row_ends(lines, tower)
    for k, (old, new) in enumerate(zip(lines, trimmed)):
        assert np.allclose(new[-1], old[-1])  # the far end is clear: untouched
        if abs(k - 15) > 20:  # beyond the taper's reach (redrocks.seating.TRIM_TAPER_M)
            assert np.allclose(new[0], old[0], atol=1e-6)
        elif tower.distance(Point(*old[0])) < AISLE_HALF_WIDTH_M:
            # Shorter, and the aisle beside the new end clears the tower.
            assert np.hypot(*(new[0] - old[0])) > 0.5
            assert tower.distance(Point(*new[0])) >= AISLE_HALF_WIDTH_M - 0.3
            # Still on the row's arc.
            assert np.hypot(*new[0]) == pytest.approx(R0 + k * GAP, abs=0.01)


def test_trimming_never_takes_more_than_part_of_a_row():
    from shapely.geometry import box

    from redrocks.seating import MAX_TRIM_FRACTION, trim_row_ends

    rows, lines = rows_and_lines()
    everything = box(-1000, -1000, 1000, 1000)
    trimmed = trim_row_ends(lines, everything)
    for old, new in zip(lines, trimmed):
        lo = float(np.hypot(*np.diff(old, axis=0).T).sum())
        ln = float(np.hypot(*np.diff(new, axis=0).T).sum())
        assert ln >= lo * (1 - 2 * MAX_TRIM_FRACTION) - 0.3


def test_no_obstacles_no_trim():
    from redrocks.seating import trim_row_ends

    rows, lines = rows_and_lines()
    assert all(np.array_equal(a, b) for a, b in zip(lines, trim_row_ends(lines, None)))


def test_obstacles_are_buildings_and_whatever_stands_tall_in_the_scan():
    from shapely.geometry import Point, box

    from redrocks.seating import seating_obstacles

    xs = np.arange(0.0, 50.0) + 0.5
    e, n = np.meshgrid(xs, xs)
    ground = np.full(e.shape, 100.0)
    top = ground + 0.3
    top[(np.abs(e - 10) < 2) & (np.abs(n - 10) < 2)] = 104.0  # a rock fin
    tower = box(30, 30, 35, 36)
    obs = seating_obstacles(ScanHeights(top, 0.0, 0.0, 1.0), ScanHeights(ground, 0.0, 0.0, 1.0), [tower])
    assert obs.contains(Point(10, 10)) and obs.contains(Point(32, 33))
    assert not obs.contains(Point(20, 20))


def test_one_rows_trim_tapers_into_its_neighbours():
    """A rock cutting one row back leaves no notch: the rows beside it step in
    gradually, so their ends (and the aisle stairs on them) make a smooth line."""
    from shapely.geometry import Point

    from redrocks.seating import TRIM_TAPER_M, trim_row_ends

    rows, lines = rows_and_lines()
    k = 20
    rock = Point(*lines[k][0]).buffer(0.4)  # small: only row 20's aisle hits it
    trimmed = trim_row_ends(lines, rock)
    starts = []
    for old, new in zip(lines, trimmed):
        a0 = np.arctan2(old[0, 1], old[0, 0])
        a1 = np.arctan2(new[0, 1], new[0, 0])
        starts.append((a1 - a0) * np.hypot(*old[0]))  # metres trimmed off the start
    starts = np.array(starts)
    assert starts[k] > 1.0
    assert np.abs(np.diff(starts)).max() <= TRIM_TAPER_M + 0.05
    assert starts[0] == pytest.approx(0.0, abs=1e-6) and starts[-1] == pytest.approx(0.0, abs=1e-6)


