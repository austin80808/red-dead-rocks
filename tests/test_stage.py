import json
from collections import Counter

import numpy as np
import pytest

from redrocks import cli, stage
from redrocks.dem import Dem
from redrocks.osm import OsmData, Way
from redrocks.stage import (
    DIMENSIONS,
    ROOF_GRID_ABOVE_DECK_M,
    STAGE_DEPTH_M,
    STAGE_DROP_M,
    STAGE_FRONT_SETBACK_M,
    STAGE_WIDTH_M,
    dim,
    export_stage_obj,
    prism,
    stage_axis,
    stage_parts,
)
from redrocks.venue import Row

BASE = 1750.0


def make_rows(n=68, spacing=1.0, rise=0.4355, half_width=40.0, north0=100.0, east0=0.0):
    rows, lines = [], []
    for i in range(n):
        north = north0 + i * spacing
        rows.append(Row(index=i + 1, seats=100, length_m=2 * half_width, elevation_m=1800.0 + i * rise,
                        centre_east=east0, centre_north=north))
        lines.append(np.array([[east0 - half_width, north], [east0 + half_width, north]]))
    return rows, lines


def load(path):
    verts, uvs, faces, objects = [], [], [], []
    for line in path.read_text().splitlines():
        tok = line.split()
        if not tok or tok[0].startswith("#"):
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


def export(tmp_path, rows=None, lines=None):
    if rows is None:
        rows, lines = make_rows()
    mesh = export_stage_obj(rows, lines, tmp_path / "Stage.obj", BASE)
    return mesh, load(tmp_path / "Stage.obj"), rows


def part_verts(mesh, verts, faces, name):
    a, b = mesh.parts[name]
    return verts[sorted({i - 1 for f in faces[a:b] for i, _ in f})]


def test_single_object_and_parts(tmp_path):
    mesh, (verts, uvs, faces, objects), _ = export(tmp_path)
    assert objects == ["Stage"]
    for expected in ("deck", "roof", "shell_back", "retaining_wall", "walkway"):
        assert expected in mesh.parts
    assert sum(1 for p in mesh.parts if p.startswith("column_")) == stage.COLUMN_COUNT
    assert mesh.triangles == len(faces)
    assert mesh.vertices == len(verts)
    assert sum(b - a for a, b in mesh.parts.values()) == len(faces)


def test_watertight_and_closed_manifold(tmp_path):
    _, (verts, _, faces, _), _ = export(tmp_path)
    directed = Counter()
    for f in faces:
        ids = [i for i, _ in f]
        for a, b in zip(ids, ids[1:] + ids[:1]):
            directed[(a, b)] += 1
    assert all(c == 1 for c in directed.values())
    assert all(directed.get((b, a)) == 1 for a, b in directed)


def test_every_part_outward_facing_ccw(tmp_path):
    mesh, (verts, _, faces, _), _ = export(tmp_path)
    for name, (a, b) in mesh.parts.items():
        fs = faces[a:b]
        centre = part_verts(mesh, verts, faces, name).mean(axis=0)  # every part is convex
        vol = 0.0
        for f in fs:
            p, q, r = (verts[i - 1] for i, _ in f)
            assert np.cross(q - p, r - p) @ (p - centre) > 0, name
            vol += p @ np.cross(q, r) / 6.0
        assert vol > 0, name


def test_every_face_has_valid_uv(tmp_path):
    _, (verts, uvs, faces, _), _ = export(tmp_path)
    assert len(uvs) > 0
    for f in faces:
        assert len(f) == 3
        for v, vt in f:
            assert 1 <= v <= len(verts)
            assert 1 <= vt <= len(uvs)


def test_no_degenerate_triangles(tmp_path):
    _, (verts, _, faces, _), _ = export(tmp_path)
    for f in faces:
        p, q, r = (verts[i - 1] for i, _ in f)
        assert np.linalg.norm(np.cross(q - p, r - p)) > 1.0  # cm^2


def test_deck_elevation_is_row_one_minus_stage_drop(tmp_path):
    mesh, (verts, _, faces, _), rows = export(tmp_path)
    top = part_verts(mesh, verts, faces, "deck")[:, 2].max()
    assert top == pytest.approx((rows[0].elevation_m - BASE - STAGE_DROP_M) * 100, abs=0.01)
    assert mesh.deck_top_elevation_m == pytest.approx(rows[0].elevation_m - STAGE_DROP_M)


def test_stage_dimensions_and_position(tmp_path):
    # Bowl rises toward +north, so the stage sits south of row 1, centred on its east.
    mesh, (verts, _, faces, _), rows = export(tmp_path, *make_rows(east0=250.0))
    deck = part_verts(mesh, verts, faces, "deck")
    assert np.ptp(deck[:, 0]) == pytest.approx(STAGE_WIDTH_M * 100, abs=0.1)
    assert np.ptp(deck[:, 1]) == pytest.approx(STAGE_DEPTH_M * 100, abs=0.1)
    assert deck[:, 0].mean() == pytest.approx(25000.0, abs=0.1)
    assert deck[:, 1].max() == pytest.approx((100.0 - STAGE_FRONT_SETBACK_M) * 100, abs=0.1)
    assert (mesh.front_edge_east, mesh.front_edge_north) == pytest.approx((250.0, 100.0 - STAGE_FRONT_SETBACK_M))


def test_stage_clear_of_first_row(tmp_path):
    from redrocks.seating import PLANK_OVERHANG_M

    mesh, (verts, _, faces, _), rows = export(tmp_path)
    deck = part_verts(mesh, verts, faces, "deck")
    assert deck[:, 1].max() / 100 < rows[0].centre_north - PLANK_OVERHANG_M - 1.0


def test_roof_sits_on_published_grid_height_above_deck(tmp_path):
    mesh, (verts, _, faces, _), rows = export(tmp_path)
    roof = part_verts(mesh, verts, faces, "roof")
    deck_top = (rows[0].elevation_m - BASE - STAGE_DROP_M) * 100
    assert roof[:, 2].min() - deck_top == pytest.approx(ROOF_GRID_ABOVE_DECK_M * 100, abs=0.01)
    assert roof[:, 2].max() - roof[:, 2].min() == pytest.approx(dim("roof_thickness") * 100, abs=0.01)
    assert ROOF_GRID_ABOVE_DECK_M == pytest.approx(12.24, abs=0.01)  # 40'2"


def test_roof_is_shorter_than_the_deck_and_starts_at_its_front(tmp_path):
    mesh, (verts, _, faces, _), _ = export(tmp_path)
    deck = part_verts(mesh, verts, faces, "deck")
    roof = part_verts(mesh, verts, faces, "roof")
    assert np.ptp(roof[:, 1]) == pytest.approx(dim("roof_depth") * 100, abs=0.1)
    assert roof[:, 1].max() == pytest.approx(deck[:, 1].max(), abs=0.1)
    assert np.ptp(roof[:, 0]) == pytest.approx(STAGE_WIDTH_M * 100, abs=0.1)
    assert dim("roof_depth") < STAGE_DEPTH_M


def test_columns_sit_under_the_roof(tmp_path):
    mesh, (verts, _, faces, _), _ = export(tmp_path)
    roof = part_verts(mesh, verts, faces, "roof")
    for name in mesh.parts:
        if name.startswith("column_"):
            v = part_verts(mesh, verts, faces, name)
            assert v[:, 1].min() > roof[:, 1].min()
            assert v[:, 1].max() < roof[:, 1].max()


def test_columns_span_deck_to_roof(tmp_path):
    mesh, (verts, _, faces, _), _ = export(tmp_path)
    roof_bot = part_verts(mesh, verts, faces, "roof")[:, 2].min()
    deck_top = part_verts(mesh, verts, faces, "deck")[:, 2].max()
    for name in mesh.parts:
        if name.startswith("column_"):
            v = part_verts(mesh, verts, faces, name)
            assert v[:, 2].min() == pytest.approx(deck_top, abs=0.01)
            assert v[:, 2].max() == pytest.approx(roof_bot, abs=0.01)


def test_retaining_wall_is_behind_the_shell(tmp_path):
    mesh, (verts, _, faces, _), _ = export(tmp_path)

    def ys(name):
        return part_verts(mesh, verts, faces, name)[:, 1]

    assert ys("retaining_wall").max() < ys("shell_back").min()
    assert ys("shell_back").max() < ys("deck").max()


def test_walkway_is_behind_last_row_at_its_elevation(tmp_path):
    mesh, (verts, _, faces, _), rows = export(tmp_path)
    w = part_verts(mesh, verts, faces, "walkway")
    assert w[:, 2].max() == pytest.approx((rows[-1].elevation_m - BASE) * 100, abs=0.01)
    assert w[:, 1].min() > rows[-1].centre_north * 100  # clear of the last bench
    assert np.ptp(w[:, 0]) == pytest.approx(8000.0, abs=0.01)  # spans the last row's chord


def test_axis_follows_bowl_direction(tmp_path):
    rows, lines = make_rows(20)
    # Rotate the whole bowl 90 degrees: rises toward +east.
    for r in rows:
        r.centre_east, r.centre_north = r.centre_north, r.centre_east
    lines = [ln[:, ::-1] for ln in lines]
    assert stage_axis(rows) == pytest.approx([1.0, 0.0])
    mesh, (verts, _, faces, _), _ = export(tmp_path, rows, lines)
    deck = part_verts(mesh, verts, faces, "deck")
    assert deck[:, 0].max() == pytest.approx((100.0 - STAGE_FRONT_SETBACK_M) * 100, abs=0.1)
    assert np.ptp(deck[:, 1]) == pytest.approx(STAGE_WIDTH_M * 100, abs=0.1)


def test_every_dimension_has_a_source():
    names = [d.name for d in DIMENSIONS]
    assert len(names) == len(set(names))
    for d in DIMENSIONS:
        assert d.kind in {"published", "derived", "measured", "estimate"}, d.name
        assert len(d.source) > 10, d.name
        assert d.value_m > 0, d.name
        if d.kind == "measured":
            assert d.source.startswith("ml0w scan (CC BY 4.0), "), d.name
    assert dim("column_rear_station") == pytest.approx(62 * 0.3048)
    assert stage.dimensions_table().count("\n") == len(DIMENSIONS) + 1


def test_scan_measured_dimensions():
    kinds = {d.name: d.kind for d in DIMENSIONS}
    for name in ("stage_width", "stage_depth", "roof_depth", "roof_thickness", "retaining_wall_height"):
        assert kinds[name] == "measured", name
    # Published figures stay published, and the scan's roof underside agrees with the grid height to 1%.
    assert kinds["roof_grid_above_deck"] == "published"
    assert dim("roof_grid_above_deck") == pytest.approx(12.33, rel=0.01)
    # Things the scan could not fix stay honest estimates.
    for name in ("column_size", "shell_wing_length", "walkway_width", "stage_front_setback"):
        assert kinds[name] == "estimate", name
    # The scan-sized stage still fits the bowl: deck deeper than the roof, wall behind it.
    assert dim("roof_depth") < dim("stage_depth")
    assert 20 < dim("stage_width") < 32 and 20 < dim("stage_depth") < 35


def test_prism_supports_wedges_and_rejects_bad_input():
    tri = np.array([[0, 0], [4, 0], [0, 3]], float)
    for plan in (tri, tri[::-1]):  # either winding in
        verts, tris, uvs, tuvs = prism(plan, 0, 2)
        v = sum(verts[a] @ np.cross(verts[b], verts[c]) / 6.0 for a, b, c in tris)
        assert v == pytest.approx(0.5 * 4 * 3 * 2)
        assert len(tris) == 2 * 1 + 2 * 3
    with pytest.raises(ValueError):
        prism(np.array([[0, 0], [4, 0], [1, 1], [4, 4], [0, 4]], float), 0, 1)  # concave
    with pytest.raises(ValueError):
        prism(tri, 1, 1)


def test_mismatched_inputs_raise(tmp_path):
    rows, lines = make_rows(5)
    with pytest.raises(ValueError):
        export_stage_obj(rows, lines[:-1], tmp_path / "x.obj", BASE)
    with pytest.raises(ValueError):
        stage_parts(rows[:1], lines[:1], BASE)


def test_cli_build_writes_stage_and_manifest(monkeypatch, tmp_path):
    def fake_dem(bbox, size=2017):
        data = np.linspace(1900.0, 1950.0, 16 * 16, dtype=np.float32).reshape(16, 16)
        return Dem(data=data, bbox=bbox)

    lat, lon = cli.RED_ROCKS_LAT, cli.RED_ROCKS_LON
    ring = [(lat - 0.0012, lon - 0.0008), (lat - 0.0012, lon + 0.0008),
            (lat + 0.0012, lon + 0.0008), (lat + 0.0012, lon - 0.0008)]
    nodes = {i + 1: p for i, p in enumerate(ring)}
    osm = OsmData(nodes=nodes, ways=[Way(id=1, node_ids=[1, 2, 3, 4, 1], tags={"amenity": "theatre"})])
    monkeypatch.setattr(cli, "fetch_dem", fake_dem)
    monkeypatch.setattr(cli, "fetch_osm", lambda bbox: osm)
    monkeypatch.setattr(cli, "build_monoliths", lambda *a, **k: [])

    assert cli.main(["build", "--out", str(tmp_path), "--landscape-size", "33", "--terrain-grid", "9"]) == 0
    assert (tmp_path / "Stage.obj").exists()
    verts, uvs, faces, objects = load(tmp_path / "Stage.obj")
    assert objects == ["Stage"] and len(uvs) > 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    st = manifest["stage"]
    assert st["obj"] == "Stage.obj"
    assert st["triangles"] == len(faces)
    assert st["parts"]["deck"]["end_triangle"] > st["parts"]["deck"]["first_triangle"]
    assert {d["name"] for d in st["dimensions"]} == {d.name for d in DIMENSIONS}
    assert st["base_elevation_m"] == manifest["benches"]["base_elevation_m"]


def test_deck_rises_above_terrain_that_would_bury_it():
    from redrocks import stage as st
    from redrocks.venue import Row
    import inspect
    fields = inspect.signature(Row).parameters
    def row(e, n, z):
        kw = {k: 0 for k in fields}
        kw.update(centre_east=e, centre_north=n, elevation_m=z)
        return Row(**kw)
    rows = [row(0.0, 0.0, 100.0), row(0.0, 50.0, 130.0)]
    flat = st.deck_top_elevation(rows)
    assert flat == 100.0 - st.STAGE_DROP_M
    buried = st.deck_top_elevation(rows, ground=lambda e, n: 101.0)
    assert buried == pytest.approx(101.0 + st.DECK_CLEARANCE_M)
    # With a ground, row 1's tread is the ground under it (where its bench
    # stands), not the row's DEM elevation: the deck follows the ground down.
    low = st.deck_top_elevation(rows, ground=lambda e, n: 50.0)
    assert low == pytest.approx(50.0 + st.DECK_CLEARANCE_M)


def test_deck_sits_on_the_scanned_deck_not_on_its_side_walls():
    """A few tall spots along the deck's sides (the scan's real stage wings) do not lift it."""
    rows, lines = make_rows()

    def ground(e, n):
        side = abs(e) > STAGE_WIDTH_M / 2.0 - 2.5  # two metres or so along each side
        return 1805.0 if side and n < 100.0 else 1799.6

    top = stage.deck_top_elevation(rows, ground)
    assert top == pytest.approx(1799.6 + stage.DECK_CLEARANCE_M)
    # Ground everywhere that high would lift it.
    assert stage.deck_top_elevation(rows, lambda e, n: 1805.0) == pytest.approx(1805.0 + stage.DECK_CLEARANCE_M)


def test_walkway_sits_on_the_ground_when_given_one(tmp_path):
    rows, lines = make_rows()
    tilt = lambda e, n: 1830.0 + 0.01 * e  # noqa: E731  a little cross-slope under the walkway
    mesh = export_stage_obj(rows, lines, tmp_path / "Stage.obj", BASE, ground=tilt)
    verts, _, faces, _ = load(tmp_path / "Stage.obj")
    w = part_verts(mesh, verts, faces, "walkway") / 100.0 + np.array([0, 0, BASE])
    # Top clears most of the ground under it; bottom reaches its lowest point.
    assert 1830.0 < w[:, 2].max() < 1830.0 + 0.4 + stage.WALKWAY_CLEARANCE_M
    assert w[:, 2].min() <= 1830.0 - 0.4 + 1e-6
    assert mesh.walkway_elevation_m == pytest.approx(w[:, 2].max(), abs=1e-3)


def test_cli_stands_benches_stage_and_spawn_on_the_scan_ground(monkeypatch, tmp_path):
    """With a scan ground raster, the treads clear it and the deck and spawn stand on it."""
    from redrocks.scan_blend import ScanHeights

    def fake_dem(bbox, size=2017):
        data = np.linspace(1900.0, 1950.0, 16 * 16, dtype=np.float32).reshape(16, 16)
        return Dem(data=data, bbox=bbox)

    lat, lon = cli.RED_ROCKS_LAT, cli.RED_ROCKS_LON
    ring = [(lat - 0.0012, lon - 0.0008), (lat - 0.0012, lon + 0.0008),
            (lat + 0.0012, lon + 0.0008), (lat + 0.0012, lon - 0.0008)]
    nodes = {i + 1: p for i, p in enumerate(ring)}
    osm = OsmData(nodes=nodes, ways=[Way(id=1, node_ids=[1, 2, 3, 4, 1], tags={"amenity": "theatre"})])
    monkeypatch.setattr(cli, "fetch_dem", fake_dem)
    monkeypatch.setattr(cli, "fetch_osm", lambda bbox: osm)
    monkeypatch.setattr(cli, "build_monoliths", lambda *a, **k: [])

    def plane(e, n):
        # Rising to the south like the fake DEM, so the scan's bowl climbs with the rows.
        return 1955.0 + 0.05 * e - 0.2 * n

    def raster(cell, lift):
        xs = np.arange(-200.0, 200.0, cell) + cell / 2
        e, n = np.meshgrid(xs, xs)
        return ScanHeights(heights=plane(e, n) + lift, x0=-200.0, y0=-200.0, cell=cell)

    raster(2.0, -0.3).save(tmp_path / "floor.npz")
    raster(1.0, 0.0).save(tmp_path / "ground.npz")
    out = tmp_path / "out"
    args = ["build", "--out", str(out), "--side-m", "2400", "--landscape-size", "33", "--terrain-grid", "65",
            "--scan-heights", str(tmp_path / "floor.npz")]
    assert cli.main(args + ["--scan-ground", str(tmp_path / "ground.npz")]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["benches"]["ground"] == "ml0w scan (ground.npz)"
    assert manifest["venue"]["seats"] == 9525

    base = manifest["benches"]["base_elevation_m"]
    bverts, _, bfaces, _ = load(out / "Benches.obj")
    m = bverts / 100.0
    z = m[:, 2] + base - plane(m[:, 0], m[:, 1])
    levels = np.array(manifest["benches"]["tread_levels_m"])
    tops = np.isin(np.round(m[:, 2] + base, 3), np.round(levels, 3))
    # Treads are level along their rows and clear the scan's ground by the
    # clearance where they sit (this fake plane also slopes along the rows), and
    # the terraces' buried bottoms go under it.
    assert np.median(z[tops]) == pytest.approx(0.45, abs=0.3)
    assert z[~tops].max() < z[tops].max() and np.median(z[~tops]) < 0.0

    deck = manifest["stage"]["deck_top_elevation_m"]
    spawn = manifest["spawn"]
    assert spawn["ground_elevation_m"] == pytest.approx(deck)
    under_spawn = plane(spawn["east_m"], spawn["north_m"])
    assert 0.0 < deck - under_spawn < 2.0  # on the scan's ground, not the DEM's

    # Without the ground raster everything goes back onto the terrain mesh.
    assert cli.main(args + ["--scan-ground", ""]) == 0
    assert json.loads((out / "manifest.json").read_text())["benches"]["ground"] == "terrain mesh"


def test_y_columns_and_skydeck_trusses_match_published_stations(tmp_path):
    mesh, (verts, _, faces, _), rows = export(tmp_path)
    deck_top = (rows[0].elevation_m - BASE - STAGE_DROP_M) * 100.0
    roof_bot = part_verts(mesh, verts, faces, "roof")[:, 2].min()
    fork_z = deck_top + stage.Y_COLUMN_FORK_ABOVE_DECK_M * 100.0

    y_arms = [name for name in mesh.parts if name.startswith("y_arm_")]
    assert len(y_arms) == 8
    for name in y_arms:
        v = part_verts(mesh, verts, faces, name)
        assert v[:, 2].min() == pytest.approx(fork_z, abs=0.05)
        assert v[:, 2].max() == pytest.approx(roof_bot, abs=0.05)

    for tname in ("truss_1", "truss_2", "truss_3"):
        assert tname in mesh.parts
        tv = part_verts(mesh, verts, faces, tname)
        assert tv[:, 2].max() == pytest.approx(roof_bot, abs=0.05)
        assert tv[:, 2].min() == pytest.approx(roof_bot - dim("truss_depth") * 100.0, abs=0.05)

    beams = [name for name in mesh.parts if name.startswith("skydeck_beam_")]
    assert len(beams) == 6
    assert "shell_canopy" in mesh.parts
    assert "shell_baffle_e" in mesh.parts and "shell_baffle_w" in mesh.parts


def stair_verts(mesh, name):
    """A row-end stair part's vertices, from Stairs.obj (they are not in Stage.obj)."""
    verts, _, faces, _ = load(mesh.stairs_path)
    a, b = mesh.stairs_parts[name]
    return verts[sorted({i - 1 for f in faces[a:b] for i, _ in f})]


def test_row_end_stairs_are_their_own_mesh(tmp_path):
    """They take the seating's flagstone, so they leave the concrete Stage.obj."""
    mesh, (verts, _, faces, objects), rows = export(tmp_path)
    assert not any(p.startswith("stair_") for p in mesh.parts)
    sv, suv, sfaces, sobjects = load(tmp_path / "Stairs.obj")
    assert mesh.stairs_path == tmp_path / "Stairs.obj" and sobjects == ["Stairs"]
    assert len(mesh.stairs_parts) == 3 * 2 * len(rows)
    assert sum(b - a for a, b in mesh.stairs_parts.values()) == len(sfaces) == mesh.stairs_triangles
    assert all(1 <= t <= len(suv) for f in sfaces for _, t in f)


def test_foh_booth_and_end_of_row_stairs(tmp_path):
    mesh, (verts, _, faces, _), rows = export(tmp_path)
    for part_name in ("foh_platform", "foh_wall_front", "foh_wall_e", "foh_wall_w", "walkway_parapet"):
        assert part_name in mesh.parts

    # Every row has 2 steps at both ends (start and end) plus outer cheek walls.
    for r in rows:
        for side in ("start", "end"):
            s1 = stair_verts(mesh, f"stair_{side}_r{r.index}_s1")
            s2 = stair_verts(mesh, f"stair_{side}_r{r.index}_s2")
            wall = stair_verts(mesh, f"stair_wall_{side}_r{r.index}")
            # Step 2 rises half a bench height (22.5 cm) above step 1.
            assert s2[:, 2].max() - s1[:, 2].max() == pytest.approx(22.5, abs=0.05)
            assert wall[:, 2].max() > s2[:, 2].max()


# ---- with the terraced seating's tread levels ----------------------------------

def terraced(tmp_path, n=10):
    rows, lines = make_rows(n)
    levels = [1800.5 + 0.45 * i for i in range(n)]
    mesh = export_stage_obj(rows, lines, tmp_path / "Stage.obj", BASE, levels=levels)
    verts, _, faces, _ = load(tmp_path / "Stage.obj")
    return mesh, verts, faces, rows, levels


def top(mesh, verts, faces, name):
    v = stair_verts(mesh, name) if name.startswith("stair_") else part_verts(mesh, verts, faces, name)
    return v[:, 2].max() / 100.0 + BASE


def test_side_stairs_climb_from_tread_to_tread(tmp_path):
    mesh, verts, faces, rows, levels = terraced(tmp_path)
    for k in range(1, len(rows)):
        r = rows[k]
        for side in ("start", "end"):
            assert top(mesh, verts, faces, f"stair_{side}_r{r.index}_s1") == pytest.approx(
                (levels[k - 1] + levels[k]) / 2, abs=1e-3)
            assert top(mesh, verts, faces, f"stair_{side}_r{r.index}_s2") == pytest.approx(levels[k], abs=1e-3)
            assert top(mesh, verts, faces, f"stair_wall_{side}_r{r.index}") == pytest.approx(
                levels[k] + dim("stair_wall_height"), abs=1e-3)
            # Each pair of steps fills the row's band: its line back to the next row's.
            s = np.vstack([stair_verts(mesh, f"stair_{side}_r{r.index}_s{j}") for j in (1, 2)])
            assert s[:, 1].min() / 100 == pytest.approx(r.centre_north, abs=1e-3)
            assert s[:, 1].max() / 100 == pytest.approx(r.centre_north + 1.0, abs=1e-3)


def test_stage_steps_reach_row_one_tread(tmp_path):
    mesh, verts, faces, rows, levels = terraced(tmp_path)
    for side in ("e", "w"):
        assert top(mesh, verts, faces, f"stage_step_{side}_2") == pytest.approx(levels[0], abs=1e-3)


def test_walkway_sits_just_under_the_top_tread(tmp_path):
    mesh, verts, faces, rows, levels = terraced(tmp_path)
    assert top(mesh, verts, faces, "walkway") == pytest.approx(levels[-1] - stage.WALKWAY_UNDER_TREAD_M, abs=1e-3)
    assert mesh.walkway_elevation_m == pytest.approx(levels[-1] - stage.WALKWAY_UNDER_TREAD_M, abs=1e-3)


def test_levels_must_match_rows(tmp_path):
    rows, lines = make_rows(5)
    with pytest.raises(ValueError):
        export_stage_obj(rows, lines, tmp_path / "Stage.obj", BASE, levels=[1800.0] * 4)


def test_handrail_runs_along_the_walkway_parapet(tmp_path):
    mesh, verts, faces, rows, levels = terraced(tmp_path)
    hv, huv, hfaces, objects = load(tmp_path / "Handrail.obj")
    assert objects == ["Handrail"] and mesh.handrail_path == tmp_path / "Handrail.obj"
    walk = top(mesh, verts, faces, "walkway")
    parapet = part_verts(mesh, verts, faces, "walkway_parapet")
    assert hv[:, 2].max() / 100 + BASE == pytest.approx(walk + stage.HANDRAIL_HEIGHT_M, abs=1e-3)
    assert hv[:, 2].min() == pytest.approx(parapet[:, 2].max(), abs=0.01)  # standing on the parapet
    # Over the parapet in plan, end to end.
    assert hv[:, 1].min() >= parapet[:, 1].min() - 1e-3 and hv[:, 1].max() <= parapet[:, 1].max() + 1e-3
    assert np.ptp(hv[:, 0]) == pytest.approx(np.ptp(parapet[:, 0]), abs=0.1)
    posts = sum(1 for line in (tmp_path / "Handrail.obj").read_text().splitlines() if line.startswith("# post"))
    assert posts == int(np.ptp(parapet[:, 0]) / 100 // stage.HANDRAIL_POST_SPACING_M) + 1
    assert all(1 <= t <= len(huv) for f in hfaces for _, t in f)


# ---- the sound booth from its OSM roof -------------------------------------------

def booth_rect(east=0.0, north=104.0, w=6.0, d=4.5):
    """A 6 x 4.5 m roof footprint over rows 5-9 of make_rows (rows run east-west, uphill north)."""
    return np.array([[east - w / 2, north], [east + w / 2, north], [east + w / 2, north + d], [east - w / 2, north + d]])


def test_booth_stands_on_its_osm_footprint_over_the_rows(tmp_path):
    rows, lines = make_rows(10)
    levels = [1800.5 + 0.45 * i for i in range(10)]
    plan = booth_rect()
    mesh = export_stage_obj(rows, lines, tmp_path / "Stage.obj", BASE, levels=levels, booth=plan)
    verts, _, faces, _ = load(tmp_path / "Stage.obj")
    deck = part_verts(mesh, verts, faces, "foh_platform")
    assert deck[:, 0].min() / 100 == pytest.approx(-3.0, abs=1e-3) and deck[:, 0].max() / 100 == pytest.approx(3.0, abs=1e-3)
    assert deck[:, 1].min() / 100 == pytest.approx(104.0, abs=1e-3) and deck[:, 1].max() / 100 == pytest.approx(108.5, abs=1e-3)
    # Level with the highest tread under it, plus a step: entered from the row behind.
    top = deck[:, 2].max() / 100 + BASE
    assert top == pytest.approx(levels[8] + stage.BOOTH_RAISE_M, abs=1e-3)
    assert mesh.booth_deck_elevation_m == pytest.approx(top, abs=1e-3)
    # The parapet is on the stage side (south here).
    front = part_verts(mesh, verts, faces, "foh_wall_front")
    assert front[:, 1].max() / 100 < 104.0 + 0.5
    assert front[:, 2].max() / 100 + BASE == pytest.approx(top + dim("foh_wall_height"), abs=1e-3)


def test_booth_roof_is_its_own_mesh_on_four_posts(tmp_path):
    rows, lines = make_rows(10)
    levels = [1800.5 + 0.45 * i for i in range(10)]
    mesh = export_stage_obj(rows, lines, tmp_path / "Stage.obj", BASE, levels=levels, booth=booth_rect())
    assert mesh.booth_path == tmp_path / "Booth.obj"
    v, uv, f, objects = load(tmp_path / "Booth.obj")
    assert objects == ["Booth"] and all(1 <= t <= len(uv) for face in f for _, t in face)
    text = (tmp_path / "Booth.obj").read_text()
    assert text.count("# post") == 4 and "# roof" in text
    deck_top = (mesh.booth_deck_elevation_m - BASE) * 100
    assert v[:, 2].min() == pytest.approx(deck_top, abs=0.01)
    assert v[:, 2].max() == pytest.approx(deck_top + (stage.BOOTH_ROOF_M + stage.BOOTH_ROOF_THICK_M) * 100, abs=0.01)
    # The roof overhangs the deck all round.
    assert v[:, 0].min() / 100 == pytest.approx(-3.0 - stage.BOOTH_ROOF_OVERHANG_M, abs=1e-3)


def test_without_a_footprint_the_booth_stays_a_blockout_and_has_no_roof(tmp_path):
    rows, lines = make_rows(10)
    mesh = export_stage_obj(rows, lines, tmp_path / "Stage.obj", BASE)
    assert "foh_platform" in mesh.parts and mesh.booth_path is None
    assert not (tmp_path / "Booth.obj").exists()


def test_aisles_list_every_stair_step_up_both_sides(tmp_path):
    """The play-test (and anyone scripting a walk) follows these up the row-end stairs."""
    mesh, verts, faces, rows, levels = terraced(tmp_path)
    for side in ("start", "end"):
        steps = mesh.aisles[side]
        assert len(steps) == 2 * len(rows)
        z = [p[2] for p in steps]
        assert z == sorted(z)  # each step a little higher than the one before
        assert z[1] == pytest.approx(levels[0], abs=1e-3) and z[-1] == pytest.approx(levels[-1], abs=1e-3)
        # Beside the rows' ends: past the 40 m half-width, on this side.
        xs = [p[0] for p in steps]
        assert all(abs(x) > 40.0 for x in xs) and len({np.sign(x) for x in xs}) == 1


def test_aisle_path_turns_a_corner_where_the_rows_widen(tmp_path):
    """Rows that widen put the next stair further out; heading straight there would
    cut across this row's band through its stair's cheek wall, so the path first
    steps up onto the next row's tread, then walks out along it to that stair."""
    rows, _ = make_rows(6)
    lines = [np.array([[-10.0 - 2.0 * i, 100.0 + i], [10.0 + 2.0 * i, 100.0 + i]]) for i in range(6)]
    levels = [1800.5 + 0.45 * i for i in range(6)]
    mesh = export_stage_obj(rows, lines, tmp_path / "Stage.obj", BASE, levels=levels)
    for side, path in mesh.aisles.items():
        assert [p[2] for p in path] == sorted(p[2] for p in path)
        # A corner on each of rows 2-6's treads (inside the row, not on its stair)
        # stands in for that row's first stair step, which the path goes past.
        on_tread = [p for p in path if abs(p[0]) < 10.0 + 2.0 * np.floor(p[1] - 100.0)]
        assert len(on_tread) == 5 and len(path) == 2 * 6
        for a, b in zip(path, path[1:]):
            if abs(b[0]) > abs(a[0]) + 0.3:  # an outward move stays within one row's band
                assert np.floor(a[1] - 100.0) == np.floor(b[1] - 100.0)


def test_aisle_path_keeps_to_the_inner_half_of_each_stair(tmp_path):
    """Away from the curb on the aisle's outer side, so a walker that runs wide on a turn stays on the stairs."""
    mesh, verts, faces, rows, levels = terraced(tmp_path)
    gap = stage.STAIR_GAP_M
    stair_outer = 40.0 + gap + dim("stair_width")  # make_rows' rows end at +-40 m
    for side, path in mesh.aisles.items():
        for e, n, z in path:
            assert 40.0 + gap < abs(e) < 40.0 + gap + dim("stair_width") / 2 + 1e-6
            assert stair_outer - abs(e) >= stage.AISLE_INSET_M + dim("stair_width") / 2 - 1e-6
