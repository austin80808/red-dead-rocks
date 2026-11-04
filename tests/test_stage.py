import json
from collections import Counter

import numpy as np
import pytest

from redrocks import cli, stage
from redrocks.dem import Dem
from redrocks.osm import OsmData, Way
from redrocks.stage import (
    DIMENSIONS,
    ROOF_TOP_ABOVE_DECK_M,
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


def test_stage_clear_of_first_bench(tmp_path):
    from redrocks.benches import SEAT_DEPTH_M

    mesh, (verts, _, faces, _), rows = export(tmp_path)
    deck = part_verts(mesh, verts, faces, "deck")
    assert deck[:, 1].max() / 100 < rows[0].centre_north - SEAT_DEPTH_M / 2 - 1.0


def test_roof_top_is_published_height_above_deck(tmp_path):
    mesh, (verts, _, faces, _), rows = export(tmp_path)
    roof = part_verts(mesh, verts, faces, "roof")
    deck_top = (rows[0].elevation_m - BASE - STAGE_DROP_M) * 100
    assert roof[:, 2].max() - deck_top == pytest.approx(ROOF_TOP_ABOVE_DECK_M * 100, abs=0.01)
    assert ROOF_TOP_ABOVE_DECK_M == pytest.approx(12.24, abs=0.01)  # 40'2"


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
        assert d.kind in {"published", "derived", "estimate"}, d.name
        assert len(d.source) > 10, d.name
        assert d.value_m > 0, d.name
    # Stage width/depth are not published anywhere we found; say so honestly.
    assert next(d for d in DIMENSIONS if d.name == "stage_width").kind == "estimate"
    assert dim("column_rear_station") == pytest.approx(62 * 0.3048)
    assert stage.dimensions_table().count("\n") == len(DIMENSIONS) + 1


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
