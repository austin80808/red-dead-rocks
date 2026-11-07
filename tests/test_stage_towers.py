"""The stage split by material (Stage.obj plus Stage_*.obj), its two stone towers,
side roofs and fascia, the scan cut over its own stage, and the Unreal wiring."""

import ast
import importlib.util
import json
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from redrocks import stage
from redrocks.scan_blend import ScanHeights
from redrocks.stage import (
    STAGE_MESHES, STAGE_WIDTH_M, dim, export_stage_obj, part_footprint, scan_stage_cut, stage_mesh_for,
    stage_parts,
)
from redrocks.venue import Row

ROOT = Path(__file__).resolve().parent.parent
BASE = 1750.0


def make_rows(n=40, spacing=1.0, rise=0.4355, half_width=40.0, north0=100.0):
    rows, lines = [], []
    for i in range(n):
        north = north0 + i * spacing
        rows.append(Row(index=i + 1, seats=100, length_m=2 * half_width, elevation_m=1800.0 + i * rise,
                        centre_east=0.0, centre_north=north))
        lines.append(np.array([[-half_width, north], [half_width, north]]))
    return rows, lines


def load(path):
    """(verts cm, faces as [(v, vt, vn?)...]) of an OBJ, 1-based indices."""
    verts, faces = [], []
    for line in Path(path).read_text().splitlines():
        tok = line.split()
        if tok and tok[0] == "v":
            verts.append([float(x) for x in tok[1:4]])
        elif tok and tok[0] == "f":
            faces.append([tuple(int(i) for i in t.split("/")) for t in tok[1:]])
    return np.array(verts), faces


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("stage")
    rows, lines = make_rows()
    mesh = export_stage_obj(rows, lines, out / "Stage.obj", BASE)
    return mesh, rows, lines


def parts_by_name(rows, lines):
    return {p.name: p for p in stage_parts(rows, lines, BASE)}


def test_every_stage_obj_is_closed_and_outward(built):
    mesh, _, _ = built
    objs = {"Stage": (mesh.path, mesh.parts)} | {k: (m.path, m.parts) for k, m in mesh.meshes.items()}
    assert set(objs) == set(STAGE_MESHES)
    for name, (path, ranges) in objs.items():
        verts, faces = load(path)
        assert sum(b - a for a, b in ranges.values()) == len(faces), name
        if name != "Stage":  # flat normals on the new OBJs: v/vt/vn
            assert all(len(t) == 3 for f in faces for t in f), name
        for part, (a, b) in ranges.items():
            fs = faces[a:b]
            ids = sorted({t[0] - 1 for f in fs for t in f})
            centre = verts[ids].mean(axis=0)  # every part is convex
            directed = Counter()
            vol = 0.0
            for f in fs:
                p, q, r = (verts[t[0] - 1] for t in f)
                assert np.linalg.norm(np.cross(q - p, r - p)) > 1.0, (name, part)  # no slivers (cm^2)
                assert np.cross(q - p, r - p) @ (p - centre) > 0, (name, part)
                vol += p @ np.cross(q, r) / 6.0
                vi = [t[0] for t in f]
                for u, v in zip(vi, vi[1:] + vi[:1]):
                    directed[(u, v)] += 1
            assert vol > 0, (name, part)
            assert all(c == 1 for c in directed.values()), (name, part)
            assert all(directed.get((v, u)) == 1 for u, v in directed), (name, part)


def test_steel_roof_fascia_and_stone_are_in_their_own_objs(built):
    mesh, _, _ = built
    steel = mesh.meshes["Stage_Steel"].parts
    for prefix, count in (("column_", 4), ("y_arm_", 8), ("truss_", 3), ("skydeck_beam_", 6),
                          ("wing_column_", 4), ("wing_header_", 4)):
        assert sum(1 for p in steel if p.startswith(prefix)) == count, prefix
    assert set(mesh.meshes["Stage_Roof"].parts) == {"roof", "roof_wing_left", "roof_wing_right"}
    assert set(mesh.meshes["Stage_Fascia"].parts) == {"fascia_front", "fascia_chamfer_left", "fascia_chamfer_right"}
    assert set(mesh.meshes["Stage_Towers"].parts) == {
        "tower_left_masonry", "tower_left_link", "tower_right_masonry", "tower_right_link",
        # the sound booth's flagstone base and parapet
        "foh_platform", "foh_wall_front", "foh_wall_e", "foh_wall_w"}
    # Stage.obj keeps only the concrete: the towers' bases and caps among it.
    assert all(stage_mesh_for(p) == "Stage" for p in mesh.parts)
    for p in ("deck", "shell_back", "retaining_wall", "walkway", "tower_left_base", "tower_left_cap",
              "tower_right_cap", "tower_right_pier_inner", "tower_right_pier_outer", "tower_right_arch_1"):
        assert p in mesh.parts, p
    assert stage_mesh_for("stair_start_r1_s1") == "Stairs"


def test_towers_stand_either_side_outside_the_deck_near_its_front(built):
    mesh, rows, lines = built
    parts = parts_by_name(rows, lines)
    deck = Polygon(parts["deck"].plan)
    hw = STAGE_WIDTH_M / 2
    origin = np.array([rows[0].centre_east, rows[0].centre_north]) - stage.stage_axis(rows) * stage.STAGE_FRONT_SETBACK_M
    a = stage.stage_axis(rows)
    l = np.array([a[1], -a[0]])
    for side, sign in (("left", 1), ("right", -1)):
        towers = [p for n, p in parts.items() if n.startswith(f"tower_{side}_")]
        assert len(towers) >= 4
        for p in towers:
            foot = part_footprint(p)
            assert not foot.intersects(deck), p.name
            xy = np.array(foot.exterior.coords)
            across = (xy - origin) @ l * sign
            along = (xy - origin) @ a
            assert across.min() > hw, p.name  # outside the deck's side edge, on its own side
            if not p.name.endswith("_link"):
                assert -10.0 < along.min() and along.max() < 5.0, p.name  # by the downstage end
        top = parts[f"tower_{side}_cap"].z1
        assert top == pytest.approx(parts["deck"].z1 + dim(f"tower_{side}_height"))
        # About the roof's top (eave) height: within 2 m of it.
        assert abs(top - parts["roof"].z1) < 2.0
        stone = parts[f"tower_{side}_masonry"]
        assert stone.z0 == pytest.approx(parts["deck"].z1 + dim("tower_base_height"))
        assert stone.z1 == pytest.approx(parts[f"tower_{side}_cap"].z0)


def test_the_right_towers_arch_is_a_real_opening(built):
    _, rows, lines = built
    parts = parts_by_name(rows, lines)
    a = stage.stage_axis(rows)
    l = np.array([a[1], -a[0]])
    origin = np.array([rows[0].centre_east, rows[0].centre_north]) - a * stage.STAGE_FRONT_SETBACK_M
    outer = dim("tower_right_outer")
    centre = -(outer - dim("tower_arch_outer_pier") - dim("tower_arch_width") / 2)
    springing = parts["deck"].z1 + dim("tower_arch_crown") - dim("tower_arch_width") / 2
    for y in (dim("tower_right_front") - 0.2, 0.0, -dim("tower_right_back") + 0.2):  # all the way through
        p = Point(*(origin + centre * l + y * a))
        for name in ("tower_right_pier_inner", "tower_right_pier_outer"):
            assert not Polygon(parts[name].plan).contains(p), name
    arch = [p for n, p in parts.items() if n.startswith("tower_right_arch_")]
    assert len(arch) == stage.ARCH_SEGMENTS
    lowest = min(stage.part_mesh(p)[0][:, 2].min() for p in arch)
    assert lowest == pytest.approx(springing, abs=1e-6)
    assert not any(n.startswith("tower_left_arch_") for n in parts)


def test_side_roofs_are_lower_than_the_roof_and_the_fascia_is_in_front(built):
    _, rows, lines = built
    parts = parts_by_name(rows, lines)
    roof, front = parts["roof"], parts["fascia_front"]
    for side in ("left", "right"):
        wing = parts[f"roof_wing_{side}"]
        assert wing.z1 == pytest.approx(parts["deck"].z1 + dim("roof_wing_top"))
        assert roof.z0 < wing.z1 < roof.z1 and wing.z0 > roof.z0
    assert (front.z0, front.z1) == (roof.z0, roof.z1)
    a = stage.stage_axis(rows)
    origin = np.array([rows[0].centre_east, rows[0].centre_north]) - a * stage.STAGE_FRONT_SETBACK_M
    along = (np.asarray(front.plan) - origin) @ a
    assert along.min() == pytest.approx(0.0, abs=1e-9) and along.max() == pytest.approx(dim("fascia_thickness"))


def test_towers_keep_clear_of_the_stairs_and_aisles(built):
    mesh, rows, lines = built
    parts = parts_by_name(rows, lines)
    towers = unary_union([part_footprint(p) for n, p in parts.items() if n.startswith("tower_")])
    stairs = unary_union([Polygon(p.plan) for n, p in parts.items() if n.startswith("stair_")])
    assert not towers.intersects(stairs)
    for path in mesh.aisles.values():
        assert all(towers.distance(Point(e, n)) > 1.0 for e, n, _ in path)
    # The scan is cut out under them too.
    assert towers.difference(unary_union(mesh.cut_plans)).area < 1e-6


def test_towers_reach_down_to_low_ground_beside_the_deck(tmp_path):
    rows, lines = make_rows()
    low = rows[0].elevation_m - 8.0
    parts = {p.name: p for p in stage_parts(rows, lines, BASE, ground=lambda e, n: low if abs(e) > 20 else 1800.0)}
    for side in ("left", "right"):
        bottom = min(p.z0 for n, p in parts.items() if n.startswith(f"tower_{side}_") and n.endswith(("base", "inner")))
        assert bottom == pytest.approx(low - BASE - stage.GROUND_EMBED_M)


def test_scan_stage_cut_takes_the_scans_tall_stage_near_the_new_one_only():
    rows, lines = make_rows()
    parts = stage_parts(rows, lines, BASE)
    a = stage.stage_axis(rows)
    origin = np.array([rows[0].centre_east, rows[0].centre_north]) - a * stage.STAGE_FRONT_SETBACK_M
    deck = rows[0].elevation_m - stage.STAGE_DROP_M
    heights = np.full((200, 200), deck)  # 1 m cells, x0 = y0 = -100 m
    def put(e, n, h):
        heights[int(n + 100), int(e + 100)] = deck + h
    put(18.0, 93.0, 14.0)    # a misplaced scan tower beside the stage: cut
    put(5.0, 71.0, 13.0)     # behind the new roof's back (the rock is there): kept
    put(80.0, 93.0, 14.0)    # far away: kept
    put(-18.0, 93.0, 8.0)    # a linking wall, under SCAN_STAGE_TALL_M: kept
    cut = unary_union(scan_stage_cut(ScanHeights(heights, -100.0, -100.0, 1.0), parts, origin, a, deck))
    assert cut.contains(Point(18.5, 93.5)) and cut.contains(Point(18.5 + 1.2, 93.5))  # with its edge
    for e, n in ((5.5, 71.5), (80.5, 93.5), (-17.5, 93.5)):
        assert not cut.contains(Point(e, n)), (e, n)


def test_real_build_towers_clear_the_seating():
    """On the real build (when out/ has one): no tower touches a tread, a stair or an aisle."""
    out = ROOT / "out"
    if not (out / "Stage_Towers.obj").exists() or not (out / "Benches.obj").exists():
        pytest.skip("no redrocks build in out/")
    sys.path.insert(0, str(ROOT / "tools"))
    from check_seating_clear import treads

    manifest = json.loads((out / "manifest.json").read_text())
    seat = unary_union([p for _, p in treads(out / "Benches.obj")])
    verts, faces = load(out / "Stage_Towers.obj")
    sv, sfaces = load(out / "Stairs.obj")
    stairs = unary_union([Polygon(sv[[t[0] - 1 for t in f]][:, :2] / 100) for f in sfaces]).buffer(0)
    towers = unary_union([Polygon(verts[[t[0] - 1 for t in f]][:, :2] / 100).buffer(1e-3) for f in faces])
    assert towers.distance(seat) > 1.0 and towers.distance(stairs) > 1.0
    for path in manifest["stage"]["aisles"].values():
        assert all(towers.distance(Point(e, n)) > 1.0 for e, n, _ in path)
    assert set(manifest["stage"]["objs"]) == set(STAGE_MESHES[1:])


# ---- Unreal wiring (plain Python, no editor) ---------------------------------------

def _load_gameplay():
    saved = sys.modules.get("unreal")
    sys.modules["unreal"] = types.ModuleType("unreal")
    try:
        spec = importlib.util.spec_from_file_location("setup_gameplay_stage_test", ROOT / "unreal" / "setup_gameplay.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if saved is None:
            del sys.modules["unreal"]
        else:
            sys.modules["unreal"] = saved


def test_gameplay_imports_collides_and_paints_every_stage_obj():
    gp = _load_gameplay()
    assert list(gp.STAGE_MESHES) == list(STAGE_MESHES)
    for name in STAGE_MESHES:
        assert name in gp.NEW_MESHES and name in gp.COLLIDE, name
        assert gp.MESH_MATERIAL[name] in gp.PLAIN, name
    assert gp.MESH_MATERIAL["Stage_Steel"] == "M_StageSteel"
    assert gp.MESH_MATERIAL["Stage_Roof"] == "M_Roof"
    assert gp.MESH_MATERIAL["Stage_Towers"] == "M_Benches"
    assert gp.MESH_MATERIAL["Stage"] == "M_Concrete"


def test_materials_build_the_stage_steel_and_apply_the_rest():
    tree = ast.parse((ROOT / "unreal" / "setup_materials.py").read_text())
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "build_stage_steel" in funcs
    values = {t.id: n.value for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
              if isinstance(t, ast.Name)}
    surfaces = [tuple(ast.unparse(e) for e in item.elts) for item in values["SURFACES"].elts]
    assert ("'M_StageSteel'", "build_stage_steel", "'Stage_Steel'") in surfaces
    also = ast.literal_eval(values["ALSO_APPLY"])
    assert "Stage_Roof" in also["M_Roof"] and "Stage_Fascia" in also["M_Steel"]
    assert "Stage_Towers" in also["M_Benches"]
