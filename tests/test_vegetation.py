"""The procedural vegetation: meshes, atlases, placement and the Unreal side's
plain-Python helpers."""

import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from redrocks import cli
from redrocks import vegetation as veg

UNREAL = Path(__file__).resolve().parent.parent / "unreal"


@pytest.fixture(scope="module")
def models():
    return veg.all_models()


def test_every_species_has_variants(models):
    names = {m.name for m in models}
    for species, (_, count) in veg.SPECIES.items():
        assert count >= 3
        assert {"%s_%d" % (species, i) for i in range(count)} <= names


def test_trunks_and_crowns_are_watertight_and_outward(models):
    for m in models:
        assert veg.closed_and_outward(m.parts["Trunk"]), m.name
        assert veg.closed_and_outward(m.parts["Crown"]), m.name


def test_every_face_has_a_uv_and_a_normal(models):
    for m in models:
        for part, mesh in m.parts.items():
            assert mesh.triangles > 0, (m.name, part)
            for fv, ft, fn in mesh.f:
                assert len(ft) == 3 and len(fn) == 3
                assert all(0 <= t < len(mesh.vt) for t in ft)
                assert all(0 <= n < len(mesh.vn) for n in fn)
            uv = np.asarray(mesh.vt)
            assert np.isfinite(uv).all()


def test_leaf_cards_stay_inside_one_atlas_tile(models):
    for m in models:
        for _, ft, _ in m.parts["Leaves"].f:
            uv = np.asarray([m.parts["Leaves"].vt[t] for t in ft])
            tile = np.floor(uv.mean(0) / 0.5)
            assert (uv >= tile * 0.5).all() and (uv <= tile * 0.5 + 0.5).all()


def test_variants_are_deterministic_and_distinct(tmp_path):
    a = veg.build_model("Ponderosa", 1)
    b = veg.build_model("Ponderosa", 1)
    c = veg.build_model("Ponderosa", 2)
    pa = veg.write_obj(a.parts["Trunk"], tmp_path / "a.obj", "a").read_bytes()
    pb = veg.write_obj(b.parts["Trunk"], tmp_path / "b.obj", "a").read_bytes()
    pc = veg.write_obj(c.parts["Trunk"], tmp_path / "c.obj", "a").read_bytes()
    assert pa == pb and pa != pc


@pytest.mark.parametrize("species,lo,hi", [
    ("JuniperRM", 3.5, 6.5), ("JuniperOS", 2.5, 5.0), ("Ponderosa", 15.0, 21.0),
    ("Oak", 1.0, 2.2), ("Mahogany", 1.2, 2.6),
])
def test_model_heights_match_the_species(species, lo, hi):
    for i in range(veg.SPECIES[species][1]):
        m = veg.build_model(species, i)
        top = max(mesh.positions()[:, 2].max() for mesh in m.parts.values())
        assert lo <= top <= hi, (m.name, top)
        assert m.parts["Trunk"].positions()[:, 2].min() < 0  # trunk base sinks into the ground


def test_ponderosa_trunk_is_clear_low_and_scaled_instances_reach_10_to_25_m():
    m = veg.build_model("Ponderosa", 0)
    leaves = m.parts["Leaves"].positions()
    assert leaves[:, 2].min() > 0.3 * m.height_m  # no needles on the lower trunk
    lo, hi = veg.SCALE["Ponderosa"]
    assert 9.0 <= lo * m.height_m and hi * m.height_m <= 26.0


def test_obj_is_in_centimetres_with_v_vt_vn_faces(tmp_path):
    m = veg.build_model("Oak", 0)
    path = veg.write_obj(m.parts["Trunk"], tmp_path / "t.obj", "Trunk")
    lines = path.read_text().splitlines()
    faces = [ln for ln in lines if ln.startswith("f ")]
    assert len(faces) == m.parts["Trunk"].triangles
    assert all(len(c.split("/")) == 3 for ln in faces for c in ln.split()[1:])
    zs = [float(ln.split()[3]) for ln in lines if ln.startswith("v ")]
    assert max(zs) > 100.0  # a 1.6 m shrub is > 100 cm tall


def test_tube_is_closed_and_its_volume_is_a_cylinder():
    mesh = veg.Mesh()
    veg.add_tube(mesh, [(0, 0, 0), (0, 0, 1), (0, 0, 2)], [0.5, 0.5, 0.5], 64)
    assert veg.closed_and_outward(mesh)
    assert veg.signed_volume(mesh) == pytest.approx(math.pi * 0.25 * 2, rel=0.01)


@pytest.mark.parametrize("atlas", veg.ATLASES)
def test_atlases_have_cutout_alpha_and_no_black_fringe(atlas):
    img = veg.paint_atlas(atlas, size=256)
    assert img.shape == (256, 256, 4)
    alpha = img[..., 3]
    cover = (alpha > 128).mean()
    assert 0.05 < cover < 0.8
    # Transparent texels carry foliage colour, not black, so mips don't darken.
    clear = img[alpha == 0][:, :3]
    assert clear.mean() > 30
    # Each quarter holds a spray.
    for ty in (0, 128):
        for tx in (0, 128):
            assert (alpha[ty:ty + 128, tx:tx + 128] > 128).mean() > 0.02


def _slope_grid(n=61, size=240000.0, fn=lambda x, y: 0.0):
    xs = np.linspace(-size / 2, size / 2, n)
    return np.asarray([(x, y, fn(x, y)) for y in xs for x in xs])


def test_placement_respects_masks_and_is_seeded(models):
    by_name = {m.name: m for m in models}
    ground = veg.Ground(_slope_grid(fn=lambda x, y: 0.15 * x + 30000.0))  # rising to the east
    road = np.asarray([(80000.0 + i * 100.0, 0.0, 0.0) for i in range(300)])
    blocked = veg.blocked_cells(road, dilate=1)
    out = veg.place(ground, blocked, by_name)
    rows = [r for rs in out.values() for r in rs]
    assert len(rows) > 500
    for x, y, z, yaw, pitch, roll, sxy, sz in rows:
        assert math.hypot(x - veg.SCAN_CENTRE_CM[0], y - veg.SCAN_CENTRE_CM[1]) >= veg.SCAN_CLEAR_CM - 2000
        assert z < float(ground.height(np.array([x]), np.array([y]))[0])  # sunk, never floating
        assert 0.3 < sz < 1.5 and abs(pitch) < 20 and abs(roll) < 20
    for name, rs in out.items():
        if not name.startswith(("Oak", "Mahogany")):  # shrubs scatter around their spot
            for x, y, *_ in rs:
                assert (int(math.floor(x / veg.BLOCK_CELL_CM)), int(math.floor(y / veg.BLOCK_CELL_CM))) not in blocked
    assert veg.place(ground, blocked, by_name) == out
    pines = [r for n, rs in out.items() if n.startswith("Ponderosa") for r in rs]
    shrubs = [r for n, rs in out.items() if n.startswith(("Oak", "Mahogany")) for r in rs]
    assert np.mean([r[0] for r in pines]) > np.mean([r[0] for r in shrubs])  # pines higher up


def test_no_plants_on_cliffs(models):
    ground = veg.Ground(_slope_grid(fn=lambda x, y: 2.0 * x))
    out = veg.place(ground, set(), {m.name: m for m in models})
    assert sum(len(r) for r in out.values()) == 0


def test_north_facing_slopes_favour_ponderosa(models):
    by_name = {m.name: m for m in models}
    def frac(fn):
        out = veg.place(veg.Ground(_slope_grid(fn=fn)), set(), by_name)
        pines = sum(len(r) for n, r in out.items() if n.startswith("Ponderosa"))
        return pines / max(1, sum(len(r) for r in out.values()))
    # Ground falling to the north faces north (y up is north).
    assert frac(lambda x, y: 60000.0 - 0.3 * y) > frac(lambda x, y: 60000.0 + 0.3 * y)


def test_cli_vegetation_writes_models_atlases_and_instances(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    verts = _slope_grid(n=21, size=240000.0, fn=lambda x, y: 0.1 * x + 20000.0)
    lines = ["v %.1f %.1f %.1f" % tuple(v) for v in verts]
    (out / "RedRocks_Terrain.obj").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(veg, "ATLAS_SIZE", 128)
    assert cli.main(["vegetation", "--out", str(out)]) == 0
    info = json.loads((out / "vegetation" / "vegetation.json").read_text())
    assert len(info["models"]) == sum(n for _, n in veg.SPECIES.values())
    for m in info["models"]:
        for part in veg.PARTS:
            assert (out / "vegetation" / m["parts"][part]["obj"]).exists()
    for png in info["atlases"].values():
        assert (out / "vegetation" / png).exists()
    assert sum(info["counts"].values()) > 0


def test_cli_vegetation_needs_a_terrain(tmp_path):
    assert cli.main(["vegetation", "--out", str(tmp_path)]) == 1


# ------------------------------------------------------------- Unreal script

@pytest.fixture(scope="module")
def sv():
    saved = sys.modules.get("unreal")
    sys.modules["unreal"] = types.ModuleType("unreal")
    try:
        spec = importlib.util.spec_from_file_location("setup_vegetation_under_test", UNREAL / "setup_vegetation.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if saved is None:
            del sys.modules["unreal"]
        else:
            sys.modules["unreal"] = saved


def test_only_tree_trunks_collide(sv):
    assert sv.part_collides("Ponderosa_0", "Trunk")
    assert sv.part_collides("JuniperOS_2", "Trunk")
    assert not sv.part_collides("Ponderosa_0", "Leaves")
    assert not sv.part_collides("JuniperRM_0", "Crown")
    assert not sv.part_collides("Oak_1", "Trunk")


def test_every_model_has_bark_and_foliage_materials(sv, models):
    for m in models:
        assert sv.bark_group(m.name) in sv.BARK
        assert m.atlas in sv.FOLIAGE


def test_instance_rows_use_keyword_rotations(sv):
    (loc, rot, scale), = sv.instance_rows([[1.0, 2.0, 3.0, 90.0, 1.0, -1.0, 1.1, 1.2]])
    assert loc == (1.0, 2.0, 3.0) and scale == (1.1, 1.1, 1.2)
    assert rot == {"pitch": 1.0, "yaw": 90.0, "roll": -1.0}


def test_out_dir_can_be_overridden(monkeypatch):
    src = (UNREAL / "setup_vegetation.py").read_text()
    assert 'os.environ.get("REDROCKS_OUT_DIR", "/Users/austin/dev/redrocks-ue/out")' in src


@pytest.fixture(scope="module")
def shots():
    saved = sys.modules.get("unreal")
    sys.modules["unreal"] = types.ModuleType("unreal")
    try:
        spec = importlib.util.spec_from_file_location("shoot_vegetation_under_test", UNREAL / "shoot_vegetation.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if saved is None:
            del sys.modules["unreal"]
        else:
            sys.modules["unreal"] = saved


def test_matched_views_skip_unset_cameras_and_carry_fov(shots):
    views = [{"name": "trail", "camera": {"x": 1.0, "y": 2.0, "z": 3.0, "pitch": -5.0, "yaw": 90.0, "fov": 70.0}},
             {"name": "ponderosa", "camera": {"x": 0, "y": 0, "z": 0, "pitch": 0, "yaw": 0, "fov": 0}}]
    assert shots.matched_views(views) == [
        ("trail", (1.0, 2.0, 3.0), {"pitch": -5.0, "yaw": 90.0, "roll": 0.0}, 70.0)]


def test_shot_views_default_to_the_repo_views_file(shots):
    assert Path(shots.VIEWS_PATH).resolve() == (UNREAL.parent / "docs/media/vegetation-real/views.json").resolve()
