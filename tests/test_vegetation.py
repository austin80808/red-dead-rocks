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
from redrocks import foliage_atlas as fa
from redrocks import vegetation as veg
from redrocks import vegetation_metrics as vm

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
    ("Oak", 2.0, 3.0), ("Mahogany", 1.6, 2.6), ("OakPatch", 2.0, 3.2),
])
def test_model_heights_match_the_species(species, lo, hi):
    for i in range(veg.SPECIES[species][1]):
        m = veg.build_model(species, i)
        top = max(mesh.positions()[:, 2].max() for mesh in m.parts.values())
        assert lo <= top <= hi, (m.name, top)
        assert m.parts["Trunk"].positions()[:, 2].min() < 0  # trunk base sinks into the ground


def test_ponderosa_trunk_is_clear_low_and_scaled_instances_reach_12_to_25_m():
    m = veg.build_model("Ponderosa", 0)
    leaves = m.parts["Leaves"].positions()
    assert leaves[:, 2].min() >= 0.3 * m.height_m  # no needles on the lower trunk
    lo, hi = veg.SCALE["Ponderosa"]
    assert 12.0 <= lo * m.height_m and hi * m.height_m <= 25.0


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


@pytest.mark.parametrize("atlas", veg.ATLASES)
def test_atlas_paints_at_real_size(atlas):
    sizes = fa.painted_sizes(atlas)
    lo, hi = fa.SIZE_M[atlas]
    main = sizes.get("leaf") or sizes.get("spray") or sizes.get("needle")
    assert main and lo <= np.median(main) <= hi and min(main) >= lo * 0.7 and max(main) <= hi * 1.3
    if atlas == "Juniper":
        b = sizes["branchlet"]
        assert 0.001 <= np.median(b) <= 0.002


@pytest.mark.parametrize("atlas", ["Juniper", "Pine", "Oak"])
def test_atlas_colour_is_near_the_photo(atlas):
    assert vm.delta_e(fa.mean_colour(fa.paint_atlas(atlas, 256)), fa.PHOTO_COLOUR[atlas]) <= 20


def test_mahogany_is_greyer_and_lighter_than_oak():
    oak, mah = (vm.hex_to_lab(fa.mean_colour(fa.paint_atlas(a, 256))) for a in ("Oak", "Mahogany"))
    assert mah[0] > oak[0] and np.hypot(*mah[1:]) < np.hypot(*oak[1:])


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
    # Into the level frame: Y = -north, and the mirror flips yaw and roll.
    assert loc == (1.0, -2.0, 3.0) and scale == (1.1, 1.1, 1.2)
    assert rot == {"pitch": 1.0, "yaw": -90.0, "roll": 1.0}


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
    # views.json is in the terrain frame (east, north); the level has north on -Y.
    assert shots.matched_views(views) == [
        ("trail", (1.0, -2.0, 3.0), {"pitch": -5.0, "yaw": -90.0, "roll": 0.0}, 70.0)]


def test_shot_views_default_to_the_repo_views_file(shots):
    assert Path(shots.VIEWS_PATH).resolve() == (UNREAL.parent / "docs/media/vegetation-real/views.json").resolve()


@pytest.mark.parametrize("species", ["JuniperRM", "JuniperOS"])
def test_junipers_are_dense_from_the_side(species):
    alpha = fa.paint_atlas("Juniper", 256)[..., 3] / 255.0
    for i in range(veg.SPECIES[species][1]):
        model = veg.build_model(species, i)
        assert vm.side_fill(model, alpha) >= 0.85, model.name


@pytest.mark.parametrize("species", list(veg.SPECIES))
def test_leaf_cards_use_their_atlas_card_width(species):
    for i in range(veg.SPECIES[species][1]):
        model = veg.build_model(species, i)
        w = vm.card_widths(model)
        assert np.all(np.abs(w / fa.CARD_M[model.atlas] - 1) <= 0.2), model.name


def test_one_seed_juniper_instances_are_2_to_4_m():
    lo, hi = veg.SCALE["JuniperOS"]
    h = veg.build_model("JuniperOS", 0).height_m
    assert 1.9 <= lo * h and hi * h <= 4.1


PINES = range(5)


def test_ponderosa_has_five_variants():
    assert veg.SPECIES["Ponderosa"][1] == 5


@pytest.mark.parametrize("i", PINES)
def test_ponderosa_crown_base_and_trunk(i):
    p = veg.build_model("Ponderosa", i)
    assert 0.28 <= vm.crown_base_fraction(p) <= 0.60  # ref02: the crown starts about 28 % up
    lo, hi = veg.SCALE["Ponderosa"]
    d = vm.trunk_diameter_at(p, 1.4)
    assert d * lo >= 0.35 and d * hi <= 0.80
    assert 12.0 <= lo * p.height_m and hi * p.height_m <= 25.0


@pytest.mark.parametrize("i", PINES)
def test_ponderosa_is_open_with_needles_at_the_tips(i):
    p = veg.build_model("Ponderosa", i)
    alpha = fa.paint_atlas("Pine", 256)[..., 3] / 255.0
    assert 0.45 <= vm.side_fill(p, alpha) <= 0.75
    assert vm.tip_fraction(p, 1.0) >= 0.70


def _crown_width_at(p, f):
    """90th-percentile XY radius of leaf vertices within 5 % of height fraction f of the crown."""
    z = p.parts["Leaves"].positions()
    base, top = z[:, 2].min(), z[:, 2].max()
    zf = base + f * (top - base)
    band = z[np.abs(z[:, 2] - zf) <= 0.05 * (top - base)]
    return float(np.percentile(np.hypot(band[:, 0], band[:, 1]), 90)) if len(band) else 0.0


def test_old_ponderosa_is_flat_topped_and_young_is_conical():
    young, old = veg.build_model("Ponderosa", 0), veg.build_model("Ponderosa", 4)
    assert _crown_width_at(young, 0.9) < 0.35 * _crown_width_at(young, 0.3)
    assert _crown_width_at(old, 0.9) > 0.6 * _crown_width_at(old, 0.5)


def test_oak_patches_are_3_to_6_m_across_with_a_continuous_canopy():
    alpha = fa.paint_atlas("Oak", 256)[..., 3] / 255.0
    for i in range(3):
        p = veg.build_model("OakPatch", i)
        lo, hi = veg.SCALE["OakPatch"]
        across = 2 * vm.footprint_radius_m(p)
        assert across * lo >= 3.0 and across * hi <= 6.0, (p.name, across)
        assert vm.side_fill(p, alpha) >= 0.7, p.name
        assert len(p.parts["Trunk"].f) > 0 and veg.closed_and_outward(p.parts["Trunk"])
        assert p.species == "gambel_oak" and p.atlas == "Oak"


@pytest.mark.parametrize("species,lo_m,hi_m", [("Oak", 1.5, 4.0), ("Mahogany", 1.0, 3.0)])
def test_scaled_shrubs_reach_their_real_heights(species, lo_m, hi_m):
    lo, hi = veg.SCALE[species]
    for i in range(veg.SPECIES[species][1]):
        h = veg.build_model(species, i).height_m
        assert lo * h >= lo_m * 0.95 and hi * h <= hi_m * 1.05, (species, i, h)


# ------------------------------------------------------------- thickets & groves

SHRUBS = ("Oak", "OakPatch", "Mahogany")


def _low_gentle_ground():
    return veg.Ground(_slope_grid(n=81, size=320000.0, fn=lambda x, y: 0.08 * x + 20000.0))


def _open_points(ground, step=400.0):
    x0, y0, x1, y1 = ground.extent
    xx, yy = np.meshgrid(np.arange(x0 + 1000, x1 - 1000, step), np.arange(y0 + 1000, y1 - 1000, step))
    pts = np.stack([xx.ravel(), yy.ravel()], 1)
    far = np.hypot(pts[:, 0] - veg.SCAN_CENTRE_CM[0], pts[:, 1] - veg.SCAN_CENTRE_CM[1]) > veg.SCAN_CLEAR_CM + 1000
    return pts[far]


@pytest.fixture(scope="module")
def low_ground_plants(models):
    by_name = {mm.name: mm for mm in models}
    g = _low_gentle_ground()
    return g, by_name, veg.place(g, set(), by_name)


def test_thicket_field_is_seeded_noise_in_0_1():
    x = np.linspace(-100000, 100000, 500)
    f = veg.thicket_field(x, x * 0.3, veg.SEED)
    assert f.min() >= 0.0 and f.max() <= 1.0 and f.std() > 0.05
    assert np.array_equal(f, veg.thicket_field(x, x * 0.3, veg.SEED))


def test_thickets_are_continuous_and_openings_are_open(low_ground_plants):
    g, by_name, out = low_ground_plants
    pts = _open_points(g)
    inside = veg.in_thicket(g, pts[:, 0], pts[:, 1], veg.SEED)
    assert 0.15 < inside.mean() < 0.6
    assert vm.canopy_cover(out, by_name, pts[inside], SHRUBS) >= 0.60
    assert vm.canopy_cover(out, by_name, pts[~inside], SHRUBS) <= 0.15


def test_shrub_canopy_stays_off_masked_cells(models):
    by_name = {mm.name: mm for mm in models}
    g = veg.Ground(_slope_grid(n=41, size=160000.0, fn=lambda x, y: 0.05 * x + 20000.0))
    road = np.asarray([(-80000.0 + i * 200.0, 30000.0, 0.0) for i in range(800)])
    blocked = veg.blocked_cells(road, dilate=1)
    out = veg.place(g, blocked, by_name)

    def cell_of(x, y):
        return int(math.floor(x / veg.BLOCK_CELL_CM)), int(math.floor(y / veg.BLOCK_CELL_CM))

    n = 0
    for name, rows in out.items():
        if not name.startswith(SHRUBS):
            continue
        r = vm.footprint_radius_m(by_name[name]) * 100
        for x, y, z, yaw, pitch, roll, sxy, sz in rows:
            n += 1
            for dx, dy in ((r * sxy, 0), (-r * sxy, 0), (0, r * sxy), (0, -r * sxy)):
                assert cell_of(x + dx, y + dy) not in blocked, name
    assert n > 100


def _group_sizes(pts, link):
    """Single-linkage group sizes: union-find over pairs closer than ``link``."""
    parent = list(range(len(pts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(pts)):
        d = np.hypot(pts[i + 1:, 0] - pts[i, 0], pts[i + 1:, 1] - pts[i, 1])
        for j in np.flatnonzero(d < link) + i + 1:
            parent[find(i)] = find(int(j))
    roots = [find(i) for i in range(len(pts))]
    return np.bincount(np.unique(roots, return_inverse=True)[1])


def test_ponderosa_grows_in_groves(models):
    by_name = {mm.name: mm for mm in models}
    g = veg.Ground(_slope_grid(n=81, size=320000.0, fn=lambda x, y: 90000.0 - 0.25 * y))
    out = veg.place(g, set(), by_name)
    pines = np.array([r[:2] for n, rs in out.items() if n.startswith("Ponderosa") for r in rs])
    assert len(pines) > 50
    assert 3 <= np.median(_group_sizes(pines, 1500.0)) <= 8


def test_instance_count_is_bounded(low_ground_plants):
    _, _, out = low_ground_plants
    assert sum(map(len, out.values())) <= 0.25 * (3200 ** 2) / 9  # at most one plant per 36 m²


def test_oak_patches_are_placed(low_ground_plants):
    _, _, out = low_ground_plants
    assert sum(len(r) for n, r in out.items() if n.startswith("OakPatch")) > 100


# ------------------------------------------------------------- Unreal: colour and culling

def test_every_model_maps_to_materials_and_cull(sv, models):
    for mm in models:
        assert sv.bark_group(mm.name) in sv.BARK and mm.atlas in sv.FOLIAGE
        for part in veg.PARTS:
            s, e = sv.cull_distances(mm.name, part)
            assert (s, e) == (0, 0) or 0 < s < e


def test_shrub_parts_cull_together_and_crowns_never(sv):
    for name in ("Oak_0", "OakPatch_1", "Mahogany_2"):
        assert sv.cull_distances(name, "Crown") == (0, 0)
        assert sv.cull_distances(name, "Leaves") == sv.cull_distances(name, "Trunk") == (32000, 40000)
    for name in ("Ponderosa_4", "JuniperRM_0", "JuniperOS_2"):
        for part in veg.PARTS:
            assert sv.cull_distances(name, part) == (0, 0)


def test_ponderosa_bark_is_dark_grey_brown_at_distance(sv):
    far = vm.hex_to_lab(sv.BARK["Ponderosa"][2])
    assert far[0] < 40 and np.hypot(far[1], far[2]) < 12  # dark, low-chroma grey-brown
    for group, (near_light, near_dark, far, plate, stretch, contrast) in sv.BARK.items():
        assert vm.hex_to_lab(far)[0] < vm.hex_to_lab(near_light)[0]  # far reads darker than sunlit plates


def test_foliage_carries_the_atlas_colour_and_photo_cores(sv):
    for atlas, ((ta, tb), (ca, cb), sss) in sv.FOLIAGE.items():
        for tint in (ta, tb):
            assert min(vm.hex_rgb(tint)) >= 0.88 * 255  # within ~10 % of white: the atlas carries the colour
        if atlas in fa.PHOTO_COLOUR:
            photo = vm.hex_rgb(fa.PHOTO_COLOUR[atlas])
            ga, gb = sv.CORE_GAIN[atlas]
            assert np.allclose(vm.hex_rgb(ca), photo * ga, atol=2) and np.allclose(vm.hex_rgb(cb), photo * gb, atol=2)


def test_snow_runs_after_vegetation_and_covers_veg_materials():
    saved = sys.modules.get("unreal")
    sys.modules["unreal"] = types.ModuleType("unreal")
    try:
        spec = importlib.util.spec_from_file_location("rebuild_level_under_test", UNREAL / "rebuild_level.py")
        rl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rl)
    finally:
        if saved is None:
            del sys.modules["unreal"]
        else:
            sys.modules["unreal"] = saved
    steps = [s for s, _ in rl.STEPS]
    assert steps.index("setup_snow.py") > steps.index("setup_vegetation.py")
    snow = (UNREAL / "setup_snow.py").read_text()
    assert 'GAME_DIR = "/Game/RedRocks"' in snow and "list_assets(GAME_DIR, recursive=True" in snow
    assert 'ASSET_DIR = "/Game/RedRocks/Vegetation"' in (UNREAL / "setup_vegetation.py").read_text()


def test_scrub_lines_paths_without_overhanging_them(models):
    by_name = {mm.name: mm for mm in models}
    g = veg.Ground(_slope_grid(n=41, size=160000.0, fn=lambda x, y: 0.05 * x + 20000.0))
    path = np.asarray([(-80000.0 + i * 100.0, 30000.0 + 20.0 * math.sin(i / 40.0), 0.0) for i in range(1600)])
    coarse = veg.blocked_cells(path, dilate=1)
    fine = veg.blocked_cells(path, cell=veg.FINE_CELL_CM, dilate=0)
    out = veg.place(g, coarse, by_name, shrub_masks=(set(), fine))
    near = 0
    for name, rows in out.items():
        if not name.startswith(SHRUBS):
            continue
        r = vm.footprint_radius_m(by_name[name]) * 100
        for x, y, z, yaw, pitch, roll, sxy, sz in rows:
            for dx, dy in ((r * sxy, 0), (-r * sxy, 0), (0, r * sxy), (0, -r * sxy), (0, 0)):
                c = (int(math.floor((x + dx) / veg.FINE_CELL_CM)), int(math.floor((y + dy) / veg.FINE_CELL_CM)))
                assert c not in fine, name
            near += abs(y - 30000.0) < 600.0
    assert near > 20  # scrub grows right up to the path edge
    for name, rows in out.items():  # trees still keep the coarse mask (their trunks collide)
        if not name.startswith(SHRUBS):
            for x, y, *_ in rows:
                assert (int(math.floor(x / veg.BLOCK_CELL_CM)), int(math.floor(y / veg.BLOCK_CELL_CM))) not in coarse


def test_mesh_cells_cover_a_triangle_not_just_its_corners(tmp_path):
    obj = tmp_path / "t.obj"
    obj.write_text("v 0 0 0\nv 2000 0 0\nv 0 2000 0\nf 1/1/1 2/1/1 3/1/1\n")
    cells = veg.mesh_cells(obj, cell=200.0)
    assert (5, 0) in cells and (0, 5) in cells and (3, 3) in cells  # edges and interior, between the corners
    assert (8, 8) not in cells  # outside the hypotenuse


# ------------------------------------------------------------- final-review fixes

def _worst_root_excess(row, model, ground):
    """Highest any stem root of this instance can stand above the ground, in cm,
    over every yaw (the sink must not depend on Unreal's rotation convention)."""
    x, y, z, yaw, pitch, roll, sxy, sz = row
    worst = -1e9
    for rx, ry, rz, rr in model.roots:
        rho = (math.hypot(rx, ry) + rr) * 100 * sxy
        for a in np.linspace(0, 2 * math.pi, 16, endpoint=False):
            gx, gy = x + math.cos(a) * rho, y + math.sin(a) * rho
            gz = float(ground.height(np.array([gx]), np.array([gy]))[0])
            tilt = math.tan(math.radians(abs(pitch) + abs(roll))) * rho
            worst = max(worst, z + rz * 100 * sz + tilt - gz)
    return worst


def test_no_plant_floats_on_a_25_degree_slope(models):
    by_name = {mm.name: mm for mm in models}
    g = veg.Ground(_slope_grid(n=41, size=160000.0, fn=lambda x, y: math.tan(math.radians(25)) * x + 60000.0))
    out = veg.place(g, set(), by_name)
    checked = 0
    for name, rows in out.items():
        for row in rows[::15]:
            assert _worst_root_excess(row, by_name[name], g) <= 10.0, name
            checked += 1
    assert checked > 100


def test_every_model_records_its_roots(models):
    for mm in models:
        assert mm.roots, mm.name
        for rx, ry, rz, rr in mm.roots:
            assert rz <= 0.0 and rr > 0.0


def test_ponderosa_limbs_are_thinner_than_the_trunk():
    for i in range(5):
        p = veg.build_model("Ponderosa", i)
        assert len(p.limbs) > 20
        for z, r_limb, r_trunk in p.limbs:
            assert r_limb <= 0.45 * r_trunk, (p.name, z, r_limb, r_trunk)


def test_no_plants_in_buildings_or_walls(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    verts = _slope_grid(n=21, size=240000.0, fn=lambda x, y: 0.05 * x + 20000.0)
    (out / "RedRocks_Terrain.obj").write_text("\n".join("v %.1f %.1f %.1f" % tuple(v) for v in verts) + "\n")
    # A 60 x 40 m building and a 200 m wall, as OSM extrusions (footprint triangles).
    (out / "Buildings.obj").write_text("v 60000 60000 0\nv 66000 60000 0\nv 66000 64000 0\nv 60000 64000 0\n"
                                       "f 1 2 3\nf 1 3 4\n")
    (out / "Walls.obj").write_text("v -90000 -20000 0\nv -70000 -20000 0\nv -70000 -19960 0\nv -90000 -19960 0\n"
                                   "f 1 2 3\nf 1 3 4\n")
    info = veg.export(out / "vegetation", terrain_obj=out / "RedRocks_Terrain.obj", avoid_dir=out, textures=False)
    n = 0
    for rows in info["instances"].values():
        for x, y, *_ in rows:
            n += 1
            assert not (60000 <= x <= 66000 and 60000 <= y <= 64000)
            assert not (-90000 <= x <= -70000 and -20000 <= y <= -19960)
    assert n > 100


def test_trees_keep_their_distance(models):
    by_name = {mm.name: mm for mm in models}
    g = veg.Ground(_slope_grid(n=81, size=320000.0, fn=lambda x, y: 90000.0 - 0.25 * y))
    out = veg.place(g, set(), by_name)
    for prefix, gap in (("Ponderosa", 250.0), ("Juniper", 100.0)):
        pts = np.array([r[:2] for n, rs in out.items() if n.startswith(prefix) for r in rs])
        assert len(pts) > 50
        cells = {}
        for k, (x, y) in enumerate(pts):
            cells.setdefault((int(x // gap), int(y // gap)), []).append(k)
        for (i, j), ks in cells.items():
            near = [k2 for di in (-1, 0, 1) for dj in (-1, 0, 1) for k2 in cells.get((i + di, j + dj), [])]
            for k in ks:
                d = np.hypot(pts[near, 0] - pts[k, 0], pts[near, 1] - pts[k, 1])
                assert (d[np.array(near) != k] >= gap - 0.5).all(), prefix  # rows are rounded to 0.1 cm


def test_ground_slope_is_right_at_the_grid_edge():
    g = veg.Ground(_slope_grid(n=41, size=160000.0, fn=lambda x, y: 0.5 * x))
    nx, ny, nz = g.normal(np.array([-79900.0, 0.0, 79900.0]), np.array([0.0, 0.0, 0.0]))
    assert np.allclose(-nx / nz, 0.5, atol=1e-6)


# ------------------------------------------------------------- beside the aisle stairs

def test_columnar_junipers_are_narrow_and_upright(models):
    """The planted form beside the aisles: 3.7-5.8 m tall, 1.1-1.8 m across, widest low, tapering to a point."""
    columnar = [m for m in models if m.form == "columnar"]
    assert {m.name for m in columnar} == {"JuniperRM_3", "JuniperRM_4"}
    lo, hi = veg.AISLE_SCALE["columnar"]
    for m in columnar:
        top = max(mesh.positions()[:, 2].max() for mesh in m.parts.values())
        across = 2 * vm.footprint_radius_m(m)
        assert across / top < 0.35, m.name  # the broad Rocky Mountain junipers are about 0.55
        assert 3.5 <= lo * top and hi * top <= 6.2, m.name
        assert 1.0 <= lo * across and hi * across <= 2.0, m.name
        # Widest in its lower half, tapering to the top.
        assert _crown_width_at(m, 0.85) < 0.7 * max(_crown_width_at(m, f) for f in (0.2, 0.3, 0.4))
        assert m.species == "juniper_rocky_mountain" and m.atlas == "Juniper"


def test_hillsides_never_get_the_planted_form(models):
    """The columnar junipers are only for the aisles, and adding them leaves the hillsides as they were."""
    by_name = {mm.name: mm for mm in models}
    g = veg.Ground(_slope_grid(n=81, size=320000.0, fn=lambda x, y: 90000.0 - 0.25 * y))
    out = veg.place(g, set(), by_name)
    assert not out["JuniperRM_3"] and not out["JuniperRM_4"]
    without = veg.place(g, set(), {n: mm for n, mm in by_name.items() if mm.form != "columnar"})
    assert without == {n: rows for n, rows in out.items() if by_name[n].form != "columnar"}


AISLE_BASE = 1750.0
AISLE_ROWS = 40
WEST_PATH_E = -47.5  # a footway along the west strip, 5.3 m outside its cheek walls
EAST_ROCK_E = 48.5  # a rock face along the east strip, 6.3 m outside them


def _aisle_ground(e, n):
    """The scan's ground: the bowl's rake, then rock faces past the strips."""
    z = 1800.0 + 0.45 * (n - 100.0)
    return z + 1.6 * np.clip(np.asarray(e) - EAST_ROCK_E, 0, None) + 1.6 * np.clip(-58.0 - np.asarray(e), 0, None)


@pytest.fixture(scope="module")
def aisle_build(tmp_path_factory, models):
    """A small build: 40 terraced rows 80 m long rising north, row-end stairs, a
    booth, treads, a path along the west strip, and a scan with a rock face
    along the east strip and a tall scanned bush in the west one."""
    from redrocks import stage
    from redrocks.scan_blend import ScanHeights
    from redrocks.venue import Row

    out = tmp_path_factory.mktemp("aisles")
    rows, lines = [], []
    for i in range(AISLE_ROWS):
        rows.append(Row(index=i + 1, seats=100, length_m=80.0, elevation_m=1800.0 + 0.45 * i,
                        centre_east=0.0, centre_north=100.0 + i))
        lines.append(np.array([[-40.0, 100.0 + i], [40.0, 100.0 + i]]))
    levels = [1800.45 + 0.45 * i for i in range(AISLE_ROWS)]
    booth = np.array([[-3.0, 112.0], [3.0, 112.0], [3.0, 116.5], [-3.0, 116.5]])
    mesh = stage.export_stage_obj(rows, lines, out / "Stage.obj", AISLE_BASE, levels=levels, booth=booth)
    # The treads, as Benches.obj's upward faces.
    v, f = [], []
    for i, z in enumerate(levels):
        zc = (z - AISLE_BASE) * 100.0
        k = len(v)
        v += [(-4000.0, (100.0 + i) * 100, zc), (4000.0, (100.0 + i) * 100, zc),
              (4000.0, (101.0 + i) * 100, zc), (-4000.0, (101.0 + i) * 100, zc)]
        f += [(k + 1, k + 2, k + 3), (k + 1, k + 3, k + 4)]
    (out / "Benches.obj").write_text("o Benches\n" + "".join("v %.1f %.1f %.1f\n" % p for p in v)
                                     + "".join("f %d %d %d\n" % t for t in f))
    # A 1.5 m footway running up beside the west aisle.
    x0, x1 = (WEST_PATH_E - 0.75) * 100, (WEST_PATH_E + 0.75) * 100
    (out / "Paths.obj").write_text(f"v {x0} 9000 0\nv {x1} 9000 0\nv {x1} 15000 0\nv {x0} 15000 0\nf 1 2 3\nf 1 3 4\n")
    (out / "manifest.json").write_text(json.dumps({
        "benches": {"base_elevation_m": AISLE_BASE},
        "stage": {"aisles": {side: [list(p) for p in path] for side, path in mesh.aisles.items()}},
    }))
    xs = np.arange(-80.0, 80.0) + 0.5
    ys = np.arange(80.0, 160.0) + 0.5
    e, n = np.meshgrid(xs, ys)
    ground = _aisle_ground(e, n)
    top = ground + 0.1
    bush = (np.abs(e - (-44.5)) < 1.0) & (np.abs(n - 115.0) < 1.0)  # a 2 m scanned bush, 4 m tall
    top[bush] += 4.0
    ScanHeights(ground, -80.0, 80.0, 1.0).save(out / "ground.npz")
    ScanHeights(top, -80.0, 80.0, 1.0).save(out / "top.npz")
    by_name = {mm.name: mm for mm in models}
    site = veg.load_aisle_site(out, out / "ground.npz", out / "top.npz")
    inst, summary = veg.place_aisles(site, by_name)
    return {"out": out, "site": site, "instances": inst, "summary": summary, "models": by_name,
            "ground": ScanHeights.load(out / "ground.npz"), "bush": (e[bush], n[bush]), "mesh": mesh}


def _aisle_plants(build):
    for name, rows in build["instances"].items():
        for row in rows:
            yield name, build["models"][name], row


def _plan_reach(model):
    """How far any part of the plant reaches out from its trunk, metres at scale 1."""
    pts = np.concatenate([model.parts[p].positions() for p in veg.PARTS])
    return float(np.hypot(pts[:, 0], pts[:, 1]).max())


def test_aisle_strips_are_planted_up_both_sides_in_clumps(aisle_build):
    s = aisle_build["summary"]
    assert set(s["sides"]) == {"start", "end"}
    for side, info in s["sides"].items():
        clumps = info["clumps"]
        # At least one clump in every ten rows of the climb.
        for first in range(1, AISLE_ROWS + 1, 10):
            assert any(c["rows"][0] <= first + 9 and c["rows"][1] >= first for c in clumps), (side, first)
        sizes = [c["plants"] for c in clumps]
        assert max(sizes) <= 5 and np.mean([2 <= k for k in sizes]) >= 0.7, sizes  # groups, not a picket line
        assert sum(sizes) == info["plants"]
    names = [n for n, _, _ in _aisle_plants(aisle_build)]
    assert sum(n.rsplit("_", 1)[0] in ("JuniperRM", "Oak", "Mahogany") for n in names) >= 0.9 * len(names)
    assert sum(aisle_build["models"][n].form == "columnar" for n in names) >= 0.4 * len(names)
    assert s["plants"] == len(names) and s["species"] == {
        sp: sum(1 for n in names if n.startswith(sp + "_")) for sp in s["species"]}


def test_aisle_plants_keep_clear_of_stairs_aisles_rows_and_booth(aisle_build):
    """No crown reaches within 0.3 m of a stair step, the aisle path, a tread or the booth (in plan)."""
    import shapely

    out = aisle_build["out"]
    stairs = [shapely.MultiPoint(v[:, :2]).convex_hull for v in veg.obj_parts(out / "Stairs.obj").values()]
    booth = [shapely.MultiPoint(v[:, :2]).convex_hull for name, v in veg.obj_parts(out / "Stage.obj").items()
             if name.startswith("part foh_")]
    booth += [shapely.MultiPoint(v[:, :2]).convex_hull for v in veg.obj_parts(out / "Booth.obj").values()]
    aisle = [shapely.Point(p[:2]) for path in aisle_build["mesh"].aisles.values() for p in path]
    treads = shapely.union_all([shapely.box(-40.0, 100.0 + i, 40.0, 101.0 + i) for i in range(AISLE_ROWS)])
    n = 0
    for name, model, (x, y, z, yaw, pitch, roll, sxy, sz) in _aisle_plants(aisle_build):
        p = shapely.Point(x / 100.0, y / 100.0)
        r = _plan_reach(model) * sxy
        assert min(g.distance(p) for g in stairs) >= r + 0.3 - 1e-6, name
        assert min(g.distance(p) for g in aisle) >= r + 0.3 - 1e-6, name
        assert min(g.distance(p) for g in booth) >= r + 0.3 - 1e-6, name
        assert treads.distance(p) >= max(r + 0.3, 1.0) - 1e-6, name
        assert pitch == 0.0 and roll == 0.0  # planted upright
        n += 1
    assert n >= 20


def test_aisle_strips_stop_at_paths_and_rock_faces(aisle_build):
    """The west strip ends at the footway beside it, the east one at the foot of the rock;
    no trunk stands in a tall scanned bush."""
    bx, by = aisle_build["bush"]
    for name, model, (x, y, *_rest) in _aisle_plants(aisle_build):
        e, n, sxy = x / 100.0, y / 100.0, _rest[4]
        if e < 0:
            assert e >= WEST_PATH_E + 0.75 + _plan_reach(model) * sxy + 0.3 - 1e-6, name  # crown clear of the path
        else:
            assert e <= EAST_ROCK_E + 0.5, name  # not up the rock face
        assert np.min(np.hypot(bx - e, by - n)) > veg.AISLE_TALL_REACH_M, name
    assert all(42.19 < abs(x / 100.0) < 42.2 + veg.AISLE_BAND_M[1] + 0.01 for _, _, (x, *_) in _aisle_plants(aisle_build))


def test_aisle_plants_stand_on_the_scan_and_never_float(aisle_build):
    """Every trunk base is 10-15 cm under the scan's ground at its spot, and no root of any
    plant stands over that ground anywhere around it, whatever its yaw."""
    from redrocks.scan_blend import ScanGround

    scan = ScanGround(aisle_build["ground"], fallback=lambda e, n: float("nan"))
    ring = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    for name, model, (x, y, z, yaw, pitch, roll, sxy, sz) in _aisle_plants(aisle_build):
        e, n, base = x / 100.0, y / 100.0, AISLE_BASE + z / 100.0
        surface = scan(e, n)
        assert 0.0 <= surface - base <= 0.15 + 1e-3, (name, surface - base)
        for rx, ry, rz, rr in model.roots:
            rho = (math.hypot(rx, ry) + rr) * sxy
            for a in ring:
                assert base + rz * sz <= scan(e + math.cos(a) * rho, n + math.sin(a) * rho) + 1e-6, name


def test_aisle_planting_is_seeded(aisle_build):
    site, models = aisle_build["site"], aisle_build["models"]
    assert veg.place_aisles(site, models)[0] == aisle_build["instances"]
    assert veg.place_aisles(site, models, seed=7)[0] != aisle_build["instances"]


def test_export_plants_the_aisles_when_given_the_scan(tmp_path, aisle_build):
    """vegetation.json carries the aisle plants with the rest, and sums them up."""
    out = aisle_build["out"]
    verts = _slope_grid(n=21, size=240000.0, fn=lambda x, y: 0.05 * x + 20000.0)
    (out / "RedRocks_Terrain.obj").write_text("\n".join("v %.1f %.1f %.1f" % tuple(v) for v in verts) + "\n")
    info = veg.export(tmp_path / "veg", terrain_obj=out / "RedRocks_Terrain.obj", avoid_dir=out, textures=False,
                      scan_ground=out / "ground.npz", scan_top=out / "top.npz")
    a = info["aisle_planting"]
    assert a["plants"] == aisle_build["summary"]["plants"] and set(a["sides"]) == {"start", "end"}
    for name, rows in aisle_build["instances"].items():
        for row in rows:
            assert row in info["instances"][name]
    assert info["counts"]["JuniperRM"] >= a["species"]["JuniperRM"]
    assert {m["name"]: m.get("form") for m in info["models"]}["JuniperRM_3"] == "columnar"
    written = json.loads((tmp_path / "veg" / "vegetation.json").read_text())
    assert written["aisle_planting"]["clumps"] == a["clumps"]
    # Without the scan the strips stay as they were.
    bare = veg.export(tmp_path / "veg2", terrain_obj=out / "RedRocks_Terrain.obj", avoid_dir=out, textures=False)
    assert "aisle_planting" not in bare
