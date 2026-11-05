"""The terrain frame -> Unreal level frame convention (unreal/level_frame.py).

Unreal is left-handed: X forward, Y right, Z up, and a yaw of +90 turns a
viewer from +X to +Y, to their right. So a viewer facing +X has +Y on their
right. Red Rocks has to come out with north on a viewer's left when they face
east, as it does on the ground; these tests pin that, and pin every script
that turns manifest coordinates into level ones to the same rule.
"""

import importlib.util
import math
import random
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
UNREAL = ROOT / "unreal"

spec = importlib.util.spec_from_file_location("level_frame", UNREAL / "level_frame.py")
lf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lf)


def _load(name):
    stub = types.ModuleType("unreal")
    saved = sys.modules.get("unreal")
    sys.modules["unreal"] = stub
    try:
        s = importlib.util.spec_from_file_location(name + "_frame_test", UNREAL / (name + ".py"))
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
        return mod
    finally:
        if saved is None:
            del sys.modules["unreal"]
        else:
            sys.modules["unreal"] = saved


# Unreal's own rules, written out from first principles (not from level_frame).
def ue_forward(yaw_deg):
    r = math.radians(yaw_deg)
    return math.cos(r), math.sin(r)


def ue_side(viewer, yaw_deg, point):
    """'left' or 'right' of a viewer at ``viewer`` facing ``yaw_deg``, in the level.

    Unreal's right is the forward turned by yaw +90: (-sin, cos).
    """
    r = math.radians(yaw_deg)
    right = (-math.sin(r), math.cos(r))
    d = (point[0] - viewer[0], point[1] - viewer[1])
    return "right" if d[0] * right[0] + d[1] * right[1] > 0 else "left"


def test_a_point_due_north_of_the_stage_is_on_your_left_facing_east():
    stage = lf.to_level(64.3, 8.9)  # Stage Rock, terrain frame metres
    north = lf.to_level(64.3, 58.9)
    south = lf.to_level(64.3, -41.1)
    east_yaw = lf.level_yaw(0.0)  # terrain heading 0 = east
    assert ue_side(stage, east_yaw, north) == "left"
    assert ue_side(stage, east_yaw, south) == "right"


def test_facing_up_the_bowl_creation_rock_is_right_and_ship_rock_left():
    """From the stage looking west up the rows, as on the real stage: Creation
    Rock (OSM way 91483333, north) on the right, Ship Rock (way 91483376, south)
    on the left. Centroids are the manifest's, from OSM."""
    stage = lf.to_level(64.32, 8.89)
    creation = lf.to_level(-31.25, 71.84)
    ship = lf.to_level(20.73, -126.84)
    west = lf.level_yaw(180.0)
    assert ue_side(stage, west, creation) == "right"
    assert ue_side(stage, west, ship) == "left"


def test_from_the_top_row_looking_at_the_stage_ship_rock_is_right():
    """The postcard view (Commons File:Red_Rocks_Amphitheater.JPG): from the
    top of the bowl facing the stage and Denver, Ship Rock is on the right."""
    top = lf.to_level(-60.0, 0.0)
    stage = lf.to_level(64.32, 8.89)
    yaw = math.degrees(math.atan2(stage[1] - top[1], stage[0] - top[0]))  # level yaw toward the stage
    assert ue_side(top, yaw, lf.to_level(20.73, -126.84)) == "right"
    assert ue_side(top, yaw, lf.to_level(-31.25, 71.84)) == "left"


def test_the_level_is_a_proper_map_of_the_terrain_not_a_mirror():
    """Turning from east toward north is a left (negative-yaw) turn in Unreal."""
    e = lf.to_level(1.0, 0.0)
    n = lf.to_level(0.0, 1.0)
    # Unreal's cross product sign for "n is a positive-yaw (right) turn from e".
    right_turn = e[0] * n[1] - e[1] * n[0] > 0
    assert not right_turn


@pytest.mark.parametrize("heading", [0.0, 30.0, 90.0, 135.0, 180.0, 255.0, -60.0])
def test_a_manifest_heading_faces_the_same_way_in_the_level(heading):
    """yaw_deg in the manifest is a terrain heading; its level yaw faces the
    level image of the same direction."""
    h = math.radians(heading)
    target = lf.to_level(math.cos(h), math.sin(h))
    f = ue_forward(lf.level_yaw(heading))
    assert f == pytest.approx(target, abs=1e-12)
    assert lf.heading(lf.level_yaw(heading)) == pytest.approx(heading)


@pytest.mark.parametrize("bearing,east,north", [
    (0.0, 0.0, 1.0), (90.0, 1.0, 0.0), (180.0, 0.0, -1.0), (270.0, -1.0, 0.0), (75.0, None, None)])
def test_compass_bearings_are_yaw_plus_90(bearing, east, north):
    b = math.radians(bearing)
    e, n = (math.sin(b), math.cos(b)) if east is None else (east, north)
    yaw = lf.compass_yaw(bearing)
    assert -180.0 <= yaw < 180.0
    assert ue_forward(yaw) == pytest.approx(lf.to_level(e, n), abs=1e-12)


def test_left_and_right_helpers_match_unreal():
    for yaw in (0.0, 37.0, 90.0, -120.0):
        f = ue_forward(yaw)
        assert ue_side((0, 0), yaw, lf.right_of(yaw)) == "right"
        assert ue_side((0, 0), yaw, lf.left_of(yaw)) == "left"
        assert lf.forward(yaw) == pytest.approx(f)


def test_obj_actors_need_no_flip():
    """The OBJ importer negates Y; with the actor's scale that must give the
    vertex's level position."""
    sx, sy, sz = lf.OBJ_ACTOR_SCALE
    for e, n in [(1200.0, 3400.0), (-50.0, -7.0)]:
        imported = (e, -n)  # what Interchange's OBJ import does to a vertex
        assert (sx * imported[0], sy * imported[1]) == lf.to_level(e, n)
    assert sx * sy * sz > 0  # no mirrored transform anywhere


def _fit_constants():
    src = (UNREAL / "import_scan.py").read_text()
    def num(name):
        return float(re.search(r"^%s = (-?[\d.]+)" % name, src, re.M).group(1))
    def tup(name):
        return tuple(float(v) for v in re.search(r"^%s = \(([^)]*)\)" % name, src, re.M).group(1).split(","))
    return num("FIT_SCALE"), num("FIT_YAW_DEG"), int(num("FIT_MIRROR_X")), tup("FIT_SCAN_CENTRE"), tup("FIT_OFFSET_CM")


def _fit_world(p, s, yaw_deg, mirror, centre, offset):
    """tools/fit_scan.py's map of a scan vertex into the terrain frame."""
    y = math.radians(yaw_deg)
    x, z = mirror * s * (p[0] - centre[0]), s * (p[1] - centre[1])
    return (math.cos(y) * x - math.sin(y) * z + offset[0], math.sin(y) * x + math.cos(y) * z + offset[1])


def _actor_world(p, location, yaw_deg, scale):
    y = math.radians(yaw_deg)
    x, z = scale[0] * p[0], scale[1] * p[1]
    return (math.cos(y) * x - math.sin(y) * z + location[0], math.sin(y) * x + math.cos(y) * z + location[1])


@pytest.mark.parametrize("mirror", [-1, 1])
def test_scan_actor_lands_the_fit_in_the_level(mirror):
    s, yaw, _, centre, offset = _fit_constants()
    location, ayaw, scale = lf.scan_actor_transform(s, yaw, mirror, centre, offset[:2])
    rng = random.Random(7)
    for _ in range(50):
        p = (rng.uniform(-40, 60), rng.uniform(-40, 60))
        want = lf.to_level(*_fit_world(p, s, yaw, mirror, centre, offset))
        assert _actor_world(p, location, ayaw, scale) == pytest.approx(want, abs=1e-6)


def test_the_real_scan_fit_needs_no_mirror_in_the_level():
    """The scan imports correctly handed (FIT_MIRROR_X = -1 against the
    right-handed terrain frame), so in a correct level it is only turned."""
    s, yaw, mirror, centre, offset = _fit_constants()
    assert mirror == -1
    _, _, (sx, sy) = lf.scan_actor_transform(s, yaw, mirror, centre, offset[:2])
    assert sx > 0 and sy > 0


def test_raster_uv_puts_the_north_west_corner_at_zero():
    (ox, oy), (ix, iy) = lf.raster_uv_transform(-1000.0, 3000.0, 4000.0, 2000.0)
    def uv(east, north):
        x, y = lf.to_level(east, north)
        return ((x - ox) * ix, (y - oy) * iy)
    assert uv(-1000.0, 3000.0) == pytest.approx((0.0, 0.0))  # north-west corner, row 0
    assert uv(3000.0, 1000.0) == pytest.approx((1.0, 1.0))   # south-east corner
    assert uv(1000.0, 2500.0)[1] == pytest.approx(0.25)      # a quarter of the way south


def test_vegetation_rotation_mirrors_yaw_and_roll_only():
    assert lf.level_rotator(30.0, 4.0, -2.0) == (-30.0, 4.0, 2.0)


def test_scan_cut_reads_level_vertices_in_the_terrain_frame():
    """cut_scan_collision.py tests the seating cut (a terrain-frame grid) with
    scan vertices carried back out of the level."""
    cut_mod = _load("cut_scan_collision")
    s, yaw, mirror, centre, offset = _fit_constants()
    location, ayaw, scale = lf.scan_actor_transform(s, yaw, mirror, centre, offset[:2])
    f = cut_mod.local_to_terrain(scale, ayaw, location)
    p = types.SimpleNamespace(x=12.0, y=30.0)
    assert f(p) == pytest.approx(_fit_world((12.0, 30.0), s, yaw, mirror, centre, offset), abs=1e-6)
    # A 1-row cut covering terrain (x 0..200, north 100..200): row 0 is its north edge.
    cut = (0.0, 200.0, 100.0, {0: [(0, 2)]})
    assert cut_mod.in_cut(150.0, 150.0, cut)
    assert not cut_mod.in_cut(150.0, -150.0, cut)  # the level Y of that point would be wrong


# ---- every script uses the one convention --------------------------------

SCRIPTS = sorted(p for p in UNREAL.glob("*.py") if p.name != "level_frame.py" and not p.name.startswith("probe_"))


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_no_script_places_north_on_plus_y_or_flips_an_actor(path):
    src = path.read_text()
    # north multiplied straight into a level coordinate
    assert not re.search(r"north[\w\"'\]]*\s*\*\s*100", src), "convert with level_frame.to_level_cm"
    # the old un-mirror scale
    assert not re.search(r"Vector\(\s*1(\.0)?\s*,\s*-1(\.0)?\s*,\s*1(\.0)?\s*\)", src)
    # a manifest yaw used without converting it
    for line in src.splitlines():
        if '["yaw_deg"]' in line:
            assert "yaw(" in line, line.strip()


def test_scripts_that_read_manifest_positions_import_level_frame():
    for path in SCRIPTS:
        src = path.read_text()
        if re.search(r"\[\"(east_m|north_m|centroid_north|front_edge_north_m)\"\]", src):
            assert "import level_frame" in src, path.name


def test_moon_keeps_its_place_in_the_sky():
    """The C++ moon (red-dead-rocks-unreal, RRWeather.cpp) was set at Unreal yaw
    210 in the mirrored level, i.e. terrain heading 210 (light toward the WSW,
    moon in the ENE). In the level frame that is yaw 150."""
    assert lf.level_yaw(210.0) % 360.0 == pytest.approx(150.0)
