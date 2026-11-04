"""Offline tests for the look scripts' plain-Python parts: sun angle, the
render-settings ini patch, and the terrain skirt geometry."""

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest

UNREAL = Path(__file__).resolve().parent.parent / "unreal"


def _load(name):
    """Import unreal/<name>.py with a stub `unreal` (no editor API, so no run)."""
    saved = sys.modules.get("unreal")
    sys.modules["unreal"] = types.ModuleType("unreal")
    try:
        spec = importlib.util.spec_from_file_location(name + "_under_test", UNREAL / (name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if saved is None:
            del sys.modules["unreal"]
        else:
            sys.modules["unreal"] = saved


@pytest.fixture(scope="module")
def lighting():
    return _load("setup_lighting")


@pytest.fixture(scope="module")
def configure():
    return _load("configure_level")


@pytest.fixture(scope="module")
def env():
    return _load("setup_environment")


def _direction(pitch, yaw):
    p, y = math.radians(pitch), math.radians(yaw)
    return math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)


@pytest.mark.parametrize("azimuth,east,north", [
    (270.0, 1.0, 0.0),   # sun in the west: light travels east
    (180.0, 0.0, 1.0),   # sun in the south: light travels north
    (90.0, -1.0, 0.0),
    (0.0, 0.0, -1.0),
])
def test_sun_points_away_from_the_sun(lighting, azimuth, east, north):
    pitch, yaw = lighting.sun_rotation(azimuth, 20.0)
    x, y, z = _direction(pitch, yaw)
    assert pitch == -20.0 and z < 0  # pointing down: the sun is up
    h = math.hypot(x, y)
    assert math.isclose(x / h, east, abs_tol=1e-9)
    assert math.isclose(y / h, north, abs_tol=1e-9)


def test_configured_sun_is_low_and_behind_the_spawn(lighting):
    """Morning sun in the east: the player spawns facing WSW up the bowl, so the
    light must travel west (x < 0) to fall on what they see, not backlight it."""
    pitch, yaw = lighting.sun_rotation(lighting.SUN_AZIMUTH_DEG, lighting.SUN_ELEVATION_DEG)
    x, _, _ = _direction(pitch, yaw)
    assert 5.0 <= lighting.SUN_ELEVATION_DEG <= 30.0
    assert x < 0  # light travels west: the sun is in the east, behind the player


INI = """[/Script/Engine.RendererSettings]
r.ReflectionMethod=1

[SystemSettings]
r.EyeAdaptationQuality=0
r.Lumen.DiffuseIndirect.Allow=0
r.Other=7

[/Script/Foo]
Bar=1
"""


def test_patch_rewrites_and_appends_inside_system_settings(configure):
    out = configure.patch_system_settings(INI, [
        ("r.EyeAdaptationQuality", "2"),
        ("r.Lumen.DiffuseIndirect.Allow", "1"),
        ("r.New", "5"),
    ])
    section = out.split("[SystemSettings]")[1].split("[/Script/Foo]")[0]
    assert "r.EyeAdaptationQuality=2" in section
    assert "r.Lumen.DiffuseIndirect.Allow=1" in section
    assert "r.New=5" in section and "r.Other=7" in section
    assert "=0" not in section
    assert out.startswith("[/Script/Engine.RendererSettings]\nr.ReflectionMethod=1")
    assert out.rstrip().endswith("[/Script/Foo]\nBar=1")


def test_patch_is_idempotent_and_adds_missing_section(configure):
    once = configure.patch_system_settings(INI, configure.RENDER_SETTINGS)
    assert configure.patch_system_settings(once, configure.RENDER_SETTINGS) == once
    fresh = configure.patch_system_settings("[A]\nx=1\n", [("r.Y", "2")])
    assert fresh == "[A]\nx=1\n\n[SystemSettings]\nr.Y=2\n"


def test_render_settings_turn_lumen_and_eye_adaptation_on(configure):
    s = dict(configure.RENDER_SETTINGS)
    assert s["r.DynamicGlobalIlluminationMethod"] == "1"
    assert s["r.Lumen.DiffuseIndirect.Allow"] == "1"
    assert s["r.EyeAdaptationQuality"] != "0"
    assert configure.EXPOSURE_MIN_EV100 < configure.EXPOSURE_MAX_EV100


def _grid(n=5, size=100.0):
    verts = []
    for j in range(n):
        for i in range(n):
            x = -size / 2 + size * i / (n - 1)
            y = -size / 2 + size * j / (n - 1)
            verts.append((x, y, x * 0.1 + y * 0.01))
    return verts


def test_border_walks_the_edge_counter_clockwise(env):
    border = env.grid_border(_grid())
    assert len(border) == 16 and len(set(border)) == 16
    assert border[0][:2] == (-50.0, -50.0) and border[4][:2] == (50.0, -50.0)
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(border, border[1:] + border[:1]))
    assert area > 0  # counter-clockwise seen from above


def test_border_rejects_a_ragged_grid(env):
    with pytest.raises(ValueError):
        env.grid_border(_grid()[:-1])


def test_skirt_starts_on_the_terrain_edge_and_faces_up(env):
    border = env.grid_border(_grid())
    verts, uvs, tris = env.skirt_mesh(border)
    rings = len(env.RING_OUT_CM)
    assert len(verts) == len(uvs) == rings * len(border)
    assert len(tris) == 2 * (rings - 1) * len(border)
    assert verts[:len(border)] == border  # exact seam with the terrain
    outer = verts[-len(border):]
    assert max(abs(v[0]) for v in outer) == pytest.approx(50.0 + env.RING_OUT_CM[-1])
    for a, b, c in tris:
        (ax, ay, _), (bx, by, _), (cx, cy, _) = verts[a], verts[b], verts[c]
        assert (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) > 0


def test_far_profile_rises_to_the_west(env):
    assert env.far_height(-1.0, 0.0) > env.far_height(0.0, 1.0) >= env.far_height(1.0, 0.0)


def test_terrain_grid_interpolates_and_measures_slope(env):
    grid = env.TerrainGrid(_grid(n=5, size=100.0))  # z = 0.1 x + 0.01 y
    assert grid.height(10.0, 20.0) == pytest.approx(1.2)
    assert grid.normal_z(0.0, 0.0) == pytest.approx(1 / math.sqrt(1 + 0.01 + 0.0001))
    assert grid.contains(50.0, -50.0) and not grid.contains(51.0, 0.0)


def test_value_noise_is_smooth_and_bounded(env):
    vals = [env.value_noise(x * 10.0, 3.0, 100.0, 7) for x in range(200)]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert max(abs(a - b) for a, b in zip(vals, vals[1:])) < 0.2
    assert env.value_noise(5.0, 5.0, 100.0, 7) == env.value_noise(5.0, 5.0, 100.0, 7)


def test_scrub_avoids_scan_roads_and_cliffs(env):
    size = 300000.0
    flat = env.TerrainGrid(_flat_grid(size))
    blocked = env.occupied_cells([(50000.0, 50000.0, 0.0)], env.PARK_CELL_CM, dilate=3)
    bushes = env.scrub_transforms(flat, blocked)
    assert bushes, "a flat open grid should grow scrub"
    for x, y, z, yaw, sx, sy, sz in bushes:
        assert abs(x) <= size / 2 + 300 and abs(y) <= size / 2 + 300  # a lobe may lean out
        assert math.hypot(x - env.SCAN_CENTRE_CM[0], y - env.SCAN_CENTRE_CM[1]) >= env.SCAN_CLEAR_CM * 0.95
        cell = (int(math.floor(x / env.PARK_CELL_CM)), int(math.floor(y / env.PARK_CELL_CM)))
        assert cell not in env.occupied_cells([(50000.0, 50000.0, 0.0)], env.PARK_CELL_CM, dilate=2)
        assert 0.5 < sx < 6.0 and 0.3 < sz < 6.0
    assert env.scrub_transforms(flat, blocked) == bushes  # seeded: same every rebuild
    cliff = env.TerrainGrid([(x, y, x * 2.0) for x, y, _ in _flat_grid(size)])
    assert env.scrub_transforms(cliff, set()) == []


def _flat_grid(size, n=31):
    return [(-size / 2 + size * i / (n - 1), -size / 2 + size * j / (n - 1), 0.0)
            for j in range(n) for i in range(n)]
