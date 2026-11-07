"""Offline tests for the look scripts' plain-Python parts: sun angle, the
render-settings ini patch, and the terrain skirt geometry."""

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
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
    assert math.isclose(-y / h, north, abs_tol=1e-9)  # level Y is south


def test_configured_sun_is_a_summer_late_morning_behind_the_spawn(lighting):
    """The photo board's default light: a clear summer late morning, the sun high
    in the south-east (short shadows, white light, as in most Commons photos).
    The player spawns facing WSW up the bowl, so the light must still travel
    west (x < 0) to fall on what they see, not backlight it."""
    pitch, yaw = lighting.sun_rotation(lighting.SUN_AZIMUTH_DEG, lighting.SUN_ELEVATION_DEG)
    x, _, _ = _direction(pitch, yaw)
    assert 45.0 <= lighting.SUN_ELEVATION_DEG <= 70.0
    assert 100.0 <= lighting.SUN_AZIMUTH_DEG <= 160.0  # morning: east of south
    assert x < 0  # light travels west: the sun is in the east, behind the player


def test_air_is_clear_enough_to_see_denver(lighting):
    """The postcard view shows downtown Denver crisp at ~25 km: the haze must be
    near Earth's default and the height fog thin, not the old golden-hour murk."""
    assert lighting.MIE_SCATTERING_SCALE <= 0.004
    assert lighting.AERIAL_PERSPECTIVE_SCALE <= 1.0
    assert lighting.FOG_DENSITY <= 0.01


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
    # far_height takes a position (cm from the centre) since the east follows the
    # plains by distance; this compared unit directions (1 cm), where the plains
    # profile is at its Morrison end. Compared at 20 km now.
    km20 = 2000000.0
    assert env.far_height(-km20, 0.0) > env.far_height(0.0, km20) >= env.far_height(km20, 0.0)


BASE_M = 1742.098


def _far_m(env, east_km, north_km=0.0):
    return BASE_M + env.far_height(east_km * 100000.0, north_km * 100000.0, BASE_M) / 100.0


@pytest.mark.parametrize("east_km,metres", [(2, 1780.0), (12, 1650.0), (24, 1600.0), (60, 1565.0)])
def test_far_profile_east_follows_the_plains(env, east_km, metres):
    """Morrison ~1.78 km, 12 km out ~1.65 km, Denver (24 km) 1.60 km, ~1.55-1.58 km beyond."""
    assert _far_m(env, east_km) == pytest.approx(metres, abs=30.0)


def test_far_profile_east_is_smooth_and_falls(env):
    xs = np.linspace(1.8, 125.0, 400)
    zs = np.array([_far_m(env, x) for x in xs])
    assert np.all(np.diff(zs) <= 0)
    assert np.max(np.abs(np.diff(zs, 2))) < 1.0  # no kinks
    # Downtown Denver's ground is ~1.59-1.61 km where the skirt passes under it.
    assert 1595.0 < _far_m(env, 20.6 * math.sin(math.radians(65.0)), 20.6 * math.cos(math.radians(65.0))) < 1625.0


@pytest.mark.parametrize("angle_deg", [90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0])
def test_far_profile_west_is_unchanged(env, angle_deg):
    """Due north, every western direction and due south keep the old profile."""
    a = math.radians(angle_deg)
    for r_km in (3.0, 20.0, 120.0):
        dx, dy = r_km * 100000.0 * math.cos(a), r_km * 100000.0 * math.sin(a)
        west = max(0.0, -math.cos(a))
        old = -4000.0 + (90000.0 + 4000.0) * west ** 1.5  # before the plains profile
        assert env.far_height(dx, dy, BASE_M) == pytest.approx(old, abs=1e-6)
        assert env.wave_weight(dx, dy) == pytest.approx(0.35 + 0.65 * west, abs=1e-9)


def test_plains_undulate_by_metres_not_hills(env):
    for angle in np.radians(np.arange(-65.0, 66.0, 5.0)):  # 25 to 155 degrees compass
        w = env.wave_weight(math.cos(angle), math.sin(angle))
        assert max(env.RING_WAVE_CM) * w <= 1500.0  # at most 15 m
    assert max(env.RING_WAVE_CM) * env.wave_weight(-1.0, 0.0) == max(env.RING_WAVE_CM)  # west keeps its hills


def test_skirt_curvature(env):
    assert env.curvature_drop_cm(2500000.0) / 100.0 == pytest.approx(42.67, abs=0.05)  # 25 km
    assert env.curvature_drop_cm(2400000.0) / 100.0 == pytest.approx(39.33, abs=0.05)  # 24 km
    assert env.curvature_drop_cm(12000000.0) / 100.0 == pytest.approx(983.2, abs=0.5)  # 120 km
    level = [(x, y, 500.0) for x, y, _ in _grid()]  # a flat edge: smoothing leaves it at 5 m
    border = env.grid_border(level)
    verts, _, _ = env.skirt_mesh(border, BASE_M)
    n = len(border)
    east = max(range(n), key=lambda i: border[i][0] - abs(border[i][1]))  # the east edge's middle
    for k in range(1, len(env.RING_OUT_CM)):
        x, y, z = verts[k * n + east]
        flat = env.far_height(x, y, BASE_M) * env.RING_BLEND[k] + 500.0 * (1.0 - env.RING_BLEND[k])
        wave = env.RING_WAVE_CM[k] * env.wave_weight(x, y) * env._wave(math.atan2(y, x), k)
        assert z == pytest.approx(flat + wave - env.curvature_drop_cm(math.hypot(x, y)), abs=1.0)
    # The outer ring, 120 km east: the plains at 1.58 km, ~983 m below the flat level.
    x, y, z = verts[(len(env.RING_OUT_CM) - 1) * n + east]
    assert BASE_M + z / 100.0 == pytest.approx(1580.0 - 983.2, abs=2.0)


def test_skirt_reads_the_base_from_the_manifest(env, tmp_path):
    (tmp_path / "manifest.json").write_text('{"terrain_mesh": {"min_elevation_m": 1700.5}}')
    assert env.terrain_base_m(str(tmp_path / "manifest.json")) == 1700.5
    assert env.terrain_base_m(str(tmp_path / "missing.json")) == env.BASE_M
