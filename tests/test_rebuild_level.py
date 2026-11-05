"""Offline tests for unreal/rebuild_level.py: step list and log parsing."""

import importlib.util
import os
from pathlib import Path

import pytest

UNREAL = Path(__file__).resolve().parent.parent / "unreal"


@pytest.fixture(scope="module")
def rl():
    spec = importlib.util.spec_from_file_location("rebuild_level", UNREAL / "rebuild_level.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # no `unreal` here, so it must not auto-run
    return mod


def test_step_order(rl):
    assert [s for s, _ in rl.STEPS] == [
        "import_redrocks.py", "setup_lighting.py", "configure_level.py",
        "fix_camera.py", "setup_materials.py", "setup_environment.py", "setup_vegetation.py",
        "import_scan.py", "cut_scan_collision.py", "setup_gameplay.py", "setup_credits.py", "setup_snow.py",
        "setup_explore.py",
    ]


def test_steps_exist_and_logs_match_scripts(rl):
    """Each step's log path must be the one the script really writes."""
    for name, log in rl.STEPS:
        src = (UNREAL / name).read_text()
        assert os.path.dirname(log) == rl.LOG_DIR
        expected = 'os.path.join(os.environ.get("REDROCKS_LOG_DIR", "/tmp"), "%s")' % os.path.basename(log)
        assert expected in src, "%s does not write %s" % (name, log)


def test_out_dir_and_log_dir_follow_the_environment():
    """REDROCKS_OUT_DIR / REDROCKS_LOG_DIR point a second project copy elsewhere."""
    for name in ("import_redrocks.py", "setup_environment.py", "setup_gameplay.py",
                 "setup_credits.py", "verify_walkable.py"):
        src = (UNREAL / name).read_text()
        assert 'OUT_DIR = os.environ.get("REDROCKS_OUT_DIR", "/Users/austin/dev/redrocks-ue/out")' in src, name
    for name in ("playtest.py", "playtest_credits.py"):
        assert "REDROCKS_LOG_DIR" in (UNREAL / name).read_text(), name


@pytest.mark.parametrize("text", [
    "imported\nsaved", "ok    M_Sandstone\nskip  Benches\ndone",
    "SKIP Benches: not found\nsaved",
])
def test_parse_ok(rl, text):
    assert rl.parse_log(text) == (rl.OK, "")


@pytest.mark.parametrize("text,fragment", [
    ("a\nERROR: RuntimeError('x')", "ERROR"),
    ("ABORT: scene.usdc not found", "ABORT"),
    ("ok    a\nFAIL  M_Sandstone: boom\ndone", "FAIL"),
    ("   ABORT: indented", "ABORT"),
])
def test_parse_failed(rl, text, fragment):
    status, detail = rl.parse_log(text)
    assert status == rl.FAILED and fragment in detail


def test_parse_missing_or_empty(rl):
    assert rl.parse_log(None)[0] == rl.FAILED
    assert rl.parse_log("  \n")[0] == rl.FAILED


def test_error_word_midline_is_not_failure(rl):
    assert rl.parse_log("saved; no ERROR here")[0] == rl.OK


def _drive(rl, logs, raises=()):
    """Run the real step list with fake scripts. logs: step -> log text."""
    store, ran, cleared = {}, [], []
    steps = rl.STEPS

    def clear(path):
        cleared.append(path)
        store.pop(path, None)

    def run_script(path):
        name = Path(path).name
        ran.append(name)
        if name in raises:
            raise raises[name]
        if logs.get(name) is not None:
            store[dict(steps)[name]] = logs[name]

    ticks = iter(range(0, 1000, 2))
    res = rl.run_steps(steps, "/x", run_script=run_script, read=store.get,
                       clear=clear, clock=lambda: next(ticks), say=lambda *_: None)
    return res, ran, cleared


def test_all_ok(rl):
    res, ran, cleared = _drive(rl, {n: "fine" for n, _ in rl.STEPS})
    assert rl.succeeded(res) and ran == [n for n, _ in rl.STEPS]
    assert cleared == [log for _, log in rl.STEPS]
    assert all(r["seconds"] == 2 for r in res)


def test_stops_at_first_failure(rl):
    logs = {n: "fine" for n, _ in rl.STEPS}
    logs["configure_level.py"] = "ERROR: boom"
    res, ran, _ = _drive(rl, logs)
    assert ran == ["import_redrocks.py", "setup_lighting.py", "configure_level.py"]
    assert [r["status"] for r in res] == [
        rl.OK, rl.OK, rl.FAILED] + [rl.SKIPPED] * (len(rl.STEPS) - 3)
    assert not rl.succeeded(res)
    assert "STOPPED at configure_level.py" in rl.format_summary(res)


def test_stale_log_does_not_pass(rl):
    """A script that writes no log fails even if an old log existed."""
    store = {"/tmp/redrocks_import.log": "old good log"}
    res = rl.run_steps(rl.STEPS[:1], "/x", run_script=lambda p: None,
                       read=store.get, clear=lambda p: store.pop(p, None),
                       say=lambda *_: None)
    assert res[0]["status"] == rl.FAILED


def test_script_exception_and_systemexit_fail(rl):
    for exc in (RuntimeError("x"), SystemExit(1)):
        res, ran, _ = _drive(rl, {"import_redrocks.py": "fine"},
                             raises={"import_redrocks.py": exc})
        assert res[0]["status"] == rl.FAILED and len(ran) == 1


def test_summary_fits_one_screen(rl):
    res, _, _ = _drive(rl, {n: "fine" for n, _ in rl.STEPS})
    text = rl.format_summary(res)
    assert len(text.splitlines()) <= 20 and ("All %d steps OK" % len(rl.STEPS)) in text
    assert all(log in text for _, log in rl.STEPS)


def test_no_auto_run_without_unreal(rl):
    assert rl._IN_UNREAL is False


def _load_gameplay():
    """Import unreal/setup_gameplay.py with a stub `unreal` that has no editor API."""
    import importlib.util
    import sys
    import types
    from pathlib import Path

    stub = types.ModuleType("unreal")
    saved = sys.modules.get("unreal")
    sys.modules["unreal"] = stub
    try:
        path = Path(__file__).resolve().parent.parent / "unreal" / "setup_gameplay.py"
        spec = importlib.util.spec_from_file_location("setup_gameplay_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if saved is None:
            del sys.modules["unreal"]
        else:
            sys.modules["unreal"] = saved


def test_gameplay_spawn_is_on_the_deck_facing_up_the_bowl():
    gp = _load_gameplay()
    stage = {
        "axis_east_north": [1.0, 0.0],
        "front_edge_east_m": 10.0,
        "front_edge_north_m": 20.0,
        "deck_top_elevation_m": 1800.0,
    }
    (x, y, z), yaw = gp.spawn_transform(stage, 1750.0)
    assert (x, y) == (500.0, 2000.0)  # 5 m back from the edge, onto the deck
    assert z == 5000.0 + gp.EYE_LIFT_CM
    assert yaw == 0.0  # facing along the axis, up the bowl


def test_gameplay_credits_include_scan_and_osm():
    gp = _load_gameplay()
    text = gp.credits_text({"attribution": ["Map data (c) OpenStreetMap contributors, ODbL 1.0"]})
    assert "ml0w" in text and "CC BY 4.0" in text
    assert "OpenStreetMap" in text


def test_gameplay_spawn_prefers_the_manifest_spawn_block():
    gp = _load_gameplay()
    manifest = {
        "terrain_mesh": {"min_elevation_m": 1750.0},
        "spawn": {"east_m": 1.0, "north_m": 2.0, "ground_elevation_m": 1801.0, "yaw_deg": 45.0},
    }
    (x, y, z), yaw = gp.spawn_from_manifest(manifest)
    assert (x, y, yaw) == (100.0, 200.0, 45.0)
    assert z == 5100.0 + gp.EYE_LIFT_CM
