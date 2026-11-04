"""Offline tests for unreal/rebuild_level.py: step list and log parsing."""

import importlib.util
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
        "fix_camera.py", "setup_materials.py", "import_scan.py",
    ]


def test_steps_exist_and_logs_match_scripts(rl):
    """Each step's log path must be the one the script really writes."""
    for name, log in rl.STEPS:
        src = (UNREAL / name).read_text()
        assert '"%s"' % log in src, "%s does not write %s" % (name, log)


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
        rl.OK, rl.OK, rl.FAILED, rl.SKIPPED, rl.SKIPPED, rl.SKIPPED]
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
    assert len(text.splitlines()) <= 20 and "All 6 steps OK" in text
    assert all(log in text for _, log in rl.STEPS)


def test_no_auto_run_without_unreal(rl):
    assert rl._IN_UNREAL is False
