"""Rebuild /Game/RedRocks/Lvl_RedRocks from scratch with one command.

Runs, in order, in this one Unreal Python interpreter:

    import_redrocks.py   (WIPES every actor in the level)
    setup_lighting.py
    configure_level.py
    fix_camera.py
    setup_materials.py
    setup_environment.py (terrain skirt to the horizon)
    setup_vegetation.py  (junipers, ponderosa and scrub from out/vegetation)
    import_scan.py
    cut_scan_collision.py (the seating cut out of the scan's collision)
    setup_gameplay.py    (stage/park meshes, collision, player start, credits)
    setup_credits.py     (press-C credits overlay: input action, widget, actor)
    setup_snow.py        (MPC_Weather; every material whitens with its Snow)
    setup_explore.py     (carts, sleds, snow, teleport spots, RRGameMode)

and stops at the first step whose /tmp/redrocks_*.log reports a failure, then
prints a one-screen summary (step, status, duration, log path).

Environment overrides, for a second project copy built from another checkout:
REDROCKS_OUT_DIR (the `redrocks build` output the steps import, default
/Users/austin/dev/redrocks-ue/out) and REDROCKS_LOG_DIR (where every step's
log goes, default /tmp).

Headless, editor closed:
    UnrealEditor-Cmd RedRocks.uproject -run=pythonscript \
        -script=/Users/austin/dev/redrocks-ue/unreal/rebuild_level.py \
        -unattended -nosplash -stdout

From the editor console:
    py "/Users/austin/dev/redrocks-ue/unreal/rebuild_level.py"

How it works: the step scripts are not importable functions. Each runs its
work at module level and writes its log file when it finishes. They use
absolute paths and never read __file__, so this runner just executes each one
with runpy (cwd and __file__ do not matter to them) and then reads the log.
A step's old log is deleted first so a stale log cannot pass for a new run.
A step that leaves no log at all (crashed before it could write one) fails.

The step list and the log parser (parse_log) are plain Python with no
dependency on `unreal`, so tests/test_rebuild_level.py covers them offline.
"""

import os
import runpy
import sys
import time

# Every step writes its log here; REDROCKS_LOG_DIR keeps a second copy's
# rebuild from reading (or deleting) another rebuild's logs.
LOG_DIR = os.environ.get("REDROCKS_LOG_DIR", "/tmp")


def _log(name):
    return os.path.join(LOG_DIR, "redrocks_%s.log" % name)


# (script file, log path) in the order they must run.
STEPS = [
    ("import_redrocks.py", _log("import")),
    ("setup_lighting.py", _log("lighting")),
    ("configure_level.py", _log("configure")),
    ("fix_camera.py", _log("camera")),
    ("setup_materials.py", _log("materials")),
    ("setup_environment.py", _log("environment")),
    ("setup_vegetation.py", _log("vegetation")),
    ("import_scan.py", _log("scan")),
    ("cut_scan_collision.py", _log("scan_cut")),
    ("setup_gameplay.py", _log("gameplay")),
    ("setup_credits.py", _log("credits")),
    ("setup_snow.py", _log("snow")),
    ("setup_explore.py", _log("explore")),
]

SUMMARY_LOG = _log("rebuild")

OK, FAILED, SKIPPED = "OK", "FAILED", "SKIPPED"

# A log line that starts with one of these is a failure. The scripts write
# "ERROR: ..." (uncaught exception) and "ABORT: ..." (precondition missing);
# setup_materials.py writes "FAIL  <step>: ..." for a step that raised.
_FAIL_PREFIXES = ("ERROR", "ABORT", "FAIL")


def parse_log(text):
    """Return (status, detail) for one step's log text.

    status is OK or FAILED. detail is the first failing line (stripped), or a
    reason for a missing/empty log; "" when OK. text=None means the log file
    does not exist.
    """
    if text is None:
        return FAILED, "no log written (script crashed before finishing?)"
    if not text.strip():
        return FAILED, "empty log"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(_FAIL_PREFIXES):
            return FAILED, stripped
    return OK, ""


def _read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return None


def _clear(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _run_script(path):
    # Fresh globals each time; run_name "__main__" matches a normal script run.
    runpy.run_path(path, run_name="__main__")


def run_steps(steps, unreal_dir, run_script=_run_script, read=_read,
              clear=_clear, clock=time.monotonic, say=print):
    """Run steps in order, stopping at the first failure.

    Returns a list of dicts: step, status, seconds, log, detail. Steps after a
    failure are listed as SKIPPED. run_script/read/clear/clock are injectable
    so the logic is testable without Unreal.
    """
    results = []
    failed = False
    for name, log_path in steps:
        row = {"step": name, "status": SKIPPED, "seconds": 0.0,
               "log": log_path, "detail": ""}
        results.append(row)
        if failed:
            continue
        say("[rebuild] running %s ..." % name)
        clear(log_path)
        start = clock()
        crashed = None
        try:
            run_script(os.path.join(unreal_dir, name))
        except BaseException as exc:  # SystemExit too: report, don't die mid-rebuild
            crashed = "%s raised %r" % (name, exc)
        row["seconds"] = clock() - start
        status, detail = parse_log(read(log_path))
        if crashed:
            status, detail = FAILED, crashed
        row["status"], row["detail"] = status, detail
        if status == FAILED:
            failed = True
    return results


def format_summary(results):
    """One-screen summary table plus the first failure, as a string."""
    lines = ["", "Red Rocks rebuild", "=" * 78,
             "%-20s %-8s %8s  %s" % ("step", "status", "seconds", "log"),
             "-" * 78]
    for r in results:
        lines.append("%-20s %-8s %8.1f  %s" % (
            r["step"], r["status"], r["seconds"], r["log"]))
    lines.append("-" * 78)
    bad = next((r for r in results if r["status"] == FAILED), None)
    if bad:
        lines.append("STOPPED at %s: %s" % (bad["step"], bad["detail"]))
        lines.append("See %s" % bad["log"])
    else:
        lines.append("All %d steps OK in %.1f s." % (
            len(results), sum(r["seconds"] for r in results)))
    return "\n".join(lines)


def succeeded(results):
    return all(r["status"] == OK for r in results)


def _unreal_dir():
    """Folder holding the step scripts: env override, __file__, argv, then default."""
    env = os.environ.get("REDROCKS_UNREAL_DIR")
    if env:
        return env
    here = globals().get("__file__")
    if here:
        return os.path.dirname(os.path.abspath(here))
    for arg in sys.argv:
        if arg.endswith("rebuild_level.py"):
            return os.path.dirname(os.path.abspath(arg))
    return "/Users/austin/dev/redrocks-ue/unreal"


def main():
    results = run_steps(STEPS, _unreal_dir())
    text = format_summary(results)
    print(text)
    try:
        with open(SUMMARY_LOG, "w") as fh:
            fh.write(text)
    except OSError:
        pass
    if _IN_UNREAL:
        (unreal.log if succeeded(results) else unreal.log_error)(text)
    if not succeeded(results):
        raise RuntimeError("Red Rocks rebuild stopped: see %s" % SUMMARY_LOG)


# Inside Unreal, __name__ is not guaranteed to be "__main__" for -script or the
# `py` console, so run whenever `unreal` is importable. Tests import this
# module without Unreal and call the functions directly.
# The repo's own unreal/ folder is importable as a namespace package named
# "unreal" when it is on sys.path (pytest), so check for a real engine API.
try:
    import unreal

    _IN_UNREAL = hasattr(unreal, "EditorAssetLibrary") and hasattr(unreal, "log")
except ImportError:
    _IN_UNREAL = False

if _IN_UNREAL or __name__ == "__main__":
    main()
