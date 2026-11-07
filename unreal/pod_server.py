"""Keep one editor up on the render pod and run queued jobs in it.

Starting UnrealEditor on the pod's network volume costs 3-5 minutes before any
work (asset registry, DDC, shader maps), so iterating launch by launch left the
4090 idle most of the time. This script stays registered on the Slate tick and
runs jobs dropped into $REDROCKS_QUEUE (default /workspace/work/redrocks/queue):

    <queue>/<name>.json   {"script": "/abs/script.py", "env": {"KEY": "value"}}

Jobs run one at a time, oldest first. The env is set for the job's whole life
(shoot_board.py reads it on its first tick, after run_path returns) and
REDROCKS_BOARD_STAY=1 keeps a shooter from quitting the editor. A job whose
globals hold ``state`` (shoot_board.py) is done when ``state["finished"]``;
any other job is done when run_path returns. Each job's outcome is appended to
<queue>/done.log and its file moves to <queue>/done/. A file named STOP quits
the editor.

    UnrealEditor RedRocks.uproject -RenderOffscreen ... -ExecCmds="py <checkout>/unreal/pod_server.py"
"""

import json
import os
import runpy
import time
import traceback

import unreal

QUEUE = os.environ.get("REDROCKS_QUEUE", "/workspace/work/redrocks/queue")
POLL_S = 2.0
srv = {"job": None, "next": 0.0, "busy": False}


def _note(text):
    line = "%s %s" % (time.strftime("%H:%M:%S", time.gmtime()), text)
    unreal.log("pod_server: " + line)
    with open(os.path.join(QUEUE, "done.log"), "a") as fh:
        fh.write(line + "\n")


def _finish_job(ok, why=""):
    job = srv["job"]
    for k, old in job["saved_env"].items():
        if old is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = old
    os.makedirs(os.path.join(QUEUE, "done"), exist_ok=True)
    try:
        os.replace(job["path"], os.path.join(QUEUE, "done", os.path.basename(job["path"])))
    except OSError:
        pass
    _note("%s %s %.0fs %s" % ("done" if ok else "FAIL", os.path.basename(job["path"]),
                              time.time() - job["t0"], why))
    srv["job"] = None


def _start(path):
    with open(path) as fh:
        spec = json.load(fh)
    env = dict(spec.get("env") or {})
    env.setdefault("REDROCKS_BOARD_STAY", "1")
    job = {"path": path, "t0": time.time(), "saved_env": {k: os.environ.get(k) for k in env}, "globals": None}
    srv["job"] = job
    os.environ.update({k: str(v) for k, v in env.items()})
    _note("start %s (%s)" % (os.path.basename(path), spec["script"]))
    job["globals"] = runpy.run_path(spec["script"], run_name="__main__")


def tick(delta):
    if srv["busy"]:
        return
    srv["busy"] = True
    try:
        job = srv["job"]
        if job is not None:
            st = (job.get("globals") or {}).get("state")
            if not isinstance(st, dict) or st.get("finished"):
                _finish_job(True)
            return
        now = time.time()
        if now < srv["next"]:
            return
        srv["next"] = now + POLL_S
        if os.path.exists(os.path.join(QUEUE, "STOP")):
            os.remove(os.path.join(QUEUE, "STOP"))
            _note("stop")
            unreal.SystemLibrary.quit_editor()
            return
        jobs = sorted((os.path.getmtime(os.path.join(QUEUE, f)), os.path.join(QUEUE, f))
                      for f in os.listdir(QUEUE) if f.endswith(".json"))
        if jobs:
            try:
                _start(jobs[0][1])
            except Exception as exc:
                _note("error %r\n%s" % (exc, traceback.format_exc()))
                if srv["job"] is not None:
                    _finish_job(False, repr(exc))
    finally:
        srv["busy"] = False


os.makedirs(QUEUE, exist_ok=True)
_note("ready, watching %s" % QUEUE)
unreal.register_slate_post_tick_callback(tick)
