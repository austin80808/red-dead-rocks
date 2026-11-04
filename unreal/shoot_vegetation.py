"""Play In Editor, fly the camera to the vegetation views, screenshot, log FPS.

Needs the editor GUI (-ExecCmds, not -ExecutePythonScript):

    REDROCKS_OUT_DIR=<checkout>/out REDROCKS_SHOT_PREFIX=after_ \
    "/Users/Shared/Epic Games/UE_5.8/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor" \
        RedRocks.uproject -nosplash \
        -ini:EditorPerProjectUserSettings:[/Script/UnrealEd.EditorPerformanceSettings]:bThrottleCPUWhenNotForeground=False \
        -ExecCmds="py <checkout>/unreal/shoot_vegetation.py"

The -ini override stops a background editor being throttled to ~8 FPS, so the
logged frame rate is real.

Views (each held 7 s so auto exposure settles; FPS is the mean over the last 3 s):
  bowl_top  eye level at the top of the bowl, looking out over the stage (ENE)
  hillside  a juniper and a ponderosa a few metres apart, from 20 m, eye level
            (picked from out/vegetation/vegetation.json)
  aerial    (0, -45000, 45000) cm, pitch -42, yaw 90
The pawn flies with gravity and collision off. Screenshots:
<project>/Saved/Screenshots/MacEditor/<prefix>veg_<view>.png (HighResShot 1600x900).
Log: $REDROCKS_SHOT_LOG or /tmp/redrocks_vegetation_shots.log.
"""

import json
import math
import os
import time

import unreal

OUT_DIR = os.environ.get("REDROCKS_OUT_DIR", "/Users/austin/dev/redrocks-ue/out")
LOG_PATH = os.environ.get("REDROCKS_SHOT_LOG", "/tmp/redrocks_vegetation_shots.log")
PREFIX = os.environ.get("REDROCKS_SHOT_PREFIX", "")
HOLD_S = 7.0
FPS_WINDOW_S = 3.0
SETTLE_S = 6.0
TIMEOUT_S = 300.0
EYE_CM = 170.0
PAWN_EYE_OFFSET_CM = 64.0  # first-person camera above the capsule centre

log = []
state = {"phase": "start", "t0": time.time(), "mark": None, "view": 0, "deltas": []}
handle = None


def _manifest():
    with open(os.path.join(OUT_DIR, "manifest.json")) as fh:
        return json.load(fh)


def _pick_pair(path):
    """(juniper xyz, pine xyz): the closest juniper 6-20 m from a pine."""
    with open(path) as fh:
        inst = json.load(fh)["instances"]
    cells = {}
    for name, rows in inst.items():
        if name.startswith("Juniper"):
            for r in rows:
                cells.setdefault((int(r[0] // 2000), int(r[1] // 2000)), []).append(r)
    best = None
    for name, rows in sorted(inst.items()):
        if not name.startswith("Ponderosa"):
            continue
        for p in rows:
            if p[7] < 0.85:
                continue  # a well-grown pine
            ci, cj = int(p[0] // 2000), int(p[1] // 2000)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for j in cells.get((ci + di, cj + dj), []):
                        d = math.hypot(j[0] - p[0], j[1] - p[1])
                        if 600 <= d <= 2000 and j[7] > 0.8 and (best is None or d < best[0]):
                            best = (d, (j[0], j[1], j[2]), (p[0], p[1], p[2]))
    if best is None:
        return None
    return best[1], best[2]


def _ground(world, x, y, ignore):
    hit = unreal.SystemLibrary.line_trace_single(
        world, unreal.Vector(x, y, 200000.0), unreal.Vector(x, y, -50000.0),
        unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, True, ignore, unreal.DrawDebugTrace.NONE, True)
    t = hit.to_tuple() if hit is not None else ()
    return t[4].z if t and t[0] else None


def _views(world, pawn):
    views = []
    m = _manifest()
    s = m["spawn"]
    yaw = s["yaw_deg"]
    x = s["east_m"] * 100 + math.cos(math.radians(yaw)) * 9500.0
    y = s["north_m"] * 100 + math.sin(math.radians(yaw)) * 9500.0
    z = _ground(world, x, y, [pawn]) or (s["ground_elevation_m"] - m["terrain_mesh"]["min_elevation_m"] + 30.0) * 100
    views.append(("bowl_top", (x, y, z + EYE_CM), {"pitch": -4.0, "yaw": yaw + 180.0, "roll": 0.0}))
    pair = None
    veg = os.path.join(OUT_DIR, "vegetation", "vegetation.json")
    if os.path.exists(veg):
        pair = _pick_pair(veg)
    if pair:
        (jx, jy, _), (px, py, _) = pair
        mx, my = (jx + px) / 2, (jy + py) / 2
        dx, dy = px - jx, py - jy
        d = math.hypot(dx, dy) or 1.0
        cx, cy = mx - dy / d * 2200.0, my + dx / d * 2200.0
        cz = (_ground(world, cx, cy, [pawn]) or 0.0) + EYE_CM
        gz = _ground(world, mx, my, [pawn]) or cz
        look = math.degrees(math.atan2(my - cy, mx - cx))
        pitch = math.degrees(math.atan2(gz + 500.0 - cz, math.hypot(mx - cx, my - cy)))
        views.append(("hillside", (cx, cy, cz), {"pitch": pitch, "yaw": look, "roll": 0.0}))
        log.append("      hillside pair: juniper (%.0f, %.0f), pine (%.0f, %.0f)" % (jx, jy, px, py))
    views.append(("aerial", (0.0, -45000.0, 45000.0), {"pitch": -42.0, "yaw": 90.0, "roll": 0.0}))
    return views


def _finish():
    global handle
    with open(LOG_PATH, "w") as fh:
        fh.write("\n".join(log) + "\n")
    for line in log:
        unreal.log(line)
    if handle is not None:
        unreal.unregister_slate_post_tick_callback(handle)
        handle = None
    try:
        unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).editor_request_end_play()
    except Exception:
        pass
    unreal.SystemLibrary.quit_editor()


def _game_world():
    try:
        return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
    except Exception:
        return None


def _hold(pawn, view):
    _, (x, y, z), rot = view
    pawn.set_actor_location(unreal.Vector(x, y, z - PAWN_EYE_OFFSET_CM), False, True)
    pawn.get_controller().set_control_rotation(unreal.Rotator(**rot))


def tick(delta):
    now = time.time()
    try:
        if now - state["t0"] > TIMEOUT_S:
            log.append("ERROR: timed out in phase %s" % state["phase"])
            _finish()
            return
        phase = state["phase"]
        if phase == "start":
            unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).editor_request_begin_play()
            state["phase"] = "wait_pawn"
            return
        world = _game_world()
        pawn = unreal.GameplayStatics.get_player_pawn(world, 0) if world else None
        if pawn is None:
            return
        if phase == "wait_pawn":
            state.update(phase="settle", mark=now)
            return
        if phase == "settle":
            if now - state["mark"] >= SETTLE_S:
                log.append("      pawn %s" % pawn.get_class().get_name())
                if isinstance(pawn, unreal.Character):
                    move = pawn.get_editor_property("character_movement")
                    move.set_editor_property("gravity_scale", 0.0)
                    move.set_movement_mode(unreal.MovementMode.MOVE_FLYING)
                pawn.set_actor_enable_collision(False)
                state.update(phase="view", mark=now, views=_views(world, pawn), view=0, deltas=[])
            return
        if phase == "view":
            views = state["views"]
            if state["view"] >= len(views):
                _finish()
                return
            view = views[state["view"]]
            _hold(pawn, view)
            elapsed = now - state["mark"]
            if elapsed >= HOLD_S - FPS_WINDOW_S:
                state["deltas"].append(delta)
            if elapsed >= HOLD_S:
                ds = state["deltas"] or [1.0]
                fps = len(ds) / sum(ds)
                worst = 1.0 / max(ds)
                name = PREFIX + "veg_" + view[0]
                unreal.SystemLibrary.execute_console_command(world, "HighResShot 1600x900 filename=%s" % name)
                log.append("ok    view %-9s fps %.1f (worst frame %.1f fps, %d frames) -> %s.png"
                           % (view[0], fps, worst, len(ds), name))
                state.update(phase="after_shot", mark=now)
            return
        if phase == "after_shot":
            _hold(pawn, state["views"][state["view"]])
            if now - state["mark"] >= 2.0:
                state.update(phase="view", mark=now, view=state["view"] + 1, deltas=[])
    except Exception as exc:
        log.append("ERROR: %r" % (exc,))
        _finish()


handle = unreal.register_slate_post_tick_callback(tick)
