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
  aerial    450 m south of the origin, 450 m up, pitch -42, looking north
  then every view in docs/media/vegetation-real/views.json whose camera is set
  (REDROCKS_VIEWS overrides the path): the spots framed to match the Commons
  photos, shot at the view's field of view.
The pawn flies with gravity and collision off. REDROCKS_HIDE_VEGETATION=1
hides the Vegetation actor for a baseline. At each shot ProfileGPU writes the
view's GPU frame breakdown to the editor log. Screenshots:
<project>/Saved/Screenshots/MacEditor/<prefix>veg_<view>.png (HighResShot 1600x900).
Log: $REDROCKS_LOG_DIR (default /tmp)/redrocks_vegetation_shots.log.
"""

import json
import math
import os
import sys
import time

import unreal

_HERE = os.path.dirname(os.path.abspath(globals().get("__file__") or "/Users/austin/dev/redrocks-ue/unreal/x"))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import level_frame  # noqa: E402  (beside this script: terrain frame -> level frame)

OUT_DIR = os.environ.get("REDROCKS_OUT_DIR", "/Users/austin/dev/redrocks-ue/out")
LOG_PATH = os.path.join(os.environ.get("REDROCKS_LOG_DIR", "/tmp"), "redrocks_vegetation_shots.log")
PREFIX = os.environ.get("REDROCKS_SHOT_PREFIX", "")
# Perf runs: e.g. REDROCKS_SCALABILITY=2 REDROCKS_SCREEN_PERCENT=75 (the remote-play preset).
SCALABILITY = os.environ.get("REDROCKS_SCALABILITY", "")
SCREEN_PERCENT = os.environ.get("REDROCKS_SCREEN_PERCENT", "")
VIEWS_PATH = os.environ.get("REDROCKS_VIEWS", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "docs", "media", "vegetation-real", "views.json"))
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


def _load_instances(path):
    """vegetation.json's rows with their (x, y) carried into the level frame."""
    with open(path) as fh:
        inst = json.load(fh)["instances"]
    return {name: [list(level_frame.to_level(r[0], r[1])) + r[2:] for r in rows] for name, rows in inst.items()}


def _clear(inst, x, y, radius_cm):
    """True when no plant stands within radius_cm of (x, y)."""
    for rows in inst.values():
        for r in rows:
            if abs(r[0] - x) < radius_cm and abs(r[1] - y) < radius_cm:
                if math.hypot(r[0] - x, r[1] - y) < radius_cm:
                    return False
    return True


def _pick_pair(path):
    """(juniper xyz, pine xyz): the closest juniper 6-20 m from a pine."""
    inst = _load_instances(path)
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


def matched_views(views):
    """(name, (x, y, z), rotation, fov) in the level frame for each photo-matched view
    whose camera is set. views.json holds cameras in the terrain frame (cm east,
    north, up; yaw counter-clockwise from east), like vegetation.json."""
    out = []
    for v in views:
        c = v["camera"]
        if not any(c[k] for k in ("x", "y", "z")):
            continue
        x, y = level_frame.to_level(float(c["x"]), float(c["y"]))
        out.append((v["name"], (x, y, float(c["z"])),
                    {"pitch": float(c["pitch"]), "yaw": level_frame.level_yaw(float(c["yaw"])), "roll": 0.0},
                    float(c["fov"])))
    return out


def _matched_from_file():
    if not os.path.exists(VIEWS_PATH):
        return []
    with open(VIEWS_PATH) as fh:
        return matched_views(json.load(fh)["views"])


def _views(world, pawn):
    views = []
    m = _manifest()
    s = m["spawn"]
    yaw = level_frame.level_yaw(s["yaw_deg"])
    sx, sy = level_frame.to_level_cm(s["east_m"], s["north_m"])
    fx, fy = level_frame.forward(yaw)
    x, y = sx + fx * 9500.0, sy + fy * 9500.0
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
        # Stand 14-30 m off to one side, where no plant is in the way and the
        # ground does not hide the pair (a sight line to 3 m above them).
        inst = _load_instances(veg)
        gz = _ground(world, mx, my, [pawn]) or 0.0
        cx = cy = cz = None
        for dist in (1600.0, 2000.0, 2400.0, 2800.0, 1300.0):
            for side in (1.0, -1.0):
                tx, ty = mx - side * dy / d * dist, my + side * dx / d * dist
                tz = (_ground(world, tx, ty, [pawn]) or 0.0) + EYE_CM
                if not _clear(inst, tx, ty, 500.0):
                    continue
                hit = unreal.SystemLibrary.line_trace_single(
                    world, unreal.Vector(tx, ty, tz), unreal.Vector(mx, my, gz + 300.0),
                    unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, True, [pawn],
                    unreal.DrawDebugTrace.NONE, True)
                hit = hit.to_tuple() if hit is not None else ()
                if hit and hit[0]:
                    continue
                cx, cy, cz = tx, ty, tz
                break
            if cx is not None:
                break
        if cx is None:
            cx, cy = mx - dy / d * 2000.0, my + dx / d * 2000.0
            cz = (_ground(world, cx, cy, [pawn]) or 0.0) + EYE_CM
        look = math.degrees(math.atan2(my - cy, mx - cx))
        pitch = math.degrees(math.atan2(gz + 450.0 - cz, math.hypot(mx - cx, my - cy)))
        views.append(("hillside", (cx, cy, cz), {"pitch": pitch, "yaw": look, "roll": 0.0}))
        log.append("      hillside pair: juniper (%.0f, %.0f), pine (%.0f, %.0f)" % (jx, jy, px, py))
    views.append(("aerial", level_frame.to_level(0.0, -45000.0) + (45000.0,),
                  {"pitch": -42.0, "yaw": level_frame.compass_yaw(0.0), "roll": 0.0}))
    views += _matched_from_file()
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
    (x, y, z), rot = view[1], view[2]
    pawn.set_actor_location(unreal.Vector(x, y, z - PAWN_EYE_OFFSET_CM), False, True)
    pawn.get_controller().set_control_rotation(unreal.Rotator(**rot))


def _set_fov(world, view):
    """Lock the camera to a matched view's field of view; 0 unlocks it (the default)."""
    fov = view[3] if len(view) > 3 else 0.0
    unreal.SystemLibrary.execute_console_command(world, "fov %.1f" % fov)


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
                # Clean frames at the real frame rate: no HUD text, and no
                # background-editor throttle (it pins the editor to ~8 fps).
                unreal.SystemLibrary.execute_console_command(world, "showhud")
                unreal.SystemLibrary.execute_console_command(world, "t.MaxFPS 0")
                if SCALABILITY:
                    unreal.SystemLibrary.execute_console_command(world, "scalability %s" % SCALABILITY)
                if SCREEN_PERCENT:
                    unreal.SystemLibrary.execute_console_command(world, "r.ScreenPercentage %s" % SCREEN_PERCENT)
                log.append("      scalability %s, screen percentage %s" % (SCALABILITY or "default", SCREEN_PERCENT or "default"))
                try:
                    perf = unreal.get_default_object(unreal.EditorPerformanceSettings)
                    perf.set_editor_property("throttle_cpu_when_not_foreground", False)
                    log.append("      background throttle off")
                except Exception as exc:
                    log.append("      background throttle not changed: %r" % (exc,))
                if os.environ.get("REDROCKS_HIDE_VEGETATION"):
                    # Baseline for the perf note: the same views without the plants.
                    for actor in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor):
                        if actor.get_actor_label() == "Vegetation":
                            actor.set_actor_hidden_in_game(True)
                            log.append("      vegetation hidden (baseline run)")
                state.update(phase="view", mark=now, views=_views(world, pawn), view=0, deltas=[])
            return
        if phase == "view":
            views = state["views"]
            if state["view"] >= len(views):
                _finish()
                return
            view = views[state["view"]]
            if state.get("fov_for") != state["view"]:
                _set_fov(world, view)
                state["fov_for"] = state["view"]
            _hold(pawn, view)
            elapsed = now - state["mark"]
            if elapsed >= HOLD_S - FPS_WINDOW_S:
                if not state["deltas"]:
                    state["fps_t0"] = now
                state["deltas"].append(now)
            if elapsed >= HOLD_S:
                # Frames counted against the wall clock: Slate clamps each tick's
                # delta to 1/8 s, so summed deltas cannot report below 8 fps.
                ts = state["deltas"] or [now - 1.0, now]
                span = max(ts[-1] - state.get("fps_t0", ts[0]), 1e-3)
                fps = (len(ts) - 1) / span if len(ts) > 1 else 0.0
                gaps = [b - a for a, b in zip(ts, ts[1:])] or [span]
                worst = 1.0 / max(gaps)
                name = PREFIX + "veg_" + view[0]
                # GPU time for this view goes to the editor log (search "ProfileGPU"):
                # frame rate alone is meaningless while macOS throttles a background editor.
                unreal.SystemLibrary.execute_console_command(world, "ProfileGPU")
                unreal.SystemLibrary.execute_console_command(world, "HighResShot 1600x900 filename=%s" % name)
                log.append("ok    view %-9s fps %.1f (worst frame %.1f fps, %d frames) -> %s.png"
                           % (view[0], fps, worst, len(ts), name))
                state.update(phase="after_shot", mark=now)
            return
        if phase == "after_shot":
            _hold(pawn, state["views"][state["view"]])
            if now - state["mark"] >= 2.0:
                state.update(phase="view", mark=now, view=state["view"] + 1, deltas=[])
    except Exception as exc:
        log.append("ERROR: %r" % (exc,))
        _finish()


if hasattr(unreal, "register_slate_post_tick_callback"):  # not when tests import the helpers
    handle = unreal.register_slate_post_tick_callback(tick)
