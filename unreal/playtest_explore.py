"""Play every way of exploring Red Rocks, scripted, and screenshot each.

Needs the editor GUI (Play In Editor does not run in a commandlet):

    "/Users/Shared/Epic Games/UE_5.8/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor" \
        RedRocks.uproject -ExecCmds="py /Users/austin/dev/redrocks-ue/unreal/playtest_explore.py"

It drives the ARRExplorer through its test hooks (ScriptMove / ScriptPress /
ScriptLook / RRGo), the same code paths the keys use:

  climb     teleport to the foot of Creation Rock, push into the face and climb
  wingsuit  teleport to the summit, run off the edge, Space for the wingsuit,
            fly, Space again for the parachute, land
  cart      teleport to the cart lot, climb into a shopping cart, push off and
            ride Red Rocks Park Road down to Morrison, steering along the
            manifest's road route like a player would
  sled      make it snow, teleport to the sled hill, sled the fall line
  morrison  look around downtown
  seating   walk up a row-end aisle, stair by stair, and photograph the rows

Results: /tmp/redrocks_playtest_explore.log ("ok"/"FAIL" lines with numbers;
a stalled aisle walk also logs what stands in front of it), the aisle walk's
path every 0.1 s in /tmp/walk_traj.json ([t, x, y, feet z, aim yaw, vx, vy, vz,
mode, dropping, metres along, leg]), screenshots redrocks_explore_*.png in
<project>/Saved/Screenshots/. REDROCKS_LOG_DIR moves the logs.
Pass SCENARIOS=cart,sled (env) to run a subset.
"""

import json
import math
import os
import time

import unreal

OUT_DIR = os.environ.get("REDROCKS_OUT_DIR", "/Users/austin/dev/redrocks-ue/out")
LOG_PATH = os.path.join(os.environ.get("REDROCKS_LOG_DIR", "/tmp"), "redrocks_playtest_explore.log")
ALL = ["seating", "bowl", "climb", "wingsuit", "cart", "sled", "tubing", "night", "morrison"]
SCENARIOS = [s for s in os.environ.get("SCENARIOS", ",".join(ALL)).split(",") if s]
TIMEOUT_S = 900.0
SETTLE_S = 5.0
# FILM=1 takes a frame every FILM_EVERY_S while climbing, flying and riding, for motion GIFs.
FILM = os.environ.get("FILM") == "1"
FILM_EVERY_S = float(os.environ.get("FILM_EVERY_S", "0.6"))

log = []
handle = None
try:
    with open(os.path.join(OUT_DIR, "manifest.json")) as _fh:
        manifest = json.load(_fh)
except (OSError, ValueError):
    manifest = {}
explore = manifest.get("explore") or {"spots": []}
SPOT = {s["key"]: i + 1 for i, s in enumerate(explore["spots"])}


def check(label, ok, detail=""):
    log.append("%s  %s%s" % ("ok   " if ok else "FAIL ", label, (": " + detail) if detail else ""))


def note(text):
    log.append("      " + text)


def _finish():
    global handle
    with open(LOG_PATH, "w") as fh:
        fh.write("\n".join(log) + "\n")
    for line in log:
        (unreal.log_error if line.startswith(("FAIL", "ERROR")) else unreal.log)(line)
    if handle is not None:
        unreal.unregister_slate_post_tick_callback(handle)
        handle = None
    try:
        unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).editor_request_end_play()
    except Exception:
        pass
    unreal.SystemLibrary.quit_editor()


def _world():
    try:
        return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
    except Exception:
        return None


def _shot(world, name, ui=True):
    if ui:
        unreal.SystemLibrary.execute_console_command(world, "Shot showui filename=redrocks_explore_%s" % name)
    else:
        unreal.SystemLibrary.execute_console_command(world, "HighResShot 1600x900 filename=redrocks_explore_%s" % name)
    note("screenshot %s" % name)


def _film(world, now, tag):
    if not FILM:
        return
    last = R.s.get("film_t", 0.0)
    if now - last >= FILM_EVERY_S:
        n = R.s.get("film_n", 0)
        unreal.SystemLibrary.execute_console_command(
            world, "HighResShot 960x540 filename=redrocks_film_%s_%03d" % (tag, n))
        R.s["film_t"] = now
        R.s["film_n"] = n + 1


def _mode(pawn):
    m = pawn.get_mode()
    name = getattr(m, "name", None) or str(m)
    # str() of an unreal enum looks like "<RRMode.WALK: 0>".
    return name.split(".")[-1].split(":")[0].strip("<> ").upper()


def steer_toward(ride_yaw, from_xy, route_cm, idx, lookahead_cm=2500.0):
    """(steer -1..1, new index) to follow a polyline. Plain maths, tested offline."""
    # Advance the target past points we are already near or beyond.
    while idx < len(route_cm) - 1 and math.hypot(route_cm[idx][0] - from_xy[0], route_cm[idx][1] - from_xy[1]) < lookahead_cm:
        idx += 1
    tx, ty = route_cm[idx][0], route_cm[idx][1]
    want = math.degrees(math.atan2(ty - from_xy[1], tx - from_xy[0]))
    diff = (want - ride_yaw + 180.0) % 360.0 - 180.0
    return max(-1.0, min(1.0, diff / 25.0)), idx


class Runner:
    """One scenario at a time; each is a list of (name, fn(now) -> True when done)."""

    def __init__(self):
        self.t0 = time.time()
        self.queue = list(SCENARIOS)
        self.steps = []
        self.mark = None
        self.s = {}

    def wait(self, seconds):
        def fn(now):
            return now - self.mark >= seconds
        return fn


R = Runner()


def _start_scenario(pawn, world, name):
    note("--- %s" % name)
    R.s = {"name": name, "t": time.time(), "maxz": -1e9, "modes": set(), "samples": []}
    pawn.script_release()
    unreal.SystemLibrary.execute_console_command(world, "DisableAllScreenMessages")
    if name == "climb":
        pawn.rr_go(SPOT["climb"])
        R.steps = [("settle", R.wait(SETTLE_S)), ("climb", climb_step), ("climb_end", climb_end)]
    elif name == "wingsuit":
        pawn.rr_go(SPOT["summit"])
        R.steps = [("settle", R.wait(SETTLE_S)), ("summit_shot", summit_shot), ("run_off", run_off),
                   ("fly", fly_step), ("chute", chute_step), ("wing_end", wing_end)]
    elif name == "cart":
        pawn.rr_go(SPOT["carts"])
        R.steps = [("settle", R.wait(SETTLE_S)), ("board", board_step), ("ride", ride_step), ("ride_end", ride_end)]
    elif name == "sled":
        weather = unreal.GameplayStatics.get_actor_of_class(world, unreal.RRWeather)
        if weather:
            weather.set_snow_now(True)
        pawn.rr_go(SPOT["sleds"])
        R.steps = [("settle", R.wait(SETTLE_S + 3.0)), ("snow_shot", snow_shot), ("board", board_step),
                   ("ride", ride_step), ("ride_end", ride_end), ("melt", melt)]
    elif name == "tubing":
        pawn.rr_go(SPOT["tubing"])
        R.steps = [("settle", R.wait(SETTLE_S)), ("board", board_step), ("ride", ride_step), ("ride_end", ride_end)]
    elif name == "night":
        weather = unreal.GameplayStatics.get_actor_of_class(world, unreal.RRWeather)
        if weather:
            weather.set_night(True)
        R.s["views"] = [("bowl_top", 0.0, -12.0, True), ("stage", 0.0, 10.0, True)]
        R.s["prefix"] = "night"
        # Lumen's cached lighting takes 10-20 s to let go of the daylight.
        R.steps = [("dark", R.wait(18.0)), ("views", bowl_views), ("day", day_again)]
    elif name == "bowl":
        R.s["views"] = [("stage", 0.0, 4.0, False), ("bowl_top", 0.0, -14.0, False), ("bowl_top", 0.0, -14.0, True)]
        R.steps = [("views", bowl_views)]
    elif name == "seating":
        pawn.rr_go(SPOT["stage"])
        R.steps = [("settle", R.wait(SETTLE_S)),
                   ("from_stage", seat_view("seating_stage", _aim_up, 6.0)),
                   ("to_aisle", seat_to_aisle),
                   ("settle_aisle", R.wait(3.0)),
                   ("walk_low", seat_walk(0.5)),
                   ("mid_across", seat_view("seating_mid", _aim_across, 4.0)),
                   ("walk_high", seat_walk(1.0)),
                   ("walk_end", seat_walk_end),
                   ("from_top", seat_view("seating_top", _aim_down, -16.0))]
    elif name == "morrison":
        pawn.rr_go(SPOT["morrison"])
        R.steps = [("settle", R.wait(SETTLE_S + 3.0)), ("look", morrison_shot)]
    R.mark = time.time()


# ---- climb -----------------------------------------------------------------

def climb_step(now, pawn, world):
    loc = pawn.get_actor_location()
    if "z0" not in R.s:
        R.s["z0"] = loc.z
        pawn.script_press("Camera")  # chase camera: see the climber on the rock
        pawn.script_move(unreal.Vector2D(0.0, 1.0), True)
    R.s["modes"].add(_mode(pawn))
    R.s["maxz"] = max(R.s["maxz"], loc.z)
    if _mode(pawn) == "CLIMB":
        _film(world, now, "climb")
    if int(now - R.mark) % 5 == 0 and int(now - R.mark) not in R.s.setdefault("noted", set()):
        R.s["noted"].add(int(now - R.mark))
        note("t=%2ds z=%.0f mode %s probe %s" % (now - R.mark, loc.z, _mode(pawn), pawn.get_climb_probe()))
    if not R.s.get("shot") and _mode(pawn) == "CLIMB" and loc.z - R.s["z0"] > 800:
        _shot(world, "climb")
        R.s["shot"] = True
    return now - R.mark > 45.0 or (R.s["maxz"] - R.s["z0"] > 3000 and _mode(pawn) == "WALK" and loc.z > R.s["z0"] + 2500)


def climb_end(now, pawn, world):
    pawn.script_release()
    gain = (R.s["maxz"] - R.s["z0"]) / 100.0
    check("climbed Creation Rock", "CLIMB" in R.s["modes"] and gain >= 10.0,
          "gained %.1f m, modes %s" % (gain, sorted(R.s["modes"])))
    _shot(world, "climb_high")
    pawn.script_press("Camera")
    return True


# ---- seating -----------------------------------------------------------------
# Walk from the stage up one of the row-end aisles, stair step by stair step
# (manifest stage.aisles, from redrocks.stage), with no jumping, and photograph
# the three views the Commons photos were taken from: up from the stage, across
# the rows from mid-aisle, down from the top row.

# The walk starts at row 8's stairs: below that the aisles start in the stage
# blockout's front corner (its deck stands higher than the first steps); that
# is for the stage pass, not the seating.
AISLE_FROM_ROW = 8


def _aisle_steps():
    """The more continuous of the two aisles (the smallest sideways jump between
    steps), from row AISLE_FROM_ROW up: [(east, north, z)] in metres."""
    aisles = (manifest.get("stage") or {}).get("aisles") or {}
    levels = (manifest.get("benches") or {}).get("tread_levels_m") or []
    floor = levels[AISLE_FROM_ROW - 2] if len(levels) >= AISLE_FROM_ROW else -1e9
    best, best_jump = [], None
    for steps in aisles.values():
        steps = [p for p in steps if p[2] > floor + 0.01]
        if len(steps) < 2:
            continue
        jump = max(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(steps, steps[1:]))
        if best_jump is None or jump < best_jump:
            best, best_jump = steps, jump
    return best


def _aisle():
    return [(e * 100.0, n * 100.0) for e, n, _ in _aisle_steps()]


def _other_aisle_at(fraction):
    """A point on the far aisle, ``fraction`` of the way up: what 'across the rows' looks at."""
    aisles = list(((manifest.get("stage") or {}).get("aisles") or {}).values())
    mine = _aisle()
    for steps in aisles:
        pts = [(e * 100.0, n * 100.0) for e, n, _ in steps]
        if pts and pts != mine:
            return pts[min(len(pts) - 1, int(fraction * len(pts)))]
    return None


def _seat_levels_cm():
    b = manifest.get("benches") or {}
    levels = b.get("tread_levels_m") or [0.0, 0.0]
    base = b.get("base_elevation_m", 0.0)
    return (levels[0] - base) * 100.0, (levels[-1] - base) * 100.0


def _yaw_to(pawn, x, y):
    loc = pawn.get_actor_location()
    return math.degrees(math.atan2(y - loc.y, x - loc.x))


def seat_view(name, aim, pitch):
    """Stop, aim (a function of the pawn giving a yaw), wait for streaming, shoot."""
    def fn(now, pawn, world):
        if "view_at" not in R.s:
            pawn.script_move(unreal.Vector2D(0.0, 0.0), False)
            R.s["view_at"] = now
            pawn.script_look(aim(pawn), pitch)
            return False
        if now - R.s["view_at"] > 5.0 and not R.s.get("view_shot"):
            _shot(world, name, ui=False)
            R.s["view_shot"] = True
            return False
        if now - R.s["view_at"] > 7.0:
            R.s.pop("view_at")
            R.s.pop("view_shot", None)
            return True
        return False
    return fn


def _aim_up(pawn):
    path = _aisle()
    i = min(R.s.get("wp", 0), len(path) - 1)
    return _yaw_to(pawn, *path[min(i + 4, len(path) - 1)]) if path else pawn.get_actor_rotation().yaw


def _aim_across(pawn):
    far = _other_aisle_at(0.5)
    return _yaw_to(pawn, *far) if far else pawn.get_actor_rotation().yaw + 90.0


def _aim_down(pawn):
    stage = next(sp for sp in explore["spots"] if sp["key"] == "stage")
    return _yaw_to(pawn, stage["east_m"] * 100.0, stage["north_m"] * 100.0)


def seat_to_aisle(now, pawn, world):
    """Stand on the aisle at row AISLE_FROM_ROW, 1.2 m before that step."""
    path = _aisle()
    steps = _aisle_steps()
    if not path:
        return True
    first = steps[0]
    (x0, y0), (x1, y1) = path[0], path[1]
    d = math.hypot(x1 - x0, y1 - y0) or 1.0
    base = (manifest.get("benches") or {}).get("base_elevation_m", 0.0)
    z = (first[2] - base) * 100.0 + 130.0
    pawn.set_actor_location(unreal.Vector(x0 - (x1 - x0) / d * 120.0, y0 - (y1 - y0) / d * 120.0, z), False, True)
    note("at the foot of the aisle (%.0f, %.0f)" % (x0 / 100.0, y0 / 100.0))
    return True


LOOKAHEAD_CM = 70.0
# The play-test steers once per Slate tick, about 3 Hz in the editor with this
# level (logged 2026-11-04), and the explorer walks 520 cm/s at full input:
# 1.7 m straight between steers, more than a stair leg (0.62 m), so it ran off
# the aisle at every bend. Pace it to cover at most STEER_STEP_CM per tick.
WALK_CM_S = 520.0
STEER_STEP_CM = 25.0
MIN_PACE = 0.1


def walk_pace(tick_s):
    """Stick input (0..1) that moves the walker at most STEER_STEP_CM per tick."""
    if tick_s <= 0.0:
        return 1.0
    return max(MIN_PACE, min(1.0, STEER_STEP_CM / (tick_s * WALK_CM_S)))


def _lengths(path):
    cum = [0.0]
    for (ax, ay), (bx, by) in zip(path, path[1:]):
        cum.append(cum[-1] + math.hypot(bx - ax, by - ay))
    return cum


def _point_at(path, cum, s):
    """(point, leg index) ``s`` cm along the path, clamped to its ends."""
    s = max(0.0, min(s, cum[-1]))
    for k in range(len(path) - 1):
        if s <= cum[k + 1] or k == len(path) - 2:
            leg = (cum[k + 1] - cum[k]) or 1.0
            t = (s - cum[k]) / leg
            (ax, ay), (bx, by) = path[k], path[k + 1]
            return (ax + t * (bx - ax), ay + t * (by - ay)), k
    return path[0], 0


def follow_path(path, xy, s_prev, lookahead=LOOKAHEAD_CM, window=300.0):
    """Pure pursuit along a polyline (cm): (progress s, carrot point, leg index).

    ``s`` is how far along the path the walker is: its nearest point on the
    path, looked for only from where it last was to ``window`` beyond (so a
    leg further on that passes close is not taken for this one), and never
    behind ``s_prev``. The carrot is ``lookahead`` further on: aiming at it
    starts a bend before the corner, on the inside, where aiming at the corner
    itself runs past it to the outside (the aisle's side, then a drop).
    """
    cum = _lengths(path)
    x, y = xy
    best = None
    for k, ((ax, ay), (bx, by)) in enumerate(zip(path, path[1:])):
        if cum[k + 1] < s_prev or cum[k] > s_prev + window:
            continue
        dx, dy = bx - ax, by - ay
        leg2 = dx * dx + dy * dy or 1.0
        t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / leg2))
        d = math.hypot(x - ax - t * dx, y - ay - t * dy)
        if best is None or d < best[0]:
            best = (d, cum[k] + t * math.sqrt(leg2))
    s = max(s_prev, best[1]) if best else s_prev
    carrot, _ = _point_at(path, cum, s + lookahead)
    _, leg = _point_at(path, cum, s)
    return s, carrot, leg


def _probe_ahead(world, pawn, loc, yaw):
    """What stands in front of the walker: forward traces at several heights
    above its feet, the floor under it, its velocity. Logged when it stalls."""
    half = pawn.capsule_component.get_scaled_capsule_half_height()
    feet = loc.z - half
    fx, fy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    out = []
    for h in (5.0, 25.0, 45.0, 70.0, 120.0):
        a = unreal.Vector(loc.x, loc.y, feet + h)
        b = unreal.Vector(loc.x + fx * 120.0, loc.y + fy * 120.0, feet + h)
        hit = unreal.SystemLibrary.line_trace_single(world, a, b, unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, True,
                                                     [pawn], unreal.DrawDebugTrace.NONE, True)
        t = hit.to_tuple() if hit is not None else ()
        if t and t[0]:
            what = next((v for v in t if isinstance(v, unreal.Actor)), None)
            comp = next((v for v in t if isinstance(v, unreal.PrimitiveComponent)), None)
            out.append("+%.0f: %s/%s at %.0f cm, normal z %.2f" % (
                h, what.get_actor_label() if what else "?", comp.get_name() if comp else "?", t[3], t[7].z))
        else:
            out.append("+%.0f: clear" % h)
    a = unreal.Vector(loc.x, loc.y, loc.z)
    hit = unreal.SystemLibrary.line_trace_single(world, a, unreal.Vector(loc.x, loc.y, loc.z - 400.0),
                                                 unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, True, [pawn],
                                                 unreal.DrawDebugTrace.NONE, True)
    t = hit.to_tuple() if hit is not None else ()
    if t and t[0]:
        what = next((v for v in t if isinstance(v, unreal.Actor)), None)
        out.append("floor %s at %+.0f cm from the feet" % (what.get_actor_label() if what else "?", t[5].z - feet))
    v = pawn.get_velocity()
    out.append("velocity %.0f cm/s (%.0f, %.0f, %.0f), yaw %.0f, actor yaw %.0f" % (
        v.length(), v.x, v.y, v.z, yaw, pawn.get_actor_rotation().yaw))
    note("ahead: " + "; ".join(out))


def seat_walk(fraction):
    """Follow the aisle's stair steps until ``fraction`` of the rows' rise is climbed."""
    def fn(now, pawn, world):
        loc = pawn.get_actor_location()
        lo, hi = _seat_levels_cm()
        path = _aisle()
        if not path:
            note("no aisles in the manifest")
            R.s["stuck"] = True
            return True
        if "z0" not in R.s:
            R.s["z0"] = loc.z
            R.s["t_walk"] = now
            R.s["wp"] = 0
        if R.s.get("walk_frac") != fraction:  # a new leg: the view before it stood still
            R.s["walk_frac"] = fraction
            R.s["stall"] = (now, loc.z, R.s["wp"])
        # Pure pursuit: aim at a point LOOKAHEAD_CM further along the stair
        # line from the walker's nearest point on it, so it turns into each
        # bend on the inside instead of running past the corner.
        s, carrot, i = follow_path(path, (loc.x, loc.y), R.s.get("s", 0.0))
        R.s["s"] = s
        R.s["wp"] = i
        pawn.script_look(_yaw_to(pawn, *carrot), 0.0)
        tick_s = now - R.s.get("walk_last", now - 0.33)
        R.s["walk_last"] = now
        pawn.script_move(unreal.Vector2D(0.0, walk_pace(tick_s)), False)
        if now - R.s.get("traj_t", -1.0) >= 0.1:
            R.s["traj_t"] = now
            v = pawn.get_velocity()
            R.s.setdefault("traj", []).append([
                round(now - R.s["t_walk"], 2), round(loc.x, 1), round(loc.y, 1),
                round(loc.z - pawn.capsule_component.get_scaled_capsule_half_height(), 1),
                round(_yaw_to(pawn, *carrot), 1), round(v.x), round(v.y), round(v.z), _mode(pawn),
                _mode(pawn) == "WALK" and v.z < -50.0, round(s), i])
            if len(R.s["traj"]) % 50 == 0:
                with open(os.path.join(os.path.dirname(LOG_PATH), "walk_traj.json"), "w") as fh:
                    json.dump(R.s["traj"], fh)
        R.s["modes"].add(_mode(pawn))
        R.s["maxz"] = max(R.s["maxz"], loc.z)
        _film(world, now, "seating")
        if loc.z > R.s["stall"][1] + 20.0 or i > R.s["stall"][2]:
            R.s["stall"] = (now, loc.z, i)
        elif now - R.s["stall"][0] > 8.0:
            _probe_ahead(world, pawn, loc, _yaw_to(pawn, *carrot))
            note("stuck at (%.2f, %.2f) z=%.0f by step %d of %d (heading for %.2f, %.2f) for 8 s"
                 % (loc.x / 100.0, loc.y / 100.0, loc.z, i, len(path), path[i][0] / 100.0, path[i][1] / 100.0))
            R.s["stuck"] = True
            return True
        feet = loc.z - pawn.capsule_component.get_scaled_capsule_half_height()
        goal = lo + fraction * (hi - lo) - 50.0
        if int(now - R.s["t_walk"]) % 10 == 0 and int(now - R.s["t_walk"]) not in R.s.setdefault("noted", set()):
            R.s["noted"].add(int(now - R.s["t_walk"]))
            (px, py), _ = _point_at(path, _lengths(path), s)
            note("t=%3ds step %d/%d at (%.1f, %.1f) %.0f cm off the stair line, feet z=%.0f (rows %.0f..%.0f) mode %s"
                 % (now - R.s["t_walk"], i, len(path), loc.x / 100.0, loc.y / 100.0,
                    math.hypot(loc.x - px, loc.y - py), feet, lo, hi, _mode(pawn)))
        # Paced to the editor's ~3 Hz tick the 96 m aisle takes about 2-3 minutes;
        # a stop is caught by the 8 s stall check, so this only bounds a crawl.
        return feet >= goal or now - R.s["t_walk"] > 480.0
    return fn


def seat_walk_end(now, pawn, world):
    pawn.script_release()
    lo, hi = _seat_levels_cm()
    levels = (manifest.get("benches") or {}).get("tread_levels_m") or [0.0, 0.0]
    gain = (R.s["maxz"] - R.s["z0"]) / 100.0
    need = (levels[-1] - levels[AISLE_FROM_ROW - 1]) - 1.0
    check("walked up the aisle stairs without jumping", not R.s.get("stuck") and gain >= need
          and R.s["modes"] <= {"WALK"},
          "gained %.1f m of the %.1f m from row %d to the top in %.0f s by step %d, modes %s"
          % (gain, need + 1.0, AISLE_FROM_ROW, now - R.s["t_walk"], R.s.get("wp", 0), sorted(R.s["modes"])))
    return True


# ---- wingsuit ----------------------------------------------------------------

def summit_shot(now, pawn, world):
    loc = pawn.get_actor_location()
    R.s["z0"] = loc.z
    R.s["xy0"] = (loc.x, loc.y)
    note("summit z=%.0f, %.0f m above ground" % (loc.z, pawn.get_height_above_ground_m()))
    pawn.script_look(pawn.get_actor_rotation().yaw, -10.0)
    _shot(world, "summit")
    return True


def run_off(now, pawn, world):
    pawn.script_move(unreal.Vector2D(0.0, 1.0), True)
    falling = _mode(pawn) == "WALK" and pawn.get_movement_component().is_falling()
    if falling and pawn.get_height_above_ground_m() > 8.0:
        pawn.script_press("Jump")
        pawn.script_move(unreal.Vector2D(0.0, 0.0), False)
        R.s["deploy"] = now
        note("wingsuit at %.0f m above ground" % pawn.get_height_above_ground_m())
        return True
    if now - R.mark > 20.0:
        note("never got airborne off the summit (mode %s)" % _mode(pawn))
        return True
    return False


def fly_step(now, pawn, world):
    R.s["modes"].add(_mode(pawn))
    if _mode(pawn) != "WINGSUIT":
        return True
    R.s["vmax"] = max(R.s.get("vmax", 0.0), pawn.get_speed_mps())
    _film(world, now, "wingsuit")
    yaw = pawn.get_control_rotation().yaw
    pawn.script_look(yaw, -12.0)
    if not R.s.get("shot") and now - R.mark > 3.0:
        _shot(world, "wingsuit")
        R.s["shot"] = True
    if now - R.mark > 9.0 and pawn.get_height_above_ground_m() > 40.0:
        pawn.script_press("Jump")  # parachute
        return True
    return now - R.mark > 30.0


def chute_step(now, pawn, world):
    R.s["modes"].add(_mode(pawn))
    if _mode(pawn) == "PARACHUTE":
        _film(world, now, "wingsuit")
    if _mode(pawn) == "PARACHUTE" and not R.s.get("chute_shot") and now - R.mark > 2.0:
        _shot(world, "parachute")
        R.s["chute_shot"] = True
    return _mode(pawn) == "WALK" or now - R.mark > 60.0


def wing_end(now, pawn, world):
    loc = pawn.get_actor_location()
    dist = math.hypot(loc.x - R.s["xy0"][0], loc.y - R.s["xy0"][1]) / 100.0
    check("wingsuit flight", "WINGSUIT" in R.s["modes"] and dist >= 80.0,
          "flew %.0f m, top %.0f km/h, modes %s" % (dist, R.s.get("vmax", 0) * 3.6, sorted(R.s["modes"])))
    check("parachute opened and landed", "PARACHUTE" in R.s["modes"] and _mode(pawn) == "WALK")
    pawn.script_release()
    return True


# ---- rides -------------------------------------------------------------------

def board_step(now, pawn, world):
    pawn.script_press("Interact")
    ride = pawn.get_ride()
    if ride is None:
        check("boarded %s" % R.s["name"], False, "nothing to ride within 4 m")
        R.steps = [("end", lambda *a: True)]
        return True
    start = ride.get_actor_location()
    R.s.update(ride=ride, start=(start.x, start.y, start.z), vmax=0.0, idx=0, crashes=0, dist=0.0,
               last=(start.x, start.y))
    key = {"cart": "carts", "sled": "sleds", "tubing": "tubes"}[R.s["name"]]
    datum = manifest["terrain_mesh"]["min_elevation_m"]
    pts = explore[key]["run" if key == "sleds" else "route"]
    R.s["route"] = [(p[0] * 100.0, p[1] * 100.0, (p[2] - datum) * 100.0) for p in pts]
    note("boarded at x=%.0f y=%.0f z=%.0f" % (start.x, start.y, start.z))
    return True


def ride_step(now, pawn, world):
    ride = R.s["ride"]
    loc = ride.get_actor_location()
    R.s["dist"] += math.hypot(loc.x - R.s["last"][0], loc.y - R.s["last"][1]) / 100.0
    R.s["last"] = (loc.x, loc.y)
    R.s["vmax"] = max(R.s["vmax"], ride.get_speed_kmh())
    if not ride.has_rider():
        # Wiped out: get back on, like a player would.
        R.s["crashes"] += 1
        note("wipeout %d at x=%.0f y=%.0f" % (R.s["crashes"], loc.x, loc.y))
        if R.s["crashes"] > 6:
            return True
        pawn.script_move(unreal.Vector2D(0.0, 0.0), False)
        pawn.set_actor_location(unreal.Vector(loc.x, loc.y, loc.z + 150.0), False, True)
        pawn.script_press("Interact")
        return False
    _film(world, now, R.s["name"])
    steer, R.s["idx"] = steer_toward(ride.get_actor_rotation().yaw, (loc.x, loc.y), R.s["route"], R.s["idx"])
    speed = ride.get_speed_kmh()
    # Push when slow, brake into sharp turns at speed, otherwise coast.
    throttle = 1.0 if speed < 18.0 else (-0.6 if abs(steer) > 0.8 and speed > 40.0 else 0.0)
    pawn.script_move(unreal.Vector2D(steer, throttle), False)
    if int(now - R.mark) % 20 == 0 and int(now - R.mark) not in R.s.setdefault("noted", set()):
        R.s["noted"].add(int(now - R.mark))
        note("t=%3ds x=%.0f y=%.0f z=%.0f %.0f km/h, %.0f m along, target %d/%d" % (
            now - R.mark, loc.x, loc.y, loc.z, speed, R.s["dist"], R.s["idx"], len(R.s["route"])))
    if R.s["name"] == "tubing" and int(now - R.mark) % 20 == 0 and int(now - R.mark) not in R.s.setdefault("buried", set()):
        R.s["buried"].add(int(now - R.mark))
        top = unreal.Vector(loc.x, loc.y, loc.z + 5000.0)
        hit = unreal.SystemLibrary.line_trace_single(world, top, unreal.Vector(loc.x, loc.y, loc.z - 5000.0),
                                                     unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, True, [pawn, ride],
                                                     unreal.DrawDebugTrace.NONE, True)
        t = hit.to_tuple() if hit is not None else ()
        if t and t[0]:
            what = next((v for v in t if isinstance(v, unreal.Actor)), None)
            note("tube z=%.0f, first surface above it: %s at z=%.0f (%+.0f cm)" % (
                loc.z, what.get_actor_label() if what else "?", t[4].z, t[4].z - loc.z))
    for at, name in ((6.0, "%s_start" % R.s["name"]), (40.0, "%s_mid" % R.s["name"])):
        if now - R.mark > at and name not in R.s.setdefault("shots", set()):
            R.s["shots"].add(name)
            _shot(world, name)
    end = R.s["route"][-1]
    left = math.hypot(end[0] - loc.x, end[1] - loc.y) / 100.0
    R.s["left"] = left
    limit = {"cart": 420.0, "tubing": 600.0}.get(R.s["name"], 70.0)
    return left < 60.0 or now - R.mark > limit


def ride_end(now, pawn, world):
    name = R.s["name"]
    _shot(world, "%s_end" % name)
    route_len = {"cart": explore["carts"]["route_length_m"], "tubing": explore["tubes"]["route_length_m"]}.get(
        name, explore["sleds"]["run_m"])
    detail = "%.0f m ridden of %.0f, %.0f m from the end, top %.0f km/h, %d wipeouts" % (
        R.s["dist"], route_len, R.s.get("left", -1), R.s["vmax"], R.s["crashes"])
    if name == "cart":
        check("shopping cart to Morrison", R.s.get("left", 1e9) < 150.0, detail)
    elif name == "tubing":
        check("tubed Bear Creek through Morrison", R.s["dist"] >= 2000.0, detail)
    else:
        check("sledded the hill", R.s["dist"] >= 100.0 and R.s["vmax"] >= 20.0, detail)
    pawn.script_release()
    if R.s["ride"].has_rider():
        pawn.script_press("Interact")
    return True


def snow_shot(now, pawn, world):
    pawn.script_look(pawn.get_actor_rotation().yaw, -8.0)
    _shot(world, "snow")
    return True


def melt(now, pawn, world):
    weather = unreal.GameplayStatics.get_actor_of_class(world, unreal.RRWeather)
    if weather:
        weather.set_snow_now(False)
    return True


def bowl_views(now, pawn, world):
    """The amphitheatre: from the stage up the rows, from the top down to the stage, then under snow.

    A screenshot is taken at the end of the frame it is asked for, so the next
    teleport waits 2 s after each shot.
    """
    at = R.s.get("at")
    if R.s.get("pending") and now - at > 9.0:
        _shot(world, R.s["pending"], ui=False)
        R.s["pending"] = None
        R.s["at"] = now
        return False
    if at is None or (not R.s.get("pending") and now - at > 2.0):
        if not R.s["views"]:
            weather = unreal.GameplayStatics.get_actor_of_class(world, unreal.RRWeather)
            if weather and R.s.get("prefix", "bowl") == "bowl":
                weather.set_snow_now(False)
            check("%s views" % ("amphitheatre" if R.s.get("prefix", "bowl") == "bowl" else R.s["prefix"]), True)
            return True
        key, dyaw, pitch, flag = R.s["views"].pop(0)
        prefix = R.s.get("prefix", "bowl")
        weather = unreal.GameplayStatics.get_actor_of_class(world, unreal.RRWeather)
        if weather and prefix == "bowl":
            weather.set_snow_now(flag)
        pawn.rr_go(SPOT[key])
        R.s["look"] = (dyaw, pitch)
        R.s["pending"] = "%s_%s%s" % (prefix, key, "_snow" if flag and prefix == "bowl" else "")
        R.s["at"] = now
        return False
    if R.s.get("look") and now - at > 6.0:
        dyaw, pitch = R.s["look"]
        pawn.script_look(pawn.get_actor_rotation().yaw + dyaw, pitch)
        R.s["look"] = None
    return False


def day_again(now, pawn, world):
    weather = unreal.GameplayStatics.get_actor_of_class(world, unreal.RRWeather)
    concert = unreal.GameplayStatics.get_actor_of_class(world, unreal.RRConcert)
    check("night concert played", bool(concert) and concert.is_playing())
    if weather:
        weather.set_night(False)
    return True


def morrison_shot(now, pawn, world):
    if "i" not in R.s:
        R.s["i"] = 0
        R.s["at"] = now
        R.s["yaw0"] = pawn.get_actor_rotation().yaw
    if now - R.s["at"] < 1.5:
        return False
    if R.s["i"] >= 2:
        loc = pawn.get_actor_location()
        end = explore["carts"]["route"][-1]
        d = math.hypot(end[0] * 100 - loc.x, end[1] * 100 - loc.y) / 100.0
        check("Morrison is in the world", d < 50.0, "%.0f m from Bear Creek Ave" % d)
        return True
    pawn.script_look(R.s["yaw0"] + 180.0 * R.s["i"], -4.0)
    _shot(world, "morrison_%d" % R.s["i"], ui=False)
    R.s["i"] += 1
    R.s["at"] = now
    return False


def tick(delta):
    now = time.time()
    try:
        if now - R.t0 > TIMEOUT_S:
            log.append("ERROR: timed out (%s)" % R.s.get("name"))
            _finish()
            return
        if R.mark is None:
            unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).editor_request_begin_play()
            R.mark = now
            R.steps = [("boot", None)]
            return
        world = _world()
        pawn = unreal.GameplayStatics.get_player_pawn(world, 0) if world else None
        if pawn is None:
            if now - R.mark > 60.0:
                log.append("ERROR: no pawn after 60 s")
                _finish()
            return
        if not hasattr(pawn, "rr_go"):
            log.append("ERROR: pawn is %s, not RRExplorer (game mode not set?)" % pawn.get_class().get_name())
            _finish()
            return
        if R.steps and R.steps[0][1] is None:  # booting: let the level settle once
            if now - R.mark < 8.0:
                return
            R.steps = []
        if not R.steps:
            if not R.queue:
                _finish()
                return
            _start_scenario(pawn, world, R.queue.pop(0))
            return
        name, fn = R.steps[0]
        done = fn(now) if fn.__code__.co_argcount == 1 else fn(now, pawn, world)
        if done:
            R.steps.pop(0)
            R.mark = now
    except Exception as exc:
        import traceback
        log.append("ERROR: %r in %s" % (exc, R.s.get("name")))
        log.append(traceback.format_exc())
        _finish()


if __name__ == "__main__" or hasattr(unreal, "register_slate_post_tick_callback"):
    handle = unreal.register_slate_post_tick_callback(tick)
