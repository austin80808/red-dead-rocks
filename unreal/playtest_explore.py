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

Results: /tmp/redrocks_playtest_explore.log ("ok"/"FAIL" lines with numbers),
screenshots redrocks_explore_*.png in <project>/Saved/Screenshots/.
Pass SCENARIOS=cart,sled (env) to run a subset.
"""

import json
import math
import os
import time

import unreal

OUT_DIR = os.environ.get("REDROCKS_OUT_DIR", "/Users/austin/dev/redrocks-ue/out")
LOG_PATH = os.path.join(os.environ.get("REDROCKS_LOG_DIR", "/tmp"), "redrocks_playtest_explore.log")
ALL = ["bowl", "climb", "wingsuit", "cart", "sled", "morrison"]
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
    elif name == "bowl":
        R.s["views"] = [("stage", 0.0, 4.0, False), ("bowl_top", 0.0, -14.0, False), ("bowl_top", 0.0, -14.0, True)]
        R.steps = [("views", bowl_views)]
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
    key = "carts" if R.s["name"] == "cart" else "sleds"
    datum = manifest["terrain_mesh"]["min_elevation_m"]
    pts = explore[key]["route" if key == "carts" else "run"]
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
    for at, name in ((6.0, "%s_start" % R.s["name"]), (40.0, "%s_mid" % R.s["name"])):
        if now - R.mark > at and name not in R.s.setdefault("shots", set()):
            R.s["shots"].add(name)
            _shot(world, name)
    end = R.s["route"][-1]
    left = math.hypot(end[0] - loc.x, end[1] - loc.y) / 100.0
    R.s["left"] = left
    limit = 420.0 if R.s["name"] == "cart" else 70.0
    return left < 60.0 or now - R.mark > limit


def ride_end(now, pawn, world):
    name = R.s["name"]
    _shot(world, "%s_end" % name)
    route_len = explore["carts"]["route_length_m"] if name == "cart" else explore["sleds"]["run_m"]
    detail = "%.0f m ridden of %.0f, %.0f m from the end, top %.0f km/h, %d wipeouts" % (
        R.s["dist"], route_len, R.s.get("left", -1), R.s["vmax"], R.s["crashes"])
    if name == "cart":
        check("shopping cart to Morrison", R.s.get("left", 1e9) < 150.0, detail)
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
    """The amphitheatre: from the stage up the rows, from the top down to the stage, then under snow."""
    if R.s.get("at") is None or now - R.s["at"] > 9.0:
        if R.s.get("pending"):
            _shot(world, R.s["pending"], ui=False)
            R.s["pending"] = None
        if not R.s["views"]:
            weather = unreal.GameplayStatics.get_actor_of_class(world, unreal.RRWeather)
            if weather:
                weather.set_snow_now(False)
            check("amphitheatre views", True)
            return True
        key, dyaw, pitch, snow = R.s["views"].pop(0)
        weather = unreal.GameplayStatics.get_actor_of_class(world, unreal.RRWeather)
        if weather:
            weather.set_snow_now(snow)
        pawn.rr_go(SPOT[key])
        R.s["look"] = (dyaw, pitch)
        R.s["pending"] = "bowl_%s%s" % (key, "_snow" if snow else "")
        R.s["at"] = now
        return False
    if now - R.s["at"] > 6.0 and R.s.get("look"):
        dyaw, pitch = R.s["look"]
        pawn.script_look(pawn.get_actor_rotation().yaw + dyaw, pitch)
        R.s["look"] = None
    return False


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
