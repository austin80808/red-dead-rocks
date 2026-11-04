"""Make Lvl_RedRocks an explorable realm: shopping carts at the top of Red Rocks
Park Road bound for Morrison, sleds on the sled hill, snow, teleport spots, and
the RedRocks C++ game mode (climbing, wingsuit, parachute, rides).

Needs the RedRocks C++ module built (Source/RedRocks in the Unreal project) and
out/manifest.json's explore block (`redrocks build`). Run after setup_credits.py
(rebuild_level.py runs it after setup_snow.py). Headless:
    UnrealEditor-Cmd RedRocks.uproject -run=pythonscript \
        -script=unreal/setup_explore.py -unattended -nosplash -stdout

What it does, logged to /tmp/redrocks_explore.log:
  1. Removes everything it placed before (actors labelled RR_*), so re-running
     never duplicates.
  2. Tags the Roads and Parking actors "road" / "parking" (carts roll fast on
     them and bog down off them).
  3. One ARRSpot per manifest spot, in order (number keys 1-9 in game). Spots
     drop onto whatever is below them when play starts, so the summit lands on
     the scanned rock top; the climb spot walks to the rock face.
  4. Three shopping carts at the cart lot, side by side, pointed down the road,
     with Morrison as their destination and the road route to it (the carts'
     cartoon road assist follows it); three sleds at the top of the sled hill.
  5. Three inner tubes at the top of Bear Creek with the creek as their route;
     an ARRWeather (snow off, day at start) and the HUD's elevation datum; the
     ARRConcert light rig on the stage (it plays when T makes it night).
  6. Sets the level's game mode to /Script/RedRocks.RRGameMode and saves.
"""

import json
import math
import os

import unreal

OUT_DIR = os.environ.get("REDROCKS_OUT_DIR", "/Users/austin/dev/redrocks-ue/out")
LEVEL_PATH = "/Game/RedRocks/Lvl_RedRocks"
GAME_MODE = "/Script/RedRocks.RRGameMode"
LOG_PATH = os.path.join(os.environ.get("REDROCKS_LOG_DIR", "/tmp"), "redrocks_explore.log")
PREFIX = "RR_"
LIFT_CM = 150.0  # rides and spots start this far over the map's ground and settle at play
CART_SPACING_CM = 260.0
SLED_SPACING_CM = 220.0

log = []


def _step(label, fn, *args):
    try:
        result = fn(*args)
        log.append("ok    %s%s" % (label, "" if result is None else ": %s" % (result,)))
        return result
    except Exception as exc:
        log.append("FAIL  %s: %r" % (label, exc))
        return None


def _actors():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()


def _spawn(cls, loc, yaw, label):
    sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor = sub.spawn_actor_from_class(cls, unreal.Vector(*loc))
    if actor is None:
        raise RuntimeError("could not spawn %s" % label)
    actor.set_actor_rotation(unreal.Rotator(pitch=0.0, yaw=yaw, roll=0.0), False)
    actor.set_actor_label(label)
    return actor


def to_cm(east_m, north_m, elevation_m, datum_m, lift_cm=LIFT_CM):
    """Unreal location (cm) of a manifest point. Plain arithmetic, tested offline."""
    return (east_m * 100.0, north_m * 100.0, (elevation_m - datum_m) * 100.0 + lift_cm)


def row_offsets(count, spacing_cm, yaw_deg):
    """Side-by-side offsets (cm) for ``count`` rides facing ``yaw_deg``, centred on the first."""
    rad = math.radians(yaw_deg)
    rx, ry = -math.sin(rad), math.cos(rad)  # right of forward in X east / Y north
    return [((i - (count - 1) / 2.0) * spacing_cm * rx, (i - (count - 1) / 2.0) * spacing_cm * ry)
            for i in range(count)]


def _clear():
    sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    gone = 0
    for actor in _actors():
        if actor.get_actor_label().startswith(PREFIX):
            sub.destroy_actor(actor)
            gone += 1
    return gone


def _tag_surfaces():
    tagged = []
    for actor in _actors():
        label = actor.get_actor_label()
        tag = {"Roads": "road", "Parking": "parking", "Water": "water"}.get(label)
        if tag:
            tags = [t for t in actor.tags if str(t) != tag] + [unreal.Name(tag)]
            actor.tags = tags
            tagged.append(label)
    if not tagged:
        raise RuntimeError("no Roads or Parking actor; run setup_gameplay.py first")
    return ", ".join(tagged)


def _spots(explore, datum, creation):
    placed = []
    for i, spot in enumerate(explore["spots"]):
        loc = to_cm(spot["east_m"], spot["north_m"], spot["elevation_m"], datum)
        actor = _spawn(unreal.RRSpot, loc, spot["yaw_deg"], "%sSpot_%d_%s" % (PREFIX, i + 1, spot["key"]))
        actor.set_editor_property("order", i + 1)
        actor.set_editor_property("title", spot["title"])
        actor.set_editor_property("tip", spot["tip"])
        # Summits trace down from 150 m up so they land on the rock top, not under it.
        actor.set_editor_property("snap_from_above", 15000.0 if spot.get("place") == "summit" else 3000.0)
        if spot.get("place") == "summit":
            # The scan's rock top is not exactly over the OSM centroid: find the
            # highest ground within 60 m and face the biggest drop.
            actor.set_editor_property("seek_summit_radius", 6000.0)
        if spot["key"] == "climb" and creation:
            actor.set_editor_property("face_target", unreal.Vector(
                creation["centroid_east"] * 100.0, creation["centroid_north"] * 100.0, loc[2]))
        placed.append(spot["title"])
    return "%d: %s" % (len(placed), ", ".join(placed))


def _rides(block, kind, spacing, datum, destination=None, label="Cart", route=None):
    if not block:
        return "none in manifest"
    s = block["start"]
    base = to_cm(s["east_m"], s["north_m"], s["elevation_m"], datum)
    for i, (dx, dy) in enumerate(row_offsets(block["count"], spacing, s["yaw_deg"])):
        actor = _spawn(unreal.RRRide, (base[0] + dx, base[1] + dy, base[2]), s["yaw_deg"],
                       "%s%s_%d" % (PREFIX, label, i + 1))
        actor.set_editor_property("kind", kind)
        if destination:
            actor.set_editor_property("destination", unreal.Vector(*destination[0]))
            actor.set_editor_property("destination_name", destination[1])
        if route:
            actor.set_editor_property("route", [unreal.Vector(*p) for p in route])
    return "%d at E%.0f N%.0f" % (block["count"], s["east_m"], s["north_m"])


def _weather(datum):
    actor = _spawn(unreal.RRWeather, (0.0, 0.0, 0.0), 0.0, PREFIX + "Weather")
    actor.set_editor_property("datum_elevation_m", datum)
    actor.set_editor_property("start_snowing", False)
    mpc = unreal.EditorAssetLibrary.load_asset("/Game/RedRocks/Materials/MPC_Weather")
    if mpc is None:
        raise RuntimeError("no MPC_Weather; run setup_snow.py first")
    actor.set_editor_property("collection", mpc)
    return "datum %.2f m" % datum


def _concert(manifest, datum):
    """The light rig on the stage deck, facing up the bowl like the player at spawn."""
    spawn = manifest.get("spawn")
    if not spawn:
        return "no spawn in manifest"
    loc = to_cm(spawn["east_m"], spawn["north_m"], spawn["ground_elevation_m"], datum, 0.0)
    _spawn(unreal.RRConcert, loc, spawn["yaw_deg"], PREFIX + "Concert")
    return "at E%.0f N%.0f facing %.0f" % (spawn["east_m"], spawn["north_m"], spawn["yaw_deg"])


def _game_mode():
    cls = unreal.load_class(None, GAME_MODE)
    if cls is None:
        raise RuntimeError("no %s; build the RedRocks C++ module" % GAME_MODE)
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    world.get_world_settings().set_editor_property("default_game_mode", cls)
    return GAME_MODE


def _save():
    if not unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level():
        raise RuntimeError("could not save %s (read-only .umap? git lfs lock it)" % LEVEL_PATH)


def main():
    if not hasattr(unreal, "RRRide"):
        log.append("ABORT: the RedRocks C++ module is not loaded; build Source/RedRocks first")
        return
    with open(os.path.join(OUT_DIR, "manifest.json")) as fh:
        manifest = json.load(fh)
    explore = manifest.get("explore")
    if not explore:
        log.append("ABORT: manifest.json has no explore block; run `redrocks build`")
        return
    datum = manifest["terrain_mesh"]["min_elevation_m"]
    creation = next((m for m in manifest.get("monoliths", []) if m["name"].startswith("Creation")), None)

    unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).load_level(LEVEL_PATH)
    _step("clear old RR_ actors", _clear)
    _step("tag road and parking surfaces", _tag_surfaces)
    _step("spots", _spots, explore, datum, creation)
    town = None
    road = None
    if explore.get("carts"):
        end = explore["carts"]["route"][-1]
        town = (to_cm(end[0], end[1], end[2], datum, 0.0), explore["carts"]["destination"])
        road = [to_cm(p[0], p[1], p[2], datum, 0.0) for p in explore["carts"]["route"]]
    _step("shopping carts", _rides, explore.get("carts"), unreal.RRRideKind.CART, CART_SPACING_CM, datum, town,
          "Cart", road)
    _step("sleds", _rides, explore.get("sleds"), unreal.RRRideKind.SLED, SLED_SPACING_CM, datum, None, "Sled")
    creek = None
    if explore.get("tubes"):
        creek = [to_cm(p[0], p[1], p[2], datum, 0.0) for p in explore["tubes"]["route"]]
    _step("inner tubes", _rides, explore.get("tubes"), unreal.RRRideKind.TUBE, SLED_SPACING_CM, datum,
          (creek[-1], "the bottom of Bear Creek") if creek else None, "Tube", creek)
    _step("weather", _weather, datum)
    _step("concert rig", _concert, manifest, datum)
    _step("game mode", _game_mode)
    _step("save %s" % LEVEL_PATH, _save)


if hasattr(unreal, "EditorAssetLibrary"):
    try:
        main()
    except Exception as exc:
        log.append("ERROR: %r" % (exc,))
    with open(LOG_PATH, "w") as fh:
        fh.write("\n".join(log) + "\n")
    for line in log:
        (unreal.log_error if line.startswith(("FAIL", "ERROR", "ABORT")) else unreal.log)(line)
