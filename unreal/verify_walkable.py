"""Check that Lvl_RedRocks is walkable, without pressing Play.

Run headless after rebuild_level.py:
    UnrealEditor-Cmd RedRocks.uproject -run=pythonscript \
        -script=unreal/verify_walkable.py -unattended -nosplash -stdout

Writes /tmp/redrocks_verify.log and two PNGs to <project>/Saved/:
  redrocks_spawn_view.png   what the player sees on spawn (eye height, facing
                            up the bowl), through the real renderer
  redrocks_bowl_view.png    the stage and bowl from above the top row

Checks, each logged "ok" or "FAIL":
  - the level's game mode is the FirstPerson one
  - there is exactly one PlayerStart
  (Line traces need a physics scene, which a headless commandlet may not
  build; playtest.py is the real test of walking.)
  - a trace straight down from the PlayerStart hits the Stage within 3 m, so
    the player lands on the deck instead of falling
  - traces down onto the bowl, a road and the open terrain hit something
    (collision exists across the map)
  - each imported mesh kept its triangles (Unreal's OBJ import can decimate)
  - the Credits text is in the level
Leaves the level unsaved and removes its capture actor.
"""

import json
import math
import os
import sys

import unreal

_HERE = os.path.dirname(os.path.abspath(globals().get("__file__") or "/Users/austin/dev/redrocks-ue/unreal/x"))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import level_frame  # noqa: E402  (beside this script: terrain frame -> level frame)

LEVEL_PATH = "/Game/RedRocks/Lvl_RedRocks"
# The redrocks build output; REDROCKS_OUT_DIR points a test copy at another build.
OUT_DIR = os.environ.get("REDROCKS_OUT_DIR", "/Users/austin/dev/redrocks-ue/out")
LOG_PATH = os.path.join(os.environ.get("REDROCKS_LOG_DIR", "/tmp"), "redrocks_verify.log")
GAME_MODE_NAME = "BP_FirstPersonGameMode"
EYE_CM = 160.0
WIDTH, HEIGHT = 1600, 900

log = []


def check(label, ok, detail=""):
    log.append("%s  %s%s" % ("ok   " if ok else "FAIL ", label, (": " + detail) if detail else ""))
    return ok


def _actors():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()


def _trace_down(world, x, y, top=300000.0, bottom=-50000.0):
    hit = unreal.SystemLibrary.line_trace_single(
        world,
        unreal.Vector(x, y, top),
        unreal.Vector(x, y, bottom),
        unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,  # Visibility
        True,  # trace complex
        [],
        unreal.DrawDebugTrace.NONE,
        True,
    )
    if hit is None:
        return None
    t = hit.to_tuple()
    blocking = t[0]
    if not blocking:
        return None
    location = t[4]
    actor = next((v for v in t if isinstance(v, unreal.Actor)), None)
    return location, (actor.get_actor_label() if actor else "?")


def _render(world, location, rotation, filename):
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    cam = actors.spawn_actor_from_class(unreal.SceneCapture2D, location)
    cam.set_actor_rotation(rotation, False)
    comp = cam.get_component_by_class(unreal.SceneCaptureComponent2D)
    comp.set_editor_property("fov_angle", 90.0)
    comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
    rt = unreal.RenderingLibrary.create_render_target2d(
        world, WIDTH, HEIGHT, unreal.TextureRenderTargetFormat.RTF_RGBA8
    )
    comp.set_editor_property("texture_target", rt)
    comp.capture_scene()
    out_dir = os.path.join(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_saved_dir()))
    unreal.RenderingLibrary.export_render_target(world, rt, out_dir, filename)
    actors.destroy_actor(cam)
    path = os.path.join(out_dir, filename)
    return path if os.path.exists(path) else None


def main():
    with open(os.path.join(OUT_DIR, "manifest.json")) as fh:
        manifest = json.load(fh)

    unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).load_level(LEVEL_PATH)
    for cvar in ("r.EyeAdaptationQuality 0", "r.DefaultFeature.AutoExposure 0"):
        unreal.SystemLibrary.execute_console_command(None, cvar)
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()

    gm = world.get_world_settings().get_editor_property("default_game_mode")
    check("game mode", gm is not None and GAME_MODE_NAME in str(gm), str(gm))

    starts = [a for a in _actors() if isinstance(a, unreal.PlayerStart)]
    check("one PlayerStart", len(starts) == 1, "found %d" % len(starts))
    if not starts:
        return
    start = starts[0]
    loc = start.get_actor_location()
    yaw = start.get_actor_rotation().yaw
    log.append("      spawn x=%.0f y=%.0f z=%.0f yaw=%.1f" % (loc.x, loc.y, loc.z, yaw))

    hit = _trace_down(world, loc.x, loc.y, top=loc.z, bottom=loc.z - 2000.0)
    if hit:
        drop = loc.z - hit[0].z
        check("spawn lands on the stage", hit[1] == "Stage" and drop < 300.0,
              "hit %s %.0f cm below" % (hit[1], drop))
    else:
        check("spawn lands on the stage", False, "no hit within 20 m below the PlayerStart")

    # A few more probes: up the bowl along the stage axis, and out on the map.
    stage = manifest["stage"]
    ax, ay = level_frame.to_level(*stage["axis_east_north"])
    fx, fy = level_frame.to_level_cm(stage["front_edge_east_m"], stage["front_edge_north_m"])
    probes = {
        "bowl, 20 m up the rows": (fx + 2000 * ax, fy + 2000 * ay),
        "bowl, 60 m up the rows": (fx + 6000 * ax, fy + 6000 * ay),
        "terrain 600 m east": level_frame.to_level(60000.0, 0.0),
        "terrain 900 m south-west": level_frame.to_level(-60000.0, -60000.0),
    }
    for label, (x, y) in probes.items():
        h = _trace_down(world, x, y)
        check("collision at " + label, h is not None, ("hit %s z=%.0f" % (h[1], h[0].z)) if h else "nothing")

    expected = {"RedRocks_Terrain": manifest["terrain_mesh"]["triangles"]}
    if "benches" in manifest:
        expected["Benches"] = manifest["benches"]["triangles"]
    if "stage" in manifest:
        expected["Stage"] = manifest["stage"]["triangles"]
        for name, block in manifest["stage"].get("objs", {}).items():
            expected[name] = block["triangles"]  # Stage_Steel, Stage_Roof, Stage_Fascia, Stage_Towers
    for key, block in manifest.get("park", {}).items():
        expected[block["obj"][:-4]] = block["triangles"]
    if "denver" in manifest:
        expected["Denver"] = manifest["denver"]["triangles"]
    for name, want in expected.items():
        mesh = unreal.EditorAssetLibrary.load_asset("/Game/RedRocks/%s" % name)
        if mesh is None:
            check("mesh %s imported" % name, False, "missing")
            continue
        got = mesh.get_num_triangles(0)
        check("mesh %s kept its triangles" % name, got >= 0.98 * want, "%d of %d" % (got, want))

    credits = [a for a in _actors() if a.get_actor_label() == "Credits"]
    check("credits in level", len(credits) == 1)

    rad = math.radians(yaw)
    eye = unreal.Vector(loc.x, loc.y, loc.z + EYE_CM - 120.0)
    path = _render(world, eye, unreal.Rotator(pitch=8.0, yaw=yaw, roll=0.0), "redrocks_spawn_view.png")
    check("render spawn view", path is not None, path or "")
    top = unreal.Vector(fx + 9000 * ax, fy + 9000 * ay, loc.z + 4500.0)
    path = _render(world, top, unreal.Rotator(pitch=-22.0, yaw=yaw + 180.0, roll=0.0), "redrocks_bowl_view.png")
    check("render bowl view", path is not None, path or "")


if hasattr(unreal, "EditorAssetLibrary"):
    try:
        main()
    except Exception as exc:
        log.append("ERROR: %r" % (exc,))
    with open(LOG_PATH, "w") as fh:
        fh.write("\n".join(log) + "\n")
    for line in log:
        (unreal.log_error if line.startswith(("FAIL", "ERROR")) else unreal.log)(line)
