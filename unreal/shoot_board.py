"""Shoot the photo board in the editor world: every view in
docs/media/photo-board/views.json, from its fitted camera, under its photo's sun.

Needs the editor GUI (-ExecCmds, not -ExecutePythonScript or -run=pythonscript):
the shots come from the level viewport.

    REDROCKS_OUT_DIR=<checkout>/out REDROCKS_LOG_DIR=/tmp REDROCKS_BOARD_PREFIX=before_ \
    "/Users/Shared/Epic Games/UE_5.8/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor" \
        "<project>/RedRocks.uproject" -nosplash \
        -ini:EditorPerProjectUserSettings:[/Script/UnrealEd.EditorPerformanceSettings]:bThrottleCPUWhenNotForeground=False \
        -ExecCmds="py <checkout>/unreal/shoot_board.py"

The -ini override stops a background editor being throttled. The script may
also be a later command of the same editor process, after a long synchronous
script: -ExecCmds="py <checkout>/unreal/rebuild_level.py, py <checkout>/unreal/shoot_board.py".
It (re)loads the level itself and waits for it, so it never assumes it starts
right after the editor loaded. It quits the editor when done, and on any error
after writing its log.

It works in the editor world, not Play In Editor, so no HUD, credits or debug
text draws. Once the level is loaded it hides every TextRenderActor for the run
(the "Credits" and "Scan_ml0w_Credit" labels float in view), turns on game view
(no editor icons or billboards), DisableAllScreenMessages, and spawns one
transient CameraActor with constrain_aspect_ratio on. Then, per shot: the camera
takes the view's level-frame location, rotation and horizontal fov, at the
shot's own aspect; the "Sun" actor turns to the view's sun
(setup_lighting.sun_rotation); shaders and streaming are finished; at least
8 s of editor ticks pass (Lumen and streaming settle); then
unreal.AutomationLibrary.take_high_res_screenshot(w, h, path, camera), and the
script waits for a complete PNG on disk before going on. At the end it restores
the sun, un-hides the text, destroys what it spawned and quits. Nothing is
saved: QUIT_EDITOR exits without a save prompt, and every spawned actor is
transient.

Shots, 1600 px wide at the view's aspect, in $REDROCKS_OUT_DIR/board/:
  <prefix><view>.png         under the view's sun (REDROCKS_BOARD_SUN=view, the default)
                             or the level's own sun (=level: one light for a whole pass)
  <prefix>level_<view>.png   =both: the view's sun shot, then this one under the level's sun
  calib_<view>.png           REDROCKS_BOARD_CALIB=1: a 60 cm unlit magenta (#FF00FF) sphere
                             at each of the view's landmarks, tonemapper and bloom off so the
                             markers stay pure magenta for tools/make_board_compare.py --calib.
                             Marker positions (level cm) come from
                             $REDROCKS_OUT_DIR/board/landmarks_cm.json, written beforehand by
                             tools/make_board_compare.py --write-landmarks (the editor's Python
                             has no numpy to build the landmark catalog itself).

Env: REDROCKS_OUT_DIR (default <checkout>/out), REDROCKS_LEVEL (default
/Game/RedRocks/Lvl_RedRocks), REDROCKS_BOARD_PREFIX (default before_),
REDROCKS_BOARD_ONLY (comma list of views), REDROCKS_BOARD_SUN (view | level |
both), REDROCKS_BOARD_CALIB, REDROCKS_BOARD_MARKER_CM (marker diameter, default
60), REDROCKS_BOARD_LANDMARKS (marker positions file), REDROCKS_VIEWS or
`py shoot_board.py --views PATH` (the board file). Log:
$REDROCKS_LOG_DIR (default /tmp)/redrocks_board.log: details, then one ok/FAIL
line per view.
"""

import importlib.util
import json
import math
import os
import struct
import sys
import time
import traceback
import types

import unreal

_HERE = os.path.dirname(os.path.abspath(globals().get("__file__") or "/Users/austin/dev/redrocks-ue/unreal/x"))
CHECKOUT = os.path.dirname(_HERE)
DEFAULT_VIEWS = os.path.join(CHECKOUT, "docs", "media", "photo-board", "views.json")
DEFAULT_OUT = os.path.join(CHECKOUT, "out")
DEFAULT_LEVEL = "/Game/RedRocks/Lvl_RedRocks"
SUN_LABEL = "Sun"  # setup_lighting.py's directional light
SUN_MODES = ("view", "level", "both")

SHOT_WIDTH = 1600
SETTLE_S = 8.0          # Lumen and streaming, after every camera or sun change
SETTLE_FRAMES = 30      # and at least this many editor ticks, however slow they are
CALIB_SETTLE_S = 3.0    # markers only: same camera and light as the shot before
RUNUP_FRAMES = 16       # r.HighResScreenshotDelay: frames rendered at the shot size first
SHOT_TIMEOUT_S = 120.0  # for the PNG to land
LOAD_TIMEOUT_S = 900.0  # level load and the first shader compile
PER_SHOT_S = 90.0       # run deadline once the plan is known
LANDED_STABLE_S = 0.5   # a complete PNG whose size has not changed for this long

MARKER_CM = 60.0                        # diameter
MARKER_RGB = (1.0, 0.0, 1.0)            # #FF00FF
MARKER_FALLBACK_GAIN = 1.0e5            # emissive without EyeAdaptationInverse: daylight EV100 ~11-16
MIN_MARKER_PX = 3.0                     # a smaller marker may not survive anti-aliasing
SPHERE_MESH = "/Engine/BasicShapes/Sphere.Sphere"  # 100 cm across
CAMERA_LABEL = "BoardCamera"
MARKER_LABEL = "BoardMarker"

log = []
state = {"phase": "start"}
handle = None
_LIGHTING = None


# ---- Plain Python, tested offline (tests/test_shoot_board.py) ----------------

def read_settings(env, argv=()):
    """The run's settings from the environment, and ``--views PATH`` from argv."""
    out_dir = env.get("REDROCKS_OUT_DIR") or DEFAULT_OUT
    board_dir = os.path.join(out_dir, "board")
    sun = (env.get("REDROCKS_BOARD_SUN") or "view").strip().lower()
    if sun not in SUN_MODES:
        raise ValueError("REDROCKS_BOARD_SUN=%s: expected view, level or both" % sun)
    views = env.get("REDROCKS_VIEWS") or DEFAULT_VIEWS
    argv = list(argv)
    for i, arg in enumerate(argv):
        if arg == "--views" and i + 1 < len(argv):
            views = argv[i + 1]
        elif arg.startswith("--views="):
            views = arg.split("=", 1)[1]
    return {
        "level": env.get("REDROCKS_LEVEL") or DEFAULT_LEVEL,
        "out_dir": out_dir,
        "board_dir": board_dir,
        "prefix": env.get("REDROCKS_BOARD_PREFIX", "before_"),
        "only": [n.strip() for n in env.get("REDROCKS_BOARD_ONLY", "").split(",") if n.strip()],
        "sun": sun,
        "calib": env.get("REDROCKS_BOARD_CALIB", "").strip().lower() not in ("", "0", "false", "no"),
        "marker_cm": float(env.get("REDROCKS_BOARD_MARKER_CM") or MARKER_CM),
        "landmarks": env.get("REDROCKS_BOARD_LANDMARKS") or os.path.join(board_dir, "landmarks_cm.json"),
        "views": views,
        "log": os.path.join(env.get("REDROCKS_LOG_DIR") or "/tmp", "redrocks_board.log"),
        "width": SHOT_WIDTH,
    }


def load_views(path):
    """The board's views: views.json's ``views`` list (the schema of redrocks.photo_board)."""
    with open(path) as fh:
        return json.load(fh)["views"]


def load_landmarks(path):
    """{name: (x, y, z)} level cm from landmarks_cm.json, or None when there is no file."""
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        rows = json.load(fh)["landmarks"]
    return {name: tuple(float(c) for c in xyz) for name, xyz in rows.items()}


def shot_size(aspect, width=SHOT_WIDTH):
    """(width, height) of a shot ``width`` px wide at ``aspect`` = [w, h]."""
    try:
        aw, ah = (float(a) for a in aspect)
    except (TypeError, ValueError):
        raise ValueError("aspect must be [w, h], got %r" % (aspect,))
    if not (aw > 0 and ah > 0):
        raise ValueError("aspect must be positive, got %r" % (aspect,))
    return int(width), max(1, int(round(width * ah / aw)))


def camera_settings(view):
    """The view's camera for the CameraActor: ``location`` (level cm), ``rotation``
    (pitch/yaw/roll, already Unreal level-frame degrees: used as they are), ``fov``
    (horizontal) and ``aspect_ratio``: the shot's own w / h, so the constrained view
    fills the shot exactly, with no letterbox bars."""
    name = view.get("name", "?")
    cam = view.get("camera")
    if not cam:
        raise ValueError("%s: no fitted camera (run tools/fit_photo_camera.py)" % name)
    w, h = shot_size(view.get("aspect"))
    return {
        "location": (float(cam["x"]), float(cam["y"]), float(cam["z"])),
        "rotation": {"pitch": float(cam["pitch"]), "yaw": float(cam["yaw"]), "roll": float(cam.get("roll", 0.0))},
        "fov": float(cam.get("fov", 70.0)),
        "aspect_ratio": w / h,
    }


def _lighting():
    """unreal/setup_lighting.py, loaded for its sun maths only.

    Importing it inside the editor runs its main(): it rebuilds the light rig and
    saves the level. Loaded under a bare stand-in `unreal` module, its guard
    (hasattr(unreal, "EditorAssetLibrary")) is false, so only its constants and
    pure functions load; the real module is put back straight after.
    """
    global _LIGHTING
    if _LIGHTING is None:
        real = sys.modules.get("unreal")
        sys.modules["unreal"] = types.ModuleType("unreal")
        try:
            spec = importlib.util.spec_from_file_location(
                "redrocks_board_lighting", os.path.join(_HERE, "setup_lighting.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        finally:
            if real is None:
                sys.modules.pop("unreal", None)
            else:
                sys.modules["unreal"] = real
        _LIGHTING = mod
    return _LIGHTING


def sun_rotation(sun):
    """The Sun actor's rotation for a view's ``sun`` (azimuth_deg, elevation_deg)."""
    pitch, yaw = _lighting().sun_rotation(float(sun["azimuth_deg"]), float(sun["elevation_deg"]))
    return {"pitch": pitch, "yaw": yaw, "roll": 0.0}


def marker_size_px(cam, point_cm, width, diameter_cm):
    """About how many pixels across a sphere at ``point_cm`` draws from ``cam``
    (camera_settings); 0 when it is behind the camera."""
    yaw, pitch = math.radians(cam["rotation"]["yaw"]), math.radians(cam["rotation"]["pitch"])
    forward = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))
    depth = sum((p - c) * f for p, c, f in zip(point_cm, cam["location"], forward))
    if depth <= 0.0:
        return 0.0
    focal = (width / 2.0) / math.tan(math.radians(cam["fov"]) / 2.0)
    return focal * diameter_cm / depth


def shot_plan(views, settings, landmarks=None):
    """(shots, problems): what to shoot, in order, and why any view cannot be.

    Each shot is a dict: ``view``, ``kind`` ("view", "level" or "calib"), ``file``,
    ``size`` (w, h), ``camera`` (camera_settings), ``sun`` (a rotation, or None to
    keep the level's own), ``settle_s`` and, for calib, ``markers``
    [(landmark name, (x, y, z) level cm)]. Per view: its main shot, then its
    calibration shot (same camera and light), then with sun=both its level-sun
    shot. ``problems`` is [(view name, reason)].
    """
    names = [v.get("name") for v in views]
    only = settings["only"]
    problems = [(n, "not on the board") for n in only if n not in names]
    prefix = settings["prefix"]
    shots = []
    for view in views:
        name = view.get("name")
        if only and name not in only:
            continue
        try:
            cam = camera_settings(view)
            size = shot_size(view.get("aspect"), settings["width"])
            if settings["sun"] == "level":
                first = ("level", prefix + name + ".png", None)
            else:
                if not view.get("sun"):
                    raise ValueError("%s: no sun in the board file" % name)
                first = ("view", prefix + name + ".png", sun_rotation(view["sun"]))
        except (ValueError, KeyError, TypeError) as exc:
            problems.append((name, str(exc)))
            continue
        base = {"view": name, "size": size, "camera": cam}
        shots.append(dict(base, kind=first[0], file=first[1], sun=first[2], settle_s=SETTLE_S))
        if settings["calib"]:
            wanted = [lm["name"] for lm in view.get("landmarks", [])]
            if landmarks is None:
                problems.append((name, "calib: no marker positions at %s (run tools/make_board_compare.py "
                                       "--write-landmarks first)" % settings["landmarks"]))
            else:
                missing = [n for n in wanted if n not in landmarks]
                if missing:
                    problems.append((name, "calib: %s has no position for %s"
                                     % (os.path.basename(settings["landmarks"]), ", ".join(missing))))
                markers = [(n, tuple(landmarks[n])) for n in wanted if n in landmarks]
                if markers:
                    shots.append(dict(base, kind="calib", file="calib_%s.png" % name, sun=first[2],
                                      settle_s=CALIB_SETTLE_S, markers=markers))
        if settings["sun"] == "both":
            shots.append(dict(base, kind="level", file=prefix + "level_" + name + ".png", sun=None,
                              settle_s=SETTLE_S))
    return shots, problems


def png_size(path):
    """(width, height) from a PNG's IHDR, or None."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(24)
    except OSError:
        return None
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", head[16:24])


def png_complete(path):
    """True once ``path`` is a whole PNG: the signature at the start, IEND at the end."""
    try:
        if os.path.getsize(path) < 8 + 12:
            return False
        with open(path, "rb") as fh:
            head = fh.read(8)
            fh.seek(-12, os.SEEK_END)
            tail = fh.read(12)
    except OSError:
        return False
    return head == b"\x89PNG\r\n\x1a\n" and tail[4:8] == b"IEND"


def summary_lines(order, results):
    """One ``ok``/``FAIL`` line per view, in board order."""
    lines = []
    for name in order:
        r = results.get(name, {})
        errors, files = r.get("errors", []), r.get("files", [])
        if errors:
            lines.append("FAIL  %-16s %s" % (name, "; ".join(errors)))
        elif files:
            lines.append("ok    %-16s %s" % (name, ", ".join(files)))
        else:
            lines.append("FAIL  %-16s not shot (the run ended first)" % name)
    return lines


# ---- In the editor -------------------------------------------------------------

def _sub(name):
    return unreal.get_editor_subsystem(getattr(unreal, name))


def _try(label, fn):
    """An editor call that may be missing or refuse in this engine version: log, go on."""
    try:
        return fn()
    except Exception as exc:
        log.append("warn  %s: %r" % (label, exc))
        return None


def _console(cmd):
    world = state.get("world")
    _try("console %s" % cmd, lambda: unreal.SystemLibrary.execute_console_command(world, cmd))


def _game_world():
    try:
        return _sub("UnrealEditorSubsystem").get_game_world()
    except Exception:
        return None


def _editor_world():
    try:
        return _sub("UnrealEditorSubsystem").get_editor_world()
    except Exception:
        return None


def _hide_text(actors):
    """Temporarily hide (in the editor, not saved) every actor that draws text."""
    hidden = []
    for actor in actors:
        try:
            text = isinstance(actor, unreal.TextRenderActor) or (
                actor.get_component_by_class(unreal.TextRenderComponent) is not None)
            if text and not actor.is_temporarily_hidden_in_editor():
                actor.set_is_temporarily_hidden_in_editor(True)
                hidden.append(actor)
        except Exception as exc:
            log.append("warn  hide text %s: %r" % (_label(actor), exc))
    log.append("ok    hid %d text actor(s): %s" % (len(hidden), ", ".join(_label(a) for a in hidden) or "none"))
    return hidden


def _label(actor):
    try:
        return actor.get_actor_label()
    except Exception:
        return "?"


def _spawn(cls, location, label):
    actor = _sub("EditorActorSubsystem").spawn_actor_from_class(
        cls, unreal.Vector(*location), unreal.Rotator(pitch=0.0, yaw=0.0, roll=0.0), True)  # transient
    if actor is None:
        raise RuntimeError("could not spawn %s" % label)
    _try("label %s" % label, lambda: actor.set_actor_label(label, False))
    return actor


def _set_camera(actor, cam):
    x, y, z = cam["location"]
    r = cam["rotation"]
    # views.json holds level-frame Unreal angles: straight into the Rotator, by keyword.
    actor.set_actor_location_and_rotation(
        unreal.Vector(x, y, z), unreal.Rotator(pitch=r["pitch"], yaw=r["yaw"], roll=r["roll"]), False, True)
    comp = actor.get_component_by_class(unreal.CameraComponent)
    comp.set_editor_property("field_of_view", cam["fov"])
    comp.set_editor_property("constrain_aspect_ratio", True)
    comp.set_editor_property("aspect_ratio", cam["aspect_ratio"])


def _set_sun(rotation):
    sun = state.get("sun")
    if sun is None:
        return
    if rotation is None:
        sun.set_actor_rotation(state["sun_rotation"], False)
    else:
        sun.set_actor_rotation(unreal.Rotator(pitch=rotation["pitch"], yaw=rotation["yaw"],
                                              roll=rotation["roll"]), False)


def _marker_material():
    """Unlit magenta (#FF00FF) in the transient package (never saved). Its emissive
    is divided by the current exposure (EyeAdaptationInverse) so the markers read
    #FF00FF whatever the auto exposure; without that node, a fixed daylight gain."""
    lib = unreal.MaterialEditingLibrary
    mat = unreal.new_object(unreal.Material, unreal.get_transient_package())
    mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    r, g, b = MARKER_RGB
    color = lib.create_material_expression(mat, unreal.MaterialExpressionConstant3Vector, -600, 0)
    color.set_editor_property("constant", unreal.LinearColor(r=r, g=g, b=b, a=1.0))
    out, how = None, ""
    inverse = getattr(unreal, "MaterialExpressionEyeAdaptationInverse", None)
    if inverse is not None:
        eai = lib.create_material_expression(mat, inverse, -600, 200)
        mul = lib.create_material_expression(mat, unreal.MaterialExpressionMultiply, -300, 0)
        if lib.connect_material_expressions(color, "", mul, "A") and lib.connect_material_expressions(eai, "", mul, "B"):
            out, how = mul, "exposure-compensated"
    if out is None:
        k = MARKER_FALLBACK_GAIN
        color.set_editor_property("constant", unreal.LinearColor(r=r * k, g=g * k, b=b * k, a=1.0))
        out, how = color, "fixed gain %.0e" % k
    if not lib.connect_material_property(out, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR):
        raise RuntimeError("could not connect the marker's emissive")
    lib.recompile_material(mat)
    log.append("ok    marker material: unlit #FF00FF, %s" % how)
    return mat


def _spawn_markers(shot):
    s = state["s"]
    mesh = unreal.load_object(None, SPHERE_MESH)
    scale = s["marker_cm"] / 100.0
    for name, xyz in shot["markers"]:
        actor = _spawn(unreal.StaticMeshActor, xyz, "%s_%s" % (MARKER_LABEL, name))
        state["markers"].append(actor)
        comp = actor.get_component_by_class(unreal.StaticMeshComponent)
        _try("marker mobility", lambda: comp.set_mobility(unreal.ComponentMobility.MOVABLE))
        actor.set_actor_location(unreal.Vector(*xyz), False, True)
        actor.set_actor_scale3d(unreal.Vector(scale, scale, scale))
        comp.set_static_mesh(mesh)
        if state.get("marker_material") is not None:
            comp.set_material(0, state["marker_material"])
        # A marker must not light, shadow or show up in Lumen's scene: only its own pixels count.
        for prop in ("cast_shadow", "affect_dynamic_indirect_lighting", "affect_distance_field_lighting"):
            _try("marker %s" % prop, lambda prop=prop: comp.set_editor_property(prop, False))
        px = marker_size_px(shot["camera"], xyz, shot["size"][0], s["marker_cm"])
        small = "  (under %.0f px: raise REDROCKS_BOARD_MARKER_CM)" % MIN_MARKER_PX if px < MIN_MARKER_PX else ""
        log.append("      marker %-22s (%.0f, %.0f, %.0f) ~%.1f px%s" % (name, xyz[0], xyz[1], xyz[2], px, small))


def _clear_markers():
    for actor in state.get("markers", []):
        _try("destroy marker", lambda actor=actor: _sub("EditorActorSubsystem").destroy_actor(actor))
    state["markers"] = []


def _result(view):
    return state["results"].setdefault(view, {"files": [], "errors": []})


def _prepare(world):
    s = state["s"]
    log.append("ok    level %s" % _try("world name", world.get_path_name))
    views = load_views(s["views"])
    landmarks = load_landmarks(s["landmarks"]) if s["calib"] else None
    shots, problems = shot_plan(views, s, landmarks)
    order = [v.get("name") for v in views if not s["only"] or v.get("name") in s["only"]]
    order += [n for n in s["only"] if n not in order]
    state.update(queue=shots, order=order, markers=[], results={})
    for name, why in problems:
        _result(name)["errors"].append(why)
    log.append("      plan: %d shot(s) of %d view(s) from %s, sun %s, calib %s, prefix %r"
               % (len(shots), len(order), s["views"], s["sun"], "on" if s["calib"] else "off", s["prefix"]))
    if not shots:
        return False
    actors = _sub("EditorActorSubsystem").get_all_level_actors()
    sun = next((a for a in actors if _label(a) == SUN_LABEL), None)
    if sun is None and any(shot["sun"] is not None for shot in shots):
        raise RuntimeError("no actor labelled %s in %s: run setup_lighting.py" % (SUN_LABEL, s["level"]))
    if sun is not None:
        state["sun"], state["sun_rotation"] = sun, sun.get_actor_rotation()
        log.append("      level sun: %s" % (state["sun_rotation"],))
    state["hidden"] = _hide_text(actors)
    les = _sub("LevelEditorSubsystem")
    state["game_view_was"] = _try("read game view", les.editor_get_game_view)
    _try("game view on", lambda: les.editor_set_game_view(True))
    _try("realtime viewport", lambda: les.editor_set_viewport_realtime(True))
    state["exact_was"] = _try("read exact camera view", les.get_exact_camera_view)
    _try("exact camera view", lambda: les.set_exact_camera_view(True))
    _console("DisableAllScreenMessages")
    state["runup_was"] = _try("read r.HighResScreenshotDelay", lambda: unreal.SystemLibrary.get_console_variable_int_value(
        "r.HighResScreenshotDelay"))
    _console("r.HighResScreenshotDelay %d" % RUNUP_FRAMES)
    state["camera"] = _spawn(unreal.CameraActor, shots[0]["camera"]["location"], CAMERA_LABEL)
    if any(shot["kind"] == "calib" for shot in shots):
        state["marker_material"] = _try("marker material", _marker_material)
    os.makedirs(s["board_dir"], exist_ok=True)
    state["deadline"] = time.time() + 300.0 + PER_SHOT_S * len(shots)
    _try("finish loading", unreal.AutomationLibrary.finish_loading_before_screenshot)
    return True


def _next_shot():
    if not state["queue"]:
        _finish()
        return
    shot = state["queue"].pop(0)
    camera = state["camera"]
    _set_camera(camera, shot["camera"])
    _try("pilot camera", lambda: _sub("LevelEditorSubsystem").pilot_level_actor(camera))
    _set_sun(shot["sun"])
    if shot["kind"] == "calib":
        _spawn_markers(shot)
        flags = "0"
    else:
        _clear_markers()
        flags = "2"  # 2: no override
    for flag in ("Tonemapper", "Bloom"):
        _console("ShowFlag.%s %s" % (flag, flags))
    path = os.path.join(state["s"]["board_dir"], shot["file"])
    if os.path.exists(path):
        os.remove(path)  # a stale shot must never pass for this one
    _try("finish loading", unreal.AutomationLibrary.finish_loading_before_screenshot)
    state.update(phase="settle", shot=shot, path=path, mark=time.time(), frames=0)


def _request_shot(now):
    shot = state["shot"]
    w, h = shot["size"]
    state["settled"] = (now - state["mark"], state["frames"])
    state["task"] = unreal.AutomationLibrary.take_high_res_screenshot(w, h, state["path"], state["camera"])
    state.update(phase="shooting", mark=now, last_size=-1, size_t=now)


def _sun_text(shot):
    if shot["sun"] is None:
        return "level sun"
    return "view sun (pitch %.1f yaw %.1f)" % (shot["sun"]["pitch"], shot["sun"]["yaw"])


def _check_landed(now):
    shot, path = state["shot"], state["path"]
    if png_complete(path):
        size = os.path.getsize(path)
        if size != state["last_size"]:
            state.update(last_size=size, size_t=now)
            return
        if now - state["size_t"] < LANDED_STABLE_S:
            return
        got = png_size(path)
        odd = "" if got == tuple(shot["size"]) else " (asked for %dx%d)" % tuple(shot["size"])
        settle_s, frames = state["settled"]
        log.append("      shot  %-30s %dx%d%s, %s, settled %.1f s / %d ticks, landed in %.1f s"
                   % (shot["file"], got[0], got[1], odd, _sun_text(shot), settle_s, frames, now - state["mark"]))
        _result(shot["view"])["files"].append(shot["file"])
        state["phase"] = "next"
        return
    if now - state["mark"] > SHOT_TIMEOUT_S:
        done = _try("task state", state["task"].is_task_done) if state.get("task") is not None else None
        why = "no complete %s after %.0f s (screenshot task done: %s)" % (shot["file"], SHOT_TIMEOUT_S, done)
        log.append("      shot  %-30s FAILED: %s" % (shot["file"], why))
        _result(shot["view"])["errors"].append(why)
        state["phase"] = "next"


def _advance(now):
    phase = state["phase"]
    if phase == "start":
        state.update(t0=now, deadline=now + LOAD_TIMEOUT_S, results={}, order=[], queue=[], markers=[])
        state["s"] = read_settings(os.environ, sys.argv)
        if _game_world() is not None:
            _try("end Play In Editor", lambda: _sub("LevelEditorSubsystem").editor_request_end_play())
            state["phase"] = "end_pie"
        else:
            state["phase"] = "load"
        return
    if now > state["deadline"]:
        log.append("ERROR: timed out in phase %s" % phase)
        _finish()
        return
    if phase == "end_pie":
        if _game_world() is None:
            state["phase"] = "load"
        return
    if phase == "load":
        level = state["s"]["level"]
        if not _sub("LevelEditorSubsystem").load_level(level):
            log.append("ERROR: could not load %s" % level)
            _finish()
            return
        state.update(phase="wait_world", mark=time.time(), frames=0)
        return
    if phase == "wait_world":
        state["frames"] += 1
        world = _editor_world()
        if world is None or now - state["mark"] < 1.0 or state["frames"] < 10:
            return
        state["world"] = world
        if _prepare(world):
            state["phase"] = "next"
        else:
            _finish()
        return
    if phase == "next":
        _next_shot()
        return
    if phase == "settle":
        state["frames"] += 1
        if now - state["mark"] >= state["shot"]["settle_s"] and state["frames"] >= SETTLE_FRAMES:
            _request_shot(now)
        return
    if phase == "shooting":
        _check_landed(now)


def _restore():
    les = None
    try:
        les = _sub("LevelEditorSubsystem")
    except Exception as exc:
        log.append("warn  restore: no LevelEditorSubsystem: %r" % (exc,))
    _clear_markers()
    if les is not None:
        _try("eject pilot", les.eject_pilot_level_actor)
    if state.get("camera") is not None:
        camera = state.pop("camera")
        _try("destroy camera", lambda: _sub("EditorActorSubsystem").destroy_actor(camera))
    if state.get("sun") is not None:
        _try("restore sun", lambda: state["sun"].set_actor_rotation(state["sun_rotation"], False))
    for actor in state.pop("hidden", []):
        _try("un-hide %s" % _label(actor), lambda actor=actor: actor.set_is_temporarily_hidden_in_editor(False))
    for flag in ("Tonemapper", "Bloom"):
        _console("ShowFlag.%s 2" % flag)
    if state.get("runup_was") is not None:
        _console("r.HighResScreenshotDelay %d" % state["runup_was"])
    _console("EnableAllScreenMessages")
    if les is not None and state.get("exact_was") is not None:
        _try("restore exact camera view", lambda: les.set_exact_camera_view(state["exact_was"]))
    if les is not None and state.get("game_view_was") is not None:
        _try("restore game view", lambda: les.editor_set_game_view(state["game_view_was"]))


def _finish():
    global handle
    if state.get("finished"):
        return
    state["finished"] = True
    try:
        _restore()
    except Exception as exc:
        log.append("warn  restore: %r" % (exc,))
    log.extend(summary_lines(state.get("order", []), state.get("results", {})))
    path = (state.get("s") or {}).get("log") or os.path.join(os.environ.get("REDROCKS_LOG_DIR") or "/tmp",
                                                             "redrocks_board.log")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write("\n".join(log) + "\n")
    except OSError as exc:
        unreal.log_error("shoot_board: could not write %s: %r" % (path, exc))
    for line in log:
        unreal.log(line)
    if handle is not None:
        unreal.unregister_slate_post_tick_callback(handle)
        handle = None
    unreal.SystemLibrary.quit_editor()


def tick(delta):
    if state.get("finished"):
        return
    try:
        _advance(time.time())
    except Exception as exc:
        log.append("ERROR: %r" % (exc,))
        log.extend("      " + line for line in traceback.format_exc().rstrip().splitlines())
        _finish()


if hasattr(unreal, "register_slate_post_tick_callback"):  # in the editor; tests import the helpers
    handle = unreal.register_slate_post_tick_callback(tick)
