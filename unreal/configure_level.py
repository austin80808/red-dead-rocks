"""Turn on Lumen and auto exposure, grade the image, and place a framing camera.

Run headless:
    UnrealEditor-Cmd RedRocks.uproject -run=pythonscript \
        -script=unreal/configure_level.py -unattended -nosplash -stdout

Three things:

* Project render settings. The project's Config/DefaultEngine.ini had a
  [SystemSettings] block that switched off Lumen, distance fields, virtual
  shadow maps and eye adaptation, which is why the level needed a fake 6 lux sun
  under a fixed +12 exposure bias, and why shade went black. This script writes
  RENDER_SETTINGS into that block (other lines are left alone). The editor reads
  it at start-up, so restart the editor after a rebuild that changes it.
* An unbound PostProcessVolume: Lumen GI and reflections with a view distance
  that covers the bowl from the air, histogram auto exposure clamped to a
  daylight EV100 range (so stepping into shade opens up a little but the sky
  never blows out), and a light grade.
* A CineCameraActor high and south-east, aimed at the bowl (fix_camera.py sets
  its final position).
"""

import os

import unreal

LEVEL_PATH = "/Game/RedRocks/Lvl_RedRocks"
LOG_PATH = os.path.join(os.environ.get("REDROCKS_LOG_DIR", "/tmp"), "redrocks_configure.log")

# Console variables the level depends on, written to [SystemSettings].
RENDER_SETTINGS = [
    ("r.DynamicGlobalIlluminationMethod", "1"),  # Lumen
    ("r.ReflectionMethod", "1"),                 # Lumen
    ("r.Lumen.DiffuseIndirect.Allow", "1"),
    ("r.Lumen.Reflections.Allow", "1"),
    ("r.GenerateMeshDistanceFields", "True"),    # software Lumen traces these
    ("r.Shadow.Virtual.Enable", "1"),            # crisp shadows near and far
    ("r.EyeAdaptationQuality", "2"),
    ("r.DefaultFeature.AutoExposure", "1"),
    ("r.DefaultFeature.AutoExposure.Method", "0"),  # histogram
    ("r.AntiAliasingMethod", "4"),               # TSR: no shimmer on thin lines
]

# Exposure range, EV100 (the project extends the luminance range, so the
# min/max brightness fields are EV100). Sunlit sandstone at golden hour meters
# about 13.5; deep shade under the rocks about 11.
EXPOSURE_MIN_EV100 = 11.0
EXPOSURE_MAX_EV100 = 14.2
EXPOSURE_COMPENSATION = 0.3

# Lumen defaults trace 200 m; the bowl is ~400 m across and the aerial camera
# sits 600 m out, so let the GI see further.
LUMEN_VIEW_DISTANCE_CM = 120000.0
LUMEN_TRACE_DISTANCE_CM = 60000.0

log = []


def patch_system_settings(text, settings):
    """Return ini text with each (key, value) set inside [SystemSettings].

    Plain text handling, tested offline. Existing keys in that section are
    rewritten in place, missing ones appended to it, a missing section added at
    the end. Lines in other sections are never touched.
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "[SystemSettings]":
            start = i
            break
    if start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("[SystemSettings]")
        start = len(lines) - 1
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].strip().startswith("["):
            end = i
            break
    wanted = dict(settings)
    for i in range(start + 1, end):
        key = lines[i].split("=", 1)[0].strip()
        if key in wanted:
            lines[i] = "%s=%s" % (key, wanted.pop(key))
    # Append the rest right after the section's last non-blank line.
    insert_at = end
    while insert_at > start + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1
    for key, value in settings:
        if key in wanted:
            lines.insert(insert_at, "%s=%s" % (key, value))
            insert_at += 1
    return "\n".join(lines) + "\n"


def _step(label, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        log.append("ok    %s" % label)
        return result
    except Exception as exc:
        log.append("FAIL  %s: %r" % (label, exc))
        return None


def _render_settings():
    config_dir = unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_config_dir())
    path = os.path.join(config_dir, "DefaultEngine.ini")
    with open(path) as fh:
        old = fh.read()
    new = patch_system_settings(old, RENDER_SETTINGS)
    if new == old:
        return "%s already set" % path
    if not os.access(path, os.W_OK):
        os.chmod(path, 0o644)
    with open(path, "w") as fh:
        fh.write(new)
    return "%s updated: restart the editor to pick it up" % path


def _override(settings, **props):
    for key, value in props.items():
        settings.set_editor_property("override_" + key, True)
        settings.set_editor_property(key, value)


def _post_process(actors):
    ppv = actors.spawn_actor_from_class(unreal.PostProcessVolume, unreal.Vector(0, 0, 0))
    if ppv is None:
        raise RuntimeError("could not spawn PostProcessVolume")
    ppv.set_actor_label("PostProcess")
    ppv.set_editor_property("unbound", True)
    s = ppv.get_editor_property("settings")
    _override(
        s,
        dynamic_global_illumination_method=unreal.DynamicGlobalIlluminationMethod.LUMEN,
        reflection_method=unreal.ReflectionMethod.LUMEN,
        lumen_scene_view_distance=LUMEN_VIEW_DISTANCE_CM,
        lumen_max_trace_distance=LUMEN_TRACE_DISTANCE_CM,
        lumen_final_gather_quality=1.5,
        auto_exposure_method=unreal.AutoExposureMethod.AEM_HISTOGRAM,
        auto_exposure_min_brightness=EXPOSURE_MIN_EV100,
        auto_exposure_max_brightness=EXPOSURE_MAX_EV100,
        auto_exposure_bias=EXPOSURE_COMPENSATION,
        auto_exposure_speed_up=5.0,
        auto_exposure_speed_down=3.0,
        auto_exposure_apply_physical_camera_exposure=False,
        bloom_intensity=0.35,
        lens_flare_intensity=0.0,
        vignette_intensity=0.3,
        color_saturation=unreal.Vector4(x=1.06, y=1.06, z=1.06, w=1.0),
        color_contrast=unreal.Vector4(x=1.04, y=1.04, z=1.04, w=1.0),
    )
    ppv.set_editor_property("settings", s)
    return "EV100 %.1f..%.1f" % (EXPOSURE_MIN_EV100, EXPOSURE_MAX_EV100)


def _camera(actors):
    cam = actors.spawn_actor_from_class(
        unreal.CineCameraActor, unreal.Vector(60000.0, -90000.0, 40000.0)
    )
    if cam is None:
        raise RuntimeError("could not spawn camera")
    cam.set_actor_label("Camera")
    cam.set_actor_rotation(unreal.Rotator(pitch=-18.0, yaw=25.0, roll=0.0), False)


def _save():
    if not unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level():
        raise RuntimeError("save_current_level returned False (read-only .umap?)")


def main():
    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not level_subsystem.load_level(LEVEL_PATH):
        log.append("ERROR: could not load %s" % LEVEL_PATH)
        return

    detail = _step("render settings (Lumen, distance fields, VSM, eye adaptation)",
                   _render_settings)
    if detail:
        log.append("      %s" % detail)

    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for actor in list(actors.get_all_level_actors()):
        if actor.get_actor_label() in ("Camera", "PostProcess"):
            actors.destroy_actor(actor)

    detail = _step("post process: Lumen, auto exposure, grade", _post_process, actors)
    if detail:
        log.append("      %s" % detail)
    _step("camera", _camera, actors)
    _step("save %s" % LEVEL_PATH, _save)


if hasattr(unreal, "EditorAssetLibrary"):
    try:
        main()
    except Exception as exc:
        log.append("ERROR: %r" % (exc,))
    with open(LOG_PATH, "w") as fh:
        fh.write("\n".join(log) + "\n")
