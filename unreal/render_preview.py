"""Render Lvl_RedRocks to a PNG via a SceneCapture2D.

Run headless:
    UnrealEditor-Cmd RedRocks.uproject -run=pythonscript \
        -script=unreal/render_preview.py -unattended -nosplash -stdout

Unreal's high-res screenshot path requires a live viewport and crashes the
headless commandlet, so the scene is captured with a SceneCaptureComponent2D
into a render target and exported to PNG instead. That goes through the real
renderer, so it is honest evidence that the geometry, materials and lighting
all resolve.
"""

import os

import unreal

LEVEL_PATH = "/Game/RedRocks/Lvl_RedRocks"
PROJECT = "/Users/austin/Documents/Unreal Projects/RedRocks"
RT_DIR = "/Game/RedRocks"
OUT_PNG = os.path.join(PROJECT, "Saved", "redrocks_preview.png")

# Look north-west down into the bowl from high above the south-east ridge.
CAMERA_LOCATION = unreal.Vector(90000.0, -120000.0, 95000.0)
CAMERA_ROTATION = unreal.Rotator(pitch=-32.0, yaw=38.0, roll=0.0)

WIDTH, HEIGHT = 1600, 900

log = []


def main():
    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not level_subsystem.load_level(LEVEL_PATH):
        log.append("could not load %s" % LEVEL_PATH)
        return

    for cvar in ("r.EyeAdaptationQuality 0", "r.DefaultFeature.AutoExposure 0"):
        unreal.SystemLibrary.execute_console_command(None, cvar)

    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

    for actor in list(actors.get_all_level_actors()):
        if actor.get_actor_label() in ("CaptureCam", "RTCapture"):
            actors.destroy_actor(actor)

    capture = actors.spawn_actor_from_class(
        unreal.SceneCapture2D, CAMERA_LOCATION
    )
    if capture is None:
        log.append("could not spawn SceneCapture2D")
        return
    capture.set_actor_label("CaptureCam")
    capture.set_actor_rotation(CAMERA_ROTATION, False)

    component = capture.get_component_by_class(unreal.SceneCaptureComponent2D)
    if component is None:
        log.append("no SceneCaptureComponent2D")
        return

    component.set_editor_property("fov_angle", 70.0)
    component.set_editor_property(
        "capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR
    )
    component.set_editor_property("capture_every_frame", False)

    render_target = unreal.RenderingLibrary.create_render_target2d(
        None, WIDTH, HEIGHT, unreal.TextureRenderTargetFormat.RTF_RGBA8
    )
    if render_target is None:
        log.append("render target creation failed")
        return
    component.set_editor_property("texture_target", render_target)

    component.capture_scene()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    ok = unreal.RenderingLibrary.export_render_target(
        None, render_target, os.path.dirname(OUT_PNG), os.path.basename(OUT_PNG)
    )
    log.append("export_render_target -> %s (exists=%s)" % (ok, os.path.exists(OUT_PNG)))

    level_subsystem.save_current_level()
    log.append("saved level")


try:
    main()
except Exception as exc:
    log.append("ERROR: %r" % (exc,))

with open("/tmp/redrocks_render.log", "w") as fh:
    fh.write("\n".join(log))
