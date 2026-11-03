"""Point the level viewport camera at the terrain and fix Lumen exposure.

The terrain spans 2.4 km with 645 m of relief, so the default viewport camera
sits inside the mesh and renders black. This places the camera well above the
highest ridge, looking down at the bowl.
"""

import unreal

LEVEL_PATH = "/Game/RedRocks/Lvl_RedRocks"

# Terrain tops out at 645.7 m; 1.4 km gives a clean three-quarter view.
CAMERA_LOCATION = unreal.Vector(70000.0, -110000.0, 140000.0)
CAMERA_ROTATION = unreal.Rotator(pitch=-28.0, yaw=30.0, roll=0.0)

log = []


def main():
    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not level_subsystem.load_level(LEVEL_PATH):
        log.append("could not load %s" % LEVEL_PATH)
        return

    # Auto-exposure over a 645 m relief range clamps to black under Lumen.
    unreal.SystemLibrary.execute_console_command(None, "r.EyeAdaptationQuality 0")
    unreal.SystemLibrary.execute_console_command(
        None, "r.EyeAdaptation.CachedLightingPreExposure 12"
    )

    unreal.EditorLevelLibrary.set_level_viewport_camera_info(
        CAMERA_LOCATION, CAMERA_ROTATION
    )
    log.append("viewport camera set")

    level_subsystem.save_current_level()
    log.append("saved %s" % LEVEL_PATH)


try:
    main()
except Exception as exc:
    log.append("ERROR: %r" % (exc,))

with open("/tmp/redrocks_viewport.log", "w") as fh:
    fh.write("\n".join(log))
