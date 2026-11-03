"""Fix exposure and place a camera that actually frames the terrain.

Run headless:
    UnrealEditor-Cmd RedRocks.uproject -run=pythonscript \
        -script=unreal/configure_level.py -unattended -nosplash -stdout

Two problems this solves:

* The level had no exposure set, so Lumen clamps and the viewport renders black.
  A fixed EV100 of 12 is roughly correct for midday daylight.
* The terrain is 2.4 km across with 645 m of relief, so any default camera ends
  up inside it. A CineCameraActor is placed high and south-east, aimed at the
  bowl.
"""

import unreal

LEVEL_PATH = "/Game/RedRocks/Lvl_RedRocks"

log = []


def main():
    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not level_subsystem.load_level(LEVEL_PATH):
        log.append("could not load %s" % LEVEL_PATH)
        return

    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    existing = {a.get_actor_label() for a in actors.get_all_level_actors()}

    for label in ("Camera", "PostProcess"):
        for actor in actors.get_all_level_actors():
            if actor.get_actor_label() == label:
                actors.destroy_actor(actor)

    # Fixed exposure: auto-exposure against a 645 m relief range clamps and the
    # viewport goes black.
    ppv = actors.spawn_actor_from_class(unreal.PostProcessVolume, unreal.Vector(0, 0, 0))
    if ppv:
        ppv.set_actor_label("PostProcess")
        ppv.set_editor_property("unbound", True)
        settings = ppv.get_editor_property("settings")
        settings.set_editor_property("override_auto_exposure_min_brightness", True)
        settings.set_editor_property("auto_exposure_min_brightness", 1.0)
        settings.set_editor_property("override_auto_exposure_max_brightness", True)
        settings.set_editor_property("auto_exposure_max_brightness", 1.0)
        settings.set_editor_property("override_auto_exposure_bias", True)
        settings.set_editor_property("auto_exposure_bias", 12.0)
        ppv.set_editor_property("settings", settings)
        log.append("post process exposure fixed")

    # High south-east vantage looking back at the bowl, like the postcard view.
    cam = actors.spawn_actor_from_class(
        unreal.CineCameraActor, unreal.Vector(60000.0, -90000.0, 40000.0)
    )
    if cam:
        cam.set_actor_label("Camera")
        cam.set_actor_rotation(unreal.Rotator(pitch=-18.0, yaw=25.0, roll=0.0), False)
        log.append("camera placed")

    level_subsystem.save_current_level()
    log.append("saved %s" % LEVEL_PATH)


try:
    main()
except Exception as exc:
    log.append("ERROR: %r" % (exc,))

with open("/tmp/redrocks_configure.log", "w") as fh:
    fh.write("\n".join(log))
