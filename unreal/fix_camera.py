"""Raise the camera above the terrain and remove test scaffolding.

The terrain peaks at 645.7 m (64,572 cm). The camera was placed at 40,000 cm,
which is inside the ridge, so the view was solid rock from the inside.

Run headless:
    UnrealEditor-Cmd RedRocks.uproject -run=pythonscript \
        -script=unreal/fix_camera.py -unattended -nosplash -stdout
"""

import unreal

LEVEL_PATH = "/Game/RedRocks/Lvl_RedRocks"

# 800 m up, 1.6 km south-east of the bowl: clears the 645 m ridge with room.
CAMERA_LOCATION = unreal.Vector(70000.0, -110000.0, 80000.0)
CAMERA_ROTATION = unreal.Rotator(pitch=-24.0, yaw=32.0, roll=0.0)

log = []


def main():
    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not level_subsystem.load_level(LEVEL_PATH):
        log.append("could not load %s" % LEVEL_PATH)
        return

    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

    for actor in list(actors.get_all_level_actors()):
        if actor.get_actor_label() in ("TestSphere", "ProbeCube"):
            actors.destroy_actor(actor)
            log.append("removed %s" % actor.get_actor_label())

    camera = None
    for actor in actors.get_all_level_actors():
        if isinstance(actor, unreal.CameraActor):
            camera = actor
            break

    if camera is None:
        camera = actors.spawn_actor_from_class(unreal.CameraActor, CAMERA_LOCATION)
        if camera:
            camera.set_actor_label("Camera")

    if camera:
        camera.set_actor_location(CAMERA_LOCATION, False, False)
        camera.set_actor_rotation(CAMERA_ROTATION, False)
        log.append("camera at %s" % (CAMERA_LOCATION,))

    level_subsystem.save_current_level()
    log.append("saved")


try:
    main()
except Exception as exc:
    log.append("ERROR: %r" % (exc,))

with open("/tmp/redrocks_camera.log", "w") as fh:
    fh.write("\n".join(log))
