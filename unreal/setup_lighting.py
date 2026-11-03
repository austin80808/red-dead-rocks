"""Add daylight lighting and a player start to Lvl_RedRocks.

Run headless:
    UnrealEditor-Cmd RedRocks.uproject -run=pythonscript \
        -script=unreal/setup_lighting.py -unattended -nosplash -stdout

The level is built programmatically, so it ships with no lighting at all: a
programmatic level has no directional light, no sky, and no atmosphere, and the
viewport renders black until they exist.
"""

import unreal

LEVEL_PATH = "/Game/RedRocks/Lvl_RedRocks"

log = []


def _spawn(cls, location, label):
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor = actors.spawn_actor_from_class(cls, location)
    if actor is None:
        log.append("FAILED %s" % label)
        return None
    actor.set_actor_label(label)
    return actor


def main():
    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not level_subsystem.load_level(LEVEL_PATH):
        log.append("could not load %s" % LEVEL_PATH)
        return

    existing = {
        a.get_actor_label()
        for a in unreal.get_editor_subsystem(
            unreal.EditorActorSubsystem
        ).get_all_level_actors()
    }

    # Sun: Red Rocks sits at 39.7 N, so a mid-morning south-east sun.
    if "Sun" not in existing:
        sun = _spawn(unreal.DirectionalLight, unreal.Vector(0, 0, 10000), "Sun")
        if sun:
            sun.set_actor_rotation(unreal.Rotator(pitch=-38.0, yaw=145.0, roll=0.0), False)
            component = sun.get_component_by_class(unreal.DirectionalLightComponent)
            component.set_editor_property("intensity", 6.0)
            component.set_editor_property("light_color", unreal.Color(255, 246, 230))
            log.append("sun added")

    if "SkyAtmosphere" not in existing:
        _spawn(unreal.SkyAtmosphere, unreal.Vector(0, 0, 0), "SkyAtmosphere")
        log.append("sky atmosphere added")

    if "SkyLight" not in existing:
        sky_light = _spawn(unreal.SkyLight, unreal.Vector(0, 0, 5000), "SkyLight")
        if sky_light:
            component = sky_light.get_component_by_class(unreal.SkyLightComponent)
            component.set_editor_property("real_time_capture", True)
            component.set_editor_property("intensity", 1.0)
            log.append("sky light added")

    if "HeightFog" not in existing:
        _spawn(
            unreal.ExponentialHeightFog, unreal.Vector(0, 0, -20000), "HeightFog"
        )
        log.append("height fog added")

    # Stand the player in the bowl, looking up the seating toward the south.
    if "PlayerStart" not in existing:
        _spawn(unreal.PlayerStart, unreal.Vector(0.0, 0.0, 18200.0), "PlayerStart")
        log.append("player start added")

    level_subsystem.save_current_level()
    log.append("saved %s" % LEVEL_PATH)


try:
    main()
except Exception as exc:
    log.append("ERROR: %r" % (exc,))

with open("/tmp/redrocks_lighting.log", "w") as fh:
    fh.write("\n".join(log))
