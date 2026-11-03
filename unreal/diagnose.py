"""Report what the open Red Rocks level actually contains.

Run inside the editor (Cmd box):
    py "/Users/austin/dev/redrocks-ue/unreal/diagnose.py"

Writes /tmp/redrocks_diagnose.log: every actor with its location, each static
mesh's triangle count and bounds, the viewport camera, and the startup map.
"""

import unreal

log = []

world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
log.append("world %s" % world.get_path_name())

for actor in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
    line = "%s (%s) at %s hidden=%s" % (
        actor.get_actor_label(),
        type(actor).__name__,
        actor.get_actor_location(),
        actor.is_hidden_ed(),
    )
    if isinstance(actor, unreal.StaticMeshActor):
        mesh = actor.static_mesh_component.static_mesh
        if mesh is None:
            line += " mesh=None"
        else:
            box = mesh.get_bounding_box()
            line += " mesh=%s tris=%d bounds=%s..%s mats=%s" % (
                mesh.get_name(),
                mesh.get_num_triangles(0),
                box.min,
                box.max,
                [m.get_name() if m else None for m in actor.static_mesh_component.get_materials()],
            )
    if isinstance(actor, unreal.DirectionalLight):
        line += " intensity=%s" % actor.light_component.intensity
    log.append(line)

cam = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_level_viewport_camera_info()
log.append("viewport camera %s" % (cam,))

settings = unreal.get_default_object(unreal.GameMapsSettings)
log.append("startup map %s" % settings.get_editor_property("editor_startup_map"))
log.append("game map %s" % settings.get_editor_property("game_default_map"))

with open("/tmp/redrocks_diagnose.log", "w") as fh:
    fh.write("\n".join(str(x) for x in log))
