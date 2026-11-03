"""Report the actors and bounds in the generated level."""

import unreal

LEVEL_PATH = "/Game/RedRocks/Lvl_RedRocks"
lines = []

if unreal.EditorAssetLibrary.does_asset_exist(LEVEL_PATH):
    unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).load_level(LEVEL_PATH)
    for actor in unreal.EditorLevelLibrary.get_all_level_actors():
        loc = actor.get_actor_location()
        label = actor.get_actor_label()
        comp = actor.get_component_by_class(unreal.StaticMeshComponent)
        if comp and comp.static_mesh:
            mesh = comp.static_mesh
            lines.append(
                "%s tris=%d verts=%d at (%.1f, %.1f, %.1f)"
                % (
                    label,
                    mesh.get_num_triangles(0),
                    mesh.get_num_vertices(0),
                    loc.x,
                    loc.y,
                    loc.z,
                )
            )
        else:
            lines.append("%s (%s) at (%.1f, %.1f, %.1f)" % (label, type(actor).__name__, loc.x, loc.y, loc.z))
else:
    lines.append("LEVEL MISSING")

with open("/tmp/redrocks_level.log", "w") as fh:
    fh.write("\n".join(lines))
