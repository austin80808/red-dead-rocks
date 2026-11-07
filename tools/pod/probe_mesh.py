"""Triangle counts of the level's static meshes (did a re-import take?)."""
import unreal

out = []
for a in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
    if isinstance(a, unreal.StaticMeshActor):
        m = a.static_mesh_component.static_mesh
        if m:
            out.append("%-24s %8d tris  %s" % (a.get_actor_label(), m.get_num_triangles(0), m.get_path_name()))
open("/workspace/work/redrocks/logs/meshes.txt", "w").write("\n".join(sorted(out)))
