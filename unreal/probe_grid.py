"""Import terrain meshes at several grid sizes to find where geometry is lost."""

import unreal

DEST = "/Game/RedRocksGrid"
lines = []

if not unreal.EditorAssetLibrary.does_directory_exist(DEST):
    unreal.EditorAssetLibrary.make_directory(DEST)

for grid in (65, 129, 257, 377):
    path = "/tmp/t_%d.obj" % grid
    if grid == 377:
        path = "/Users/austin/dev/redrocks-ue/out/RedRocks_Terrain.obj"

    task = unreal.AssetImportTask()
    task.set_editor_property("filename", path)
    task.set_editor_property("destination_path", DEST)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", False)

    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])

    for obj_path in task.get_editor_property("imported_object_paths"):
        mesh = unreal.EditorAssetLibrary.load_asset(obj_path)
        if isinstance(mesh, unreal.StaticMesh):
            expected = (grid - 1) * (grid - 1) * 2
            lines.append(
                "grid %d: expected %d tris, got %d (%d verts)"
                % (grid, expected, mesh.get_num_triangles(0), mesh.get_num_vertices(0))
            )

with open("/tmp/redrocks_probe_grid.log", "w") as fh:
    fh.write("\n".join(lines))
