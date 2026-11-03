"""Import a single OBJ with verbose reporting to find why geometry is lost."""

import os
import unreal

OBJ = "/Users/austin/dev/redrocks-ue/out/RedRocks_Terrain.obj"
DEST = "/Game/RedRocksProbe"

lines = []

if not unreal.EditorAssetLibrary.does_directory_exist(DEST):
    unreal.EditorAssetLibrary.make_directory(DEST)

task = unreal.AssetImportTask()
task.set_editor_property("filename", OBJ)
task.set_editor_property("destination_path", DEST)
task.set_editor_property("automated", True)
task.set_editor_property("replace_existing", True)
task.set_editor_property("save", False)

# Turn on the interchange details so we can see what the translator decided.
options = unreal.InterchangeImportSettings()
try:
    options.set_editor_property("show_import_dialog", False)
except Exception as exc:
    lines.append("options: %r" % (exc,))

unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])

for path in task.get_editor_property("imported_object_paths"):
    lines.append("path %s" % path)
    mesh = unreal.EditorAssetLibrary.load_asset(path)
    lines.append("  type %s" % type(mesh).__name__)
    if isinstance(mesh, unreal.StaticMesh):
        lines.append("  tris %s" % mesh.get_num_triangles(0))
        lines.append("  verts %s" % mesh.get_num_vertices(0))
        lines.append("  uvs %s" % mesh.get_num_uv_channels(0) if hasattr(mesh, "get_num_uv_channels") else "  uvs ?")
        bounds = mesh.get_bounding_box()
        lines.append("  bounds min %s max %s" % (bounds.min, bounds.max))

with open("/tmp/redrocks_probe_import.log", "w") as fh:
    fh.write("\n".join(str(x) for x in lines))
