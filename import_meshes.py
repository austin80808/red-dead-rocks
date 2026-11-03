import unreal

OUT = "/Users/austin/dev/redrocks-ue/out"
DEST = "/Game/RedRocks"
log = []

def ensure(path):
    if not unreal.EditorAssetLibrary.does_directory_exist(path):
        unreal.EditorAssetLibrary.make_directory(path)

ensure(DEST)

tasks = []
for name in ["Creation_Rock", "Ship_Rock"]:
    t = unreal.AssetImportTask()
    t.set_editor_property("filename", f"{OUT}/{name}.obj")
    t.set_editor_property("destination_path", DEST)
    t.set_editor_property("automated", True)
    t.set_editor_property("replace_existing", True)
    t.set_editor_property("save", True)
    tasks.append(t)

unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks(tasks)

for t in tasks:
    log.append(f"{t.get_editor_property('filename')} -> {list(t.get_editor_property('imported_object_paths'))}")

log.append("ASSETS: " + ",".join(unreal.EditorAssetLibrary.list_assets(DEST)))
open("/tmp/import_out.txt", "w").write("\n".join(log))
