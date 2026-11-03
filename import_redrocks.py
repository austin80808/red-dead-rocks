
import unreal

ASSET_DIR = "/Game/RedRocks"
OUT_DIR = r"/Users/austin/dev/redrocks-ue/out"

# --- Landscape ---------------------------------------------------------
landscape_tool = unreal.LandscapeEditorObject()
landscape_tool.import_file_path = r"/Users/austin/dev/redrocks-ue/out/redrocks_heightmap.png"
landscape_tool.import_grid_size = float(118.9886)
landscape_tool.scale = unreal.Vector(118.9886, 118.9886, 126.201296)

unreal.log(f"RedRocks: landscape z_scale={landscape_tool.scale.z} xy={landscape_tool.scale.x}")

# --- Static meshes (monoliths + scan) ----------------------------------
tasks = []
for name in ["Creation_Rock", "Ship_Rock"]:
    t = unreal.AssetImportTask()
    t.filename = str(OUT_DIR + "/" + name + ".obj")
    t.destination_path = ASSET_DIR
    t.automated = True
    t.replace_existing = True
    t.save = True
    tasks.append(t)

unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks(tasks)
for t in tasks:
    unreal.log(f"RedRocks: imported {t.filename} -> {len(t.imported_object_paths)} objects")

unreal.log("RedRocks: import script complete")
