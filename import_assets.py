"""Generate an Unreal Editor Python script to import the Red Rocks assets.

Runs inside the editor via the Python plugin, or via
`UnrealEditor-Cmd RedRocks.uproject -run=pythonscript -script=...`
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
MANIFEST = json.loads((OUT / "manifest.json").read_text())

ls = MANIFEST["landscape"]
z_scale = ls["z_scale"]
xy_cm = ls["scale_xy_cm"]

script = f'''
import unreal

ASSET_DIR = "/Game/RedRocks"
OUT_DIR = r"{OUT}"

# --- Landscape ---------------------------------------------------------
landscape_tool = unreal.LandscapeEditorObject()
landscape_tool.import_file_path = r"{OUT / 'redrocks_heightmap.png'}"
landscape_tool.import_grid_size = float({xy_cm})
landscape_tool.scale = unreal.Vector({xy_cm}, {xy_cm}, {z_scale})

unreal.log(f"RedRocks: landscape z_scale={{landscape_tool.scale.z}} xy={{landscape_tool.scale.x}}")

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
    unreal.log(f"RedRocks: imported {{t.filename}} -> {{len(t.imported_object_paths)}} objects")

unreal.log("RedRocks: import script complete")
'''

target = ROOT / "import_redrocks.py"
target.write_text(script)
print(f"wrote {target}")
print(f"  z_scale = {z_scale}")
print(f"  xy_cm   = {xy_cm}")
