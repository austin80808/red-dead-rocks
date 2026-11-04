"""Import the ml0w drone photogrammetry scan and place it in Lvl_RedRocks.

Run headless (close the editor first; run AFTER import_redrocks.py, which
clears the level's actors each time it runs):
    UnrealEditor-Cmd RedRocks.uproject -run=pythonscript \
        -script=unreal/import_scan.py -unattended -nosplash -stdout

Scan: ml0w, "Red Rocks, Colorado", CC BY 4.0, ~3M triangles, Y-up USD. See
assets/scan/ATTRIBUTION.md. Log: /tmp/redrocks_scan.log

Importer choice: Interchange. In UE 5.8 Interchange is the default importer for
.usd/.usdc/.usdz (the older "USD Importer" plugin path is legacy), and it is
driven by the same unreal.AssetImportTask used for the OBJs, which works in a
commandlet with automated=True. Requires the USD Core / Interchange plugins to
be enabled in RedRocks.uproject.

Axis and units: Interchange converts the stage's Y-up to Z-up itself and splits
the scan into 31 chunk meshes that share one frame. The USD has no real-world
units, so scale, yaw and position are fitted against the terrain by
tools/fit_scan.py (constants below). Re-placing reuses the imported chunks, so
it is light enough to run in the open editor:
    py "/Users/austin/dev/redrocks-ue/unreal/import_scan.py"

Textures are imported explicitly (baseColor, normal, occlusion) and wired into
a small material, so the result does not depend on how Interchange resolves the
USD material. The roughness and metallic maps in the scan are constant 333-byte
JPEGs, so they become scalar values.
"""

import math
import os

import unreal

SCAN_DIR = "/Users/austin/dev/redrocks-ue/assets/scan/usdz_extract"
SCAN_FILE = os.path.join(SCAN_DIR, "scene.usdc")
TEX_DIR = os.path.join(SCAN_DIR, "0")
ASSET_DIR = "/Game/RedRocks/Scan"
LEVEL_PATH = "/Game/RedRocks/Lvl_RedRocks"
ACTOR_LABEL = "Scan_ml0w"
CREDIT = "Red Rocks scan: ml0w (CC BY 4.0)"

# ---- Placement, from tools/fit_scan.py --------------------------------------
# The USD carries no real-world units (Sketchfab normalised them) and the
# extents in ATTRIBUTION.md don't match the file, so scale, yaw and position
# come from fitting the scan's heights to the 3DEP terrain. Re-run
# unreal/export_scan_points.py and tools/fit_scan.py to refresh these. The fit
# is not conclusive (detrended correlation ~0.70); check it by eye.
FIT_SCALE = 31.1814  # cm per scan unit
FIT_YAW_DEG = 292.00
FIT_MIRROR_X = 1
FIT_SCAN_CENTRE = (15.967, 21.806)  # scan units, removed before rotating
FIT_OFFSET_CM = (-7000.0, 40000.0, 21459.2)
# Re-import the USD only when asked; re-placing reuses the existing chunks.
REIMPORT = False
# -----------------------------------------------------------------------------

log = []


def _ensure_dir(path):
    if not unreal.EditorAssetLibrary.does_directory_exist(path):
        unreal.EditorAssetLibrary.make_directory(path)


def _import_task(filename, dest):
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", filename)
    task.set_editor_property("destination_path", dest)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", True)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    return list(task.get_editor_property("imported_object_paths"))


def _import_textures():
    """Import the JPEGs and return {role: Texture2D}."""
    files = {
        "base": "material0_baseColor.jpg",
        "normal": "material0_normal.jpg",
        "occl": "material0_metallicRoughness_occl.jpg",
    }
    out = {}
    for role, fname in files.items():
        path = os.path.join(TEX_DIR, fname)
        if not os.path.exists(path):
            log.append("MISSING texture %s" % path)
            continue
        paths = _import_task(path, ASSET_DIR)
        log.append("imported texture %s -> %s" % (fname, paths))
        if not paths:
            continue
        tex = unreal.EditorAssetLibrary.load_asset(paths[0].split(".")[0])
        if tex is None:
            continue
        if role == "normal":
            tex.set_editor_property(
                "compression_settings", unreal.TextureCompressionSettings.TC_NORMALMAP
            )
            tex.set_editor_property("srgb", False)
        elif role == "occl":
            tex.set_editor_property(
                "compression_settings", unreal.TextureCompressionSettings.TC_MASKS
            )
            tex.set_editor_property("srgb", False)
        unreal.EditorAssetLibrary.save_loaded_asset(tex)
        out[role] = tex
    return out


def _build_material(textures):
    name = "M_Scan_ml0w"
    path = "%s/%s" % (ASSET_DIR, name)
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        unreal.EditorAssetLibrary.delete_asset(path)
    mat = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        name, ASSET_DIR, unreal.Material, unreal.MaterialFactoryNew()
    )
    lib = unreal.MaterialEditingLibrary

    def sample(tex, x, y, normal=False):
        node = lib.create_material_expression(
            mat, unreal.MaterialExpressionTextureSample, x, y
        )
        node.set_editor_property("texture", tex)
        if normal:
            node.set_editor_property(
                "sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL
            )
        return node

    if "base" in textures:
        lib.connect_material_property(
            sample(textures["base"], -600, 0), "RGB", unreal.MaterialProperty.MP_BASE_COLOR
        )
    if "normal" in textures:
        lib.connect_material_property(
            sample(textures["normal"], -600, 300, True),
            "RGB",
            unreal.MaterialProperty.MP_NORMAL,
        )
    if "occl" in textures:
        lib.connect_material_property(
            sample(textures["occl"], -600, 600),
            "R",
            unreal.MaterialProperty.MP_AMBIENT_OCCLUSION,
        )
    rough = lib.create_material_expression(
        mat, unreal.MaterialExpressionConstant, -600, 900
    )
    rough.set_editor_property("r", 0.9)
    lib.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)
    lib.recompile_material(mat)
    unreal.EditorAssetLibrary.save_loaded_asset(mat)
    log.append("built material %s" % path)
    return mat


def _enable_nanite(mesh):
    settings = mesh.get_editor_property("nanite_settings")
    settings.set_editor_property("enabled", True)
    try:
        # Preferred: applies the setting and rebuilds in one step.
        unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem).set_nanite_settings(
            mesh, settings, True
        )
    except Exception as exc:
        log.append("set_nanite_settings failed (%r), setting property instead" % (exc,))
        mesh.set_editor_property("nanite_settings", settings)
    log.append("nanite enabled=%s" % mesh.get_editor_property("nanite_settings").enabled)


def _find_scan_meshes(paths):
    """Every StaticMesh the import produced.

    Interchange splits the Sketchfab USD into one mesh per prim (31 chunks for
    this scan), all authored in the scene's shared frame, so they are placed
    together with one transform.
    """
    meshes = []
    for p in paths:
        asset_path = p.split(".")[0]
        asset = unreal.EditorAssetLibrary.load_asset(asset_path)
        if isinstance(asset, unreal.StaticMesh):
            meshes.append((asset, asset_path))
    return meshes


def _remove_old_actors(actor_subsystem):
    for actor in actor_subsystem.get_all_level_actors():
        if actor.get_actor_label().startswith(ACTOR_LABEL):
            actor_subsystem.destroy_actor(actor)


def _union_bounds(actors):
    """(min, max) corners of all actors' world bounds, as unreal.Vector."""
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for actor in actors:
        centre, extent = actor.get_actor_bounds(False)
        for i, (c, e) in enumerate(((centre.x, extent.x), (centre.y, extent.y), (centre.z, extent.z))):
            lo[i] = min(lo[i], c - e)
            hi[i] = max(hi[i], c + e)
    return unreal.Vector(*lo), unreal.Vector(*hi)


def main():
    if not os.path.exists(SCAN_FILE):
        log.append("ABORT: %s not found" % SCAN_FILE)
        return

    _ensure_dir(ASSET_DIR)

    chunk_dir = ASSET_DIR + "/scene/StaticMeshes"
    if REIMPORT or not unreal.EditorAssetLibrary.does_directory_exist(chunk_dir):
        paths = _import_task(SCAN_FILE, ASSET_DIR)
        log.append("imported scan (%d assets)" % len(paths))
    else:
        paths = list(unreal.EditorAssetLibrary.list_assets(chunk_dir))
        log.append("reusing %d imported chunks (REIMPORT = False)" % len(paths))
    meshes = _find_scan_meshes(paths)
    if not meshes:
        log.append(
            "ABORT: no StaticMesh imported. Check the USD / Interchange plugins "
            "are enabled and read the UE log for the translator error."
        )
        return
    log.append("scan chunks: %d" % len(meshes))

    if REIMPORT or len(meshes) != len(paths) or not unreal.EditorAssetLibrary.does_asset_exist(
        ASSET_DIR + "/M_Scan_ml0w"
    ):
        for mesh, mesh_path in meshes:
            _enable_nanite(mesh)
            unreal.EditorAssetLibrary.save_asset(mesh_path)
        material = _build_material(_import_textures())
    else:
        material = unreal.EditorAssetLibrary.load_asset(ASSET_DIR + "/M_Scan_ml0w")

    if not unreal.EditorAssetLibrary.does_asset_exist(LEVEL_PATH):
        log.append("ABORT: %s missing; run import_redrocks.py first" % LEVEL_PATH)
        return
    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    level_subsystem.load_level(LEVEL_PATH)
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    _remove_old_actors(actor_subsystem)

    # One actor per chunk, all at the origin with the same rotation, so they
    # keep their shared frame. spawn_actor_from_object no-ops headless; spawn
    # the class and set the mesh.
    actors = []
    for i, (mesh, _) in enumerate(meshes):
        actor = actor_subsystem.spawn_actor_from_class(
            unreal.StaticMeshActor, unreal.Vector(0.0, 0.0, 0.0)
        )
        if actor is None:
            log.append("SPAWN FAILED for chunk %d" % i)
            continue
        comp = actor.static_mesh_component
        comp.set_static_mesh(mesh)
        for slot in range(comp.get_num_materials()):
            comp.set_material(slot, material)
        actor.set_actor_label("%s_%02d" % (ACTOR_LABEL, i))
        actor.set_mobility(unreal.ComponentMobility.STATIC)
        # The credit travels with every chunk: tags plus a shared folder.
        actor.set_editor_property(
            "tags", [unreal.Name("CC-BY-4.0"), unreal.Name("author:ml0w"), unreal.Name(CREDIT)]
        )
        actor.set_folder_path("Scan_ml0w")
        actors.append(actor)
    if not actors:
        log.append("ABORT: no scan actors spawned")
        return

    # Every chunk shares the scan's frame, so all get the same transform:
    # world = R(yaw) . diag(mirror * s, s, s) . (p - centre) + offset, which as
    # an actor transform is scale (mirror*s, s, s), the yaw, and a location of
    # offset - R . S . centre.
    s = FIT_SCALE
    yaw = math.radians(FIT_YAW_DEG)
    cx = FIT_MIRROR_X * s * FIT_SCAN_CENTRE[0]
    cy = s * FIT_SCAN_CENTRE[1]
    location = unreal.Vector(
        FIT_OFFSET_CM[0] - (math.cos(yaw) * cx - math.sin(yaw) * cy),
        FIT_OFFSET_CM[1] - (math.sin(yaw) * cx + math.cos(yaw) * cy),
        FIT_OFFSET_CM[2],
    )
    for actor in actors:
        actor.set_actor_scale3d(unreal.Vector(FIT_MIRROR_X * s, s, s))
        actor.set_actor_rotation(unreal.Rotator(pitch=0.0, yaw=FIT_YAW_DEG, roll=0.0), False)
        actor.set_actor_location(location, False, False)
    lo, hi = _union_bounds(actors)
    log.append(
        "placed %d chunks at %s; size m (x,y,z)=(%.1f, %.1f, %.1f)"
        % (len(actors), location, (hi.x - lo.x) / 100.0, (hi.y - lo.y) / 100.0, (hi.z - lo.z) / 100.0)
    )
    credit_z = hi.z + 500.0
    try:
        credit = actor_subsystem.spawn_actor_from_class(
            unreal.TextRenderActor, unreal.Vector(location.x, location.y, credit_z)
        )
        credit.text_render.set_text(CREDIT)
        credit.text_render.set_editor_property("world_size", 120.0)
        credit.set_actor_label(ACTOR_LABEL + "_Credit")
        credit.set_folder_path("Credits")
    except Exception as exc:
        log.append("credit text actor failed: %r" % (exc,))
    log.append("CREDIT: %s" % CREDIT)

    level_subsystem.save_current_level()
    log.append("placed %s and saved %s" % (ACTOR_LABEL, LEVEL_PATH))


try:
    main()
except Exception as exc:
    log.append("ERROR: %r" % (exc,))

with open("/tmp/redrocks_scan.log", "w") as fh:
    fh.write("\n".join(log))
