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

Axis and unit conversion: Interchange's USD translator reads the stage upAxis
and converts Y-up to Z-up itself, so no import option is set. It also applies
metersPerUnit (0.01 in this file), but the scan's raw units are really metres,
so the mesh may arrive 100x too small. Rather than trust either behaviour, the
script measures the imported mesh and scales the actor so its longest side is
EXPECTED_LONGEST_M. If the up-axis was NOT converted, set ACTOR_ROTATION below
(roll=90 style, keywords) to correct it on the actor.

Textures are imported explicitly (baseColor, normal, occlusion) and wired into
a small material, so the result does not depend on how Interchange resolves the
USD material. The roughness and metallic maps in the scan are constant 333-byte
JPEGs, so they become scalar values.
"""

import os

import unreal

SCAN_DIR = "/Users/austin/dev/redrocks-ue/assets/scan/usdz_extract"
SCAN_FILE = os.path.join(SCAN_DIR, "scene.usdc")
TEX_DIR = os.path.join(SCAN_DIR, "0")
ASSET_DIR = "/Game/RedRocks/Scan"
LEVEL_PATH = "/Game/RedRocks/Lvl_RedRocks"
ACTOR_LABEL = "Scan_ml0w"
CREDIT = "Red Rocks scan: ml0w (CC BY 4.0)"

# ---- Placement: FIRST GUESS, tune by eye in the editor. ---------------------
# The scan's georeferencing is unknown. The venue sits at the manifest origin
# (0, 0) in the terrain frame, so the scan's bounding-box centre is put on the
# origin, then nudged by SCAN_OFFSET_CM. The scan's lowest point is put on the
# bowl floor: the monolith base elevation (1932.408 m, the same datum
# import_redrocks.py uses for the rocks) minus the terrain minimum
# (1753.858 m), in cm, since the terrain mesh has its lowest point at Z = 0.
BOWL_FLOOR_Z_CM = (1932.408 - 1753.858) * 100.0  # 17855 cm
SCAN_OFFSET_CM = (0.0, 0.0, 0.0)  # extra x, y, z nudge after centring
SCAN_YAW_DEG = 0.0  # spin about the vertical to line the scan up with the bowl
# Raise roll to 90 only if the importer left the scan Y-up (see docstring).
ACTOR_ROTATION = unreal.Rotator(pitch=0.0, yaw=SCAN_YAW_DEG, roll=0.0)
# Longest side of the scan: 163.3 m (ATTRIBUTION.md). 1 raw unit = 1 m, so the
# actor is scaled until the mesh is this long in cm (= 100 cm per raw unit).
EXPECTED_LONGEST_M = 163.3
NOMINAL_SCALE = 100.0  # used only if the measured size is unusable
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


def _find_scan_mesh(paths):
    for p in paths:
        asset_path = p.split(".")[0]
        asset = unreal.EditorAssetLibrary.load_asset(asset_path)
        if isinstance(asset, unreal.StaticMesh):
            return asset, asset_path
    return None, None


def _remove_old_actors(actor_subsystem):
    for actor in actor_subsystem.get_all_level_actors():
        if actor.get_actor_label() in (ACTOR_LABEL, ACTOR_LABEL + "_Credit"):
            actor_subsystem.destroy_actor(actor)


def main():
    if not os.path.exists(SCAN_FILE):
        log.append("ABORT: %s not found" % SCAN_FILE)
        return

    _ensure_dir(ASSET_DIR)

    paths = _import_task(SCAN_FILE, ASSET_DIR)
    log.append("imported scan -> %s" % paths)
    mesh, mesh_path = _find_scan_mesh(paths)
    if mesh is None:
        log.append(
            "ABORT: no StaticMesh imported. Check the USD / Interchange plugins "
            "are enabled and read the UE log for the translator error."
        )
        return

    _enable_nanite(mesh)
    material = _build_material(_import_textures())
    unreal.EditorAssetLibrary.save_asset(mesh_path)

    if not unreal.EditorAssetLibrary.does_asset_exist(LEVEL_PATH):
        log.append("ABORT: %s missing; run import_redrocks.py first" % LEVEL_PATH)
        return
    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    level_subsystem.load_level(LEVEL_PATH)
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    _remove_old_actors(actor_subsystem)

    # spawn_actor_from_object no-ops headless; spawn the class and set the mesh.
    actor = actor_subsystem.spawn_actor_from_class(
        unreal.StaticMeshActor, unreal.Vector(0.0, 0.0, 0.0)
    )
    if actor is None:
        log.append("SPAWN FAILED for %s" % ACTOR_LABEL)
        return
    comp = actor.static_mesh_component
    comp.set_static_mesh(mesh)
    for slot in range(comp.get_num_materials()):
        comp.set_material(slot, material)
    actor.set_actor_label(ACTOR_LABEL)
    actor.set_mobility(unreal.ComponentMobility.STATIC)
    actor.set_actor_rotation(ACTOR_ROTATION, False)

    # Measure at scale 1, then scale so the longest side is the known length.
    _, extent = actor.get_actor_bounds(False)
    longest = 2.0 * max(extent.x, extent.y, extent.z)
    log.append(
        "imported size cm (x,y,z)=(%.1f, %.1f, %.1f)"
        % (2 * extent.x, 2 * extent.y, 2 * extent.z)
    )
    if longest > 0.0:
        scale = EXPECTED_LONGEST_M * 100.0 / longest
    else:
        scale = NOMINAL_SCALE
        log.append("bounds empty, using nominal scale")
    actor.set_actor_scale3d(unreal.Vector(scale, scale, scale))
    log.append("scale factor %.4f (raw metres -> cm would be %.1f)" % (scale, NOMINAL_SCALE))

    # Centre on the origin in XY and put the lowest point on the bowl floor.
    centre, extent = actor.get_actor_bounds(False)
    bottom = centre.z - extent.z
    actor.set_actor_location(
        unreal.Vector(
            -centre.x + SCAN_OFFSET_CM[0],
            -centre.y + SCAN_OFFSET_CM[1],
            BOWL_FLOOR_Z_CM - bottom + SCAN_OFFSET_CM[2],
        ),
        False,
        False,
    )

    # The credit travels with the actor: tags, a Credits folder, and a text
    # actor beside it.
    actor.set_editor_property(
        "tags", [unreal.Name("CC-BY-4.0"), unreal.Name("author:ml0w"), unreal.Name(CREDIT)]
    )
    actor.set_folder_path("Credits")
    try:
        credit = actor_subsystem.spawn_actor_from_class(
            unreal.TextRenderActor, unreal.Vector(0.0, 0.0, BOWL_FLOOR_Z_CM + 300.0)
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
