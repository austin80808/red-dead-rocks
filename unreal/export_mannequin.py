"""Export the UE5 mannequin (SKM_Manny_Simple) as a binary glTF, the base the
sasquatch is built from (`redrocks sasquatch` reads it).

Headless:
    UnrealEditor-Cmd RedRocks.uproject -run=pythonscript \
        -script=unreal/export_mannequin.py -unattended -nosplash -stdout

Writes $REDROCKS_OUT_DIR/sasquatch/manny.glb (default out/ beside this repo) and
logs to $REDROCKS_LOG_DIR/redrocks_export_mannequin.log (default /tmp). The
mannequin is Epic template content: the .glb stays in out/ (git-ignored) and is
never committed to this public repo.

Also logs which Python APIs the sasquatch setup needs are present (IK Rig,
retargeter, groom binding), so a missing one shows up before the long steps.
"""

import os

import unreal

OUT_DIR = os.environ.get("REDROCKS_OUT_DIR", "/Users/austin/dev/redrocks-ue/out")
LOG_PATH = os.path.join(os.environ.get("REDROCKS_LOG_DIR", "/tmp"), "redrocks_export_mannequin.log")
MANNY = "/Game/Characters/Mannequins/Meshes/SKM_Manny_Simple"
NEEDED_APIS = (
    "GLTFExporter",
    "IKRigController",
    "IKRetargeterController",
    "IKRetargetBatchOperation",
    "GroomLibrary",  # UGroomBlueprintLibrary's Python name
    "InterchangeGenericAssetsPipeline",
    "AssetImportTask",
    "SkeletalMeshEditorSubsystem",
)

log = []


def _set(obj, name, value):
    try:
        obj.set_editor_property(name, value)
        return True
    except Exception as exc:  # property names move between engine versions
        log.append("note  option %s not set: %r" % (name, exc))
        return False


def main():
    os.makedirs(os.path.join(OUT_DIR, "sasquatch"), exist_ok=True)
    for name in NEEDED_APIS:
        log.append("%s  api %s" % ("ok   " if hasattr(unreal, name) else "FAIL ", name))
    extra = sorted(n for n in dir(unreal) if any(k in n for k in ("IKRig", "Retarget", "Groom")))
    log.append("info  related api: %s" % ", ".join(extra))

    mesh = unreal.load_asset(MANNY)
    if mesh is None:
        log.append("FAIL  could not load %s" % MANNY)
        return
    log.append("ok    loaded %s (skeleton %s)" % (MANNY, mesh.skeleton.get_path_name()))

    options = unreal.GLTFExportOptions()
    _set(options, "export_vertex_skin_weights", True)
    _set(options, "export_vertex_colors", False)
    _set(options, "export_uniform_scale", 0.01)
    _set(options, "default_level_of_detail", 0)
    _set(options, "export_animation_sequences", False)
    _set(options, "export_unlit_materials", False)
    _set(options, "texture_image_format", unreal.GLTFTextureImageFormat.NONE)
    _set(options, "bake_material_inputs", unreal.GLTFMaterialBakeMode.DISABLED)

    path = os.path.join(OUT_DIR, "sasquatch", "manny.glb")
    result = unreal.GLTFExporter.export_to_gltf(mesh, path, options, set())
    ok = result[0] if isinstance(result, tuple) else bool(result)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    log.append("%s  export %s (%d bytes)" % ("ok   " if ok and size else "FAIL ", path, size))


try:
    main()
except Exception as exc:
    log.append("FAIL  %r" % (exc,))
finally:
    with open(LOG_PATH, "w") as fh:
        fh.write("\n".join(log) + "\n")
    print("\n".join(log))
