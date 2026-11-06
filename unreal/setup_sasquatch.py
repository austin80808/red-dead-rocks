"""Put the sasquatch into the project: the player character.

Needs `redrocks sasquatch` to have written out/sasquatch/ (sasquatch.glb, the two
.rrgroom fur files and manifest.json), the HairStrands plugin and the
RedRocksEditor module (its .rrgroom translator) built. Headless:
    UnrealEditor-Cmd RedRocks.uproject -run=pythonscript \
        -script=unreal/setup_sasquatch.py -unattended -nosplash -stdout

Everything goes under /Game/Sasquatch, logged to
$REDROCKS_LOG_DIR/redrocks_sasquatch.log (default /tmp):
  1. Imports sasquatch.glb as SKM_Sasquatch with its own skeleton SK_Sasquatch
     (the mannequin's bone names, his proportions) and checks its height.
  2. Materials: M_SasquatchSkin (subsurface skin, leathery and dark where it is
     bare, fur-root brown under the fur, procedural wrinkles), M_SasquatchEye
     (wet, dark), M_SasquatchFur (hair shading coloured by the strands).
  3. Imports the fur as GR_Sasquatch_Body and GR_Sasquatch_Head and binds each
     to the mesh (GB_*), so it follows the skin.
  4. IK Rigs for the mannequin and for him (auto-generated chains and full-body
     IK), and RTG_Manny_To_Sasquatch: a lowered pelvis and a forward-pitched
     spine in his retarget pose give the bent-knee, compliant gait.
  5. Batch-retargets the template's ABP_Unarmed with everything it plays
     (idle, 8-way walk and jog, jump, fall, land, the blend space) into
     /Game/Sasquatch/Anims as ABP_Sasquatch. ARRExplorer loads it by path.
Safe to re-run: it replaces what it made before.
"""

import json
import os

import unreal

OUT_DIR = os.environ.get("REDROCKS_OUT_DIR", "/Users/austin/dev/redrocks-ue/out")
SRC = os.path.join(OUT_DIR, "sasquatch")
LOG_PATH = os.path.join(os.environ.get("REDROCKS_LOG_DIR", "/tmp"), "redrocks_sasquatch.log")
GAME = "/Game/Sasquatch"
ANIMS = GAME + "/Anims"
RIGS = GAME + "/Rigs"
MANNY = "/Game/Characters/Mannequins/Meshes/SKM_Manny_Simple"
ABP_UNARMED = "/Game/Characters/Mannequins/Anims/Unarmed/ABP_Unarmed"

log = []
eal = unreal.EditorAssetLibrary
lib = unreal.MaterialEditingLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()


def _step(label, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        log.append("ok    %s%s" % (label, "" if result is None else ": %s" % (result,)))
        return result
    except Exception as exc:
        log.append("FAIL  %s: %r" % (label, exc))
        return None


def _save(asset):
    if asset is not None and not eal.save_loaded_asset(asset, False):
        raise RuntimeError("save failed for %s" % asset.get_path_name())


def _delete(path):
    if eal.does_asset_exist(path):
        eal.delete_asset(path)


def _assets_in(path, cls):
    out = []
    for p in eal.list_assets(path, recursive=False, include_folder=False):
        a = eal.load_asset(p)
        if isinstance(a, cls):
            out.append(a)
    return out


def _rename(asset, path):
    if asset.get_path_name().split(".")[0] == path:
        return asset
    _delete(path)
    if not eal.rename_asset(asset.get_path_name().split(".")[0], path):
        raise RuntimeError("could not rename %s to %s" % (asset.get_path_name(), path))
    return eal.load_asset(path)


# --------------------------------------------------------------------- import

def import_mesh(manifest):
    glb = os.path.join(SRC, "sasquatch.glb")
    if not os.path.exists(glb):
        raise RuntimeError("%s missing: run `redrocks sasquatch` first" % glb)
    for p in eal.list_assets(GAME, recursive=False, include_folder=False):
        a = eal.load_asset(p)
        if isinstance(a, (unreal.SkeletalMesh, unreal.Skeleton, unreal.PhysicsAsset)) or \
                isinstance(a, unreal.MaterialInstance):
            eal.delete_asset(p.split(".")[0])
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", glb)
    task.set_editor_property("destination_path", GAME)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", True)
    tools.import_asset_tasks([task])
    meshes = _assets_in(GAME, unreal.SkeletalMesh)
    if not meshes:
        raise RuntimeError("no skeletal mesh after importing %s (imported: %s)"
                           % (glb, list(task.get_editor_property("imported_object_paths"))))
    mesh = _rename(meshes[0], GAME + "/SKM_Sasquatch")
    skel = mesh.get_editor_property("skeleton")
    _rename(skel, GAME + "/SK_Sasquatch")
    mesh = eal.load_asset(GAME + "/SKM_Sasquatch")
    for phys in _assets_in(GAME, unreal.PhysicsAsset):
        _rename(phys, GAME + "/PHYS_Sasquatch")
    want = manifest["height_cm"]
    try:
        height = 2 * mesh.get_bounds().box_extent.z
    except Exception as exc:  # the bounds call moved between engine versions; the render checks his size
        log.append("note  bounds not read: %r" % (exc,))
        return "%s imported (generator says %.1f cm)" % (mesh.get_name(), want)
    if abs(height - want) > 0.05 * want:
        raise RuntimeError("imported height %.1f cm, generator says %.1f cm" % (height, want))
    return "%.1f cm tall" % height


# ------------------------------------------------------------------ materials

def srgb(hex_color):
    h = hex_color.lstrip("#")
    out = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return unreal.LinearColor(r=out[0], g=out[1], b=out[2], a=1.0)


class Graph:
    def __init__(self, mat):
        self.mat = mat
        self.n = 0

    def node(self, cls_name, **props):
        self.n += 1
        node = lib.create_material_expression(self.mat, getattr(unreal, cls_name),
                                              -1800 + 240 * (self.n % 8), -600 + 180 * (self.n // 8))
        for k, v in props.items():
            node.set_editor_property(k, v)
        return node

    def val(self, v):
        if isinstance(v, (int, float)):
            return self.node("MaterialExpressionConstant", r=float(v))
        return v

    def link(self, src, dst, pin, src_pin=""):
        if not lib.connect_material_expressions(src, src_pin, dst, pin):
            raise RuntimeError("connect failed: %s.%s -> %s.%s" % (src.get_name(), src_pin, dst.get_name(), pin))

    def op(self, cls, a, b):
        n = self.node(cls)
        self.link(self.val(a), n, "A")
        self.link(self.val(b), n, "B")
        return n

    def mul(self, a, b):
        return self.op("MaterialExpressionMultiply", a, b)

    def add(self, a, b):
        return self.op("MaterialExpressionAdd", a, b)

    def lerp(self, a, b, t):
        n = self.node("MaterialExpressionLinearInterpolate")
        self.link(self.val(a), n, "A")
        self.link(self.val(b), n, "B")
        self.link(self.val(t), n, "Alpha")
        return n

    def mask(self, a, channel):
        return self.unary("MaterialExpressionComponentMask", a, **{c: c == channel for c in "rgba"})

    def unary(self, cls, src, **props):
        # `src` not `a`: ComponentMask's channel flags are r/g/b/a keywords.
        n = self.node(cls, **props)
        for pin in ("", "Input"):
            try:
                if lib.connect_material_expressions(self.val(src), "", n, pin):
                    return n
            except Exception:
                pass
        raise RuntimeError("could not connect into %s" % cls)

    def pre_skinned_position(self):
        """Local position before skinning, so noise rides with the skin; the
        world position when the engine has no such node."""
        try:
            lp = self.node("MaterialExpressionLocalPosition",
                           local_origin=unreal.LocalPositionOrigin.INSTANCE_PRE_SKINNING)
            interp = self.node("MaterialExpressionVertexInterpolator")
            self.link(lp, interp, "Input")
            return interp
        except Exception as exc:
            log.append("note  pre-skinned position unavailable, using world position: %r" % (exc,))
            return self.node("MaterialExpressionWorldPosition")

    def color(self, hex_color):
        return self.node("MaterialExpressionConstant3Vector", constant=srgb(hex_color))

    def noise(self, pos, size_cm, levels=3, func="GRADIENT_TEX"):
        n = self.node("MaterialExpressionNoise", scale=1.0 / size_cm, levels=levels, quality=1,
                      noise_function=getattr(unreal.NoiseFunction, "NOISEFUNCTION_" + func),
                      output_min=0.0, output_max=1.0, turbulence=False)
        self.link(pos, n, "World Position")
        return n

    def out(self, prop, node, pin=""):
        if not lib.connect_material_property(node, pin, getattr(unreal.MaterialProperty, prop)):
            raise RuntimeError("could not connect %s" % prop)


def _material(name):
    path = "%s/%s" % (GAME, name)
    if eal.does_asset_exist(path):
        mat = eal.load_asset(path)
        lib.delete_all_material_expressions(mat)
        return mat
    return tools.create_asset(name, GAME, unreal.Material, unreal.MaterialFactoryNew())


def build_skin():
    """Bare skin (vertex colour R = 1) is dark grey-brown and leathery, with
    pores and wrinkles from noise on the pre-skinned position (so the pattern
    rides with the skin); under the fur it is fur-root brown, so gaps between
    strands read as dense coat, not pale skin."""
    mat = _material("M_SasquatchSkin")
    mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_SUBSURFACE_PROFILE)
    mat.set_editor_property("used_with_skeletal_mesh", True)
    g = Graph(mat)
    pos = g.pre_skinned_position()
    bare = g.mask(g.node("MaterialExpressionVertexColor"), "r")
    blotch = g.noise(pos, 6.0, 3)
    skin = g.lerp(g.color("#2E2622"), g.color("#4A3D36"), blotch)
    under = g.color("#2A1C15")
    base = g.lerp(under, skin, bare)
    pores = g.noise(pos, 0.35, 2)
    wrinkles = g.noise(pos, 1.6, 3, func="VORONOI_ALU")
    height = g.add(g.mul(pores, 0.25), g.mul(g.unary("MaterialExpressionOneMinus", wrinkles), 0.9))
    g.out("MP_BASE_COLOR", base)
    g.out("MP_ROUGHNESS", g.lerp(0.9, g.lerp(0.48, 0.66, pores), bare))
    g.out("MP_SPECULAR", g.val(0.35))
    # Wrinkles as a bumped normal (PerturbNormalLQ gives a world-space normal);
    # best effort, the skin still reads without it.
    try:
        mat.set_editor_property("tangent_space_normal", False)
        perturb = g.node("MaterialExpressionMaterialFunctionCall")
        perturb.set_editor_property("material_function", unreal.load_asset(
            "/Engine/Functions/Engine_MaterialFunctions03/Procedurals/PerturbNormalLQ"))
        g.link(g.mul(height, 0.06), perturb, "Bump")
        g.out("MP_NORMAL", perturb)
    except Exception as exc:
        log.append("note  skin normal perturbation skipped: %r" % (exc,))
    lib.recompile_material(mat)
    _save(mat)
    return mat


def build_eye():
    mat = _material("M_SasquatchEye")
    mat.set_editor_property("used_with_skeletal_mesh", True)
    g = Graph(mat)
    g.out("MP_BASE_COLOR", g.color("#1C120C"))
    g.out("MP_ROUGHNESS", g.val(0.04))
    g.out("MP_SPECULAR", g.val(0.6))
    lib.recompile_material(mat)
    _save(mat)
    return mat


def build_fur():
    """Hair shading, coloured by each strand's own root-to-tip colour."""
    mat = _material("M_SasquatchFur")
    mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_HAIR)
    mat.set_editor_property("used_with_hair_strands", True)
    g = Graph(mat)
    attrs = g.node("MaterialExpressionHairAttributes")
    pins = lib.get_material_expression_output_names(attrs) if hasattr(lib, "get_material_expression_output_names") else []
    colour_pin = next((p for p in pins if "color" in str(p).lower()), "Color")
    g.out("MP_BASE_COLOR", attrs, colour_pin)
    g.out("MP_ROUGHNESS", g.val(0.42))
    g.out("MP_SPECULAR", g.val(0.35))
    g.out("MP_METALLIC", g.val(0.6))  # the Hair shading model's Scatter pin
    lib.recompile_material(mat)
    _save(mat)
    return mat


def apply_mesh_materials(skin, eye):
    mesh = eal.load_asset(GAME + "/SKM_Sasquatch")
    mats = mesh.get_editor_property("materials")
    out = []
    for m in mats:
        slot = str(m.get_editor_property("material_slot_name"))
        m.set_editor_property("material_interface", eye if "Eye" in slot else skin)
        out.append(slot)
    mesh.set_editor_property("materials", mats)
    _save(mesh)
    return ", ".join(out)


# ----------------------------------------------------------------------- fur

def import_groom(name, filename, fur_mat):
    path = "%s/%s" % (GAME, name)
    _delete(path)
    src = os.path.join(SRC, filename)
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", src)
    task.set_editor_property("destination_path", GAME)
    task.set_editor_property("destination_name", name)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", False)
    task.set_editor_property("factory", unreal.HairStrandsFactory())
    task.set_editor_property("options", unreal.GroomImportOptions())
    tools.import_asset_tasks([task])
    groom = eal.load_asset(path)
    if groom is None:
        raise RuntimeError("groom import failed for %s" % src)
    mats = list(groom.get_editor_property("hair_groups_materials"))
    if not mats:
        mats = [unreal.HairGroupsMaterial()]
    for m in mats:
        m.set_editor_property("material", fur_mat)
        m.set_editor_property("slot_name", "Fur")
    groom.set_editor_property("hair_groups_materials", mats)
    rendering = list(groom.get_editor_property("hair_groups_rendering"))
    for r in rendering:
        r.set_editor_property("material_slot_name", "Fur")
    groom.set_editor_property("hair_groups_rendering", rendering)
    physics = list(groom.get_editor_property("hair_groups_physics"))
    for p in physics:
        p.get_editor_property("solver_settings").set_editor_property("enable_simulation", False)
    groom.set_editor_property("hair_groups_physics", physics)
    _save(groom)
    desc = groom.get_editor_property("hair_description_groups") if hasattr(groom, "hair_description_groups") else None
    return "%s (%d groups)" % (path, len(rendering))


def bind_groom(groom_name, binding_name):
    path = "%s/%s" % (GAME, binding_name)
    _delete(path)
    groom = eal.load_asset("%s/%s" % (GAME, groom_name))
    mesh = eal.load_asset(GAME + "/SKM_Sasquatch")
    # UGroomBlueprintLibrary is ScriptName "GroomLibrary" in Python.
    binding = unreal.GroomLibrary.create_new_groom_binding_asset_with_path(path, groom, mesh, 100, None, 0)
    if binding is None:
        raise RuntimeError("binding failed")
    _save(binding)
    return path


# --------------------------------------------------------------- retargeting

def make_ik_rig(name, mesh):
    path = "%s/%s" % (RIGS, name)
    _delete(path)
    rig = tools.create_asset(name, RIGS, unreal.IKRigDefinition, unreal.IKRigDefinitionFactory())
    ctrl = unreal.IKRigController.get_controller(rig)
    if not ctrl.set_skeletal_mesh(mesh):
        raise RuntimeError("IK rig %s refused the mesh" % name)
    if not ctrl.apply_auto_generated_retarget_definition():
        raise RuntimeError("no template matched for %s" % name)
    fbik = ctrl.apply_auto_fbik()
    _save(rig)
    return "%s: %d chains, FBIK %s" % (path, len(ctrl.get_retarget_chains()), fbik)


def make_retargeter(manifest):
    path = RIGS + "/RTG_Manny_To_Sasquatch"
    _delete(path)
    rtg = tools.create_asset("RTG_Manny_To_Sasquatch", RIGS, unreal.IKRetargeter, unreal.IKRetargetFactory())
    rc = unreal.IKRetargeterController.get_controller(rtg)
    src_rig = eal.load_asset(RIGS + "/IK_Manny")
    tgt_rig = eal.load_asset(RIGS + "/IK_Sasquatch")
    S, T = unreal.RetargetSourceOrTarget.SOURCE, unreal.RetargetSourceOrTarget.TARGET
    rc.set_ik_rig(S, src_rig)
    rc.set_ik_rig(T, tgt_rig)
    rc.set_preview_mesh(S, eal.load_asset(MANNY))
    rc.set_preview_mesh(T, eal.load_asset(GAME + "/SKM_Sasquatch"))
    if rc.get_num_retarget_ops() == 0:
        rc.add_default_ops()
    rc.assign_ik_rig_to_all_ops(S, src_rig)
    rc.assign_ik_rig_to_all_ops(T, tgt_rig)
    rc.auto_map_chains(unreal.AutoMapChainType.FUZZY, True)
    # His retarget pose: pelvis lowered (knees bend through the stride) and the
    # trunk pitched forward, the head lifted back up to look ahead.
    gait = manifest["gait"]
    rc.set_root_offset_in_retarget_pose(unreal.Vector(0.0, 0.0, -gait["pelvis_drop_cm"]), T)
    # Local-space deltas (x, y, z, w) that pitch each bone in the body frame
    # (redrocks.sasquatch.build._pitch_offset); the retargeter applies them as
    # ref-pose rotation * delta.
    for bone, q in gait["rotation_offsets"].items():
        rc.set_rotation_offset_for_retarget_pose_bone(bone, unreal.Quat(q[0], q[1], q[2], q[3]), T)
    _save(rtg)
    ops = [str(rc.get_op_name(i)) for i in range(rc.get_num_retarget_ops())]
    return "%s ops: %s" % (path, ", ".join(ops))


def retarget_anims():
    for p in eal.list_assets(ANIMS, recursive=True, include_folder=False):
        eal.delete_asset(p.split(".")[0])
    inputs = unreal.IKRetargetBatchOperationInputs()
    reg = unreal.AssetRegistryHelpers.get_asset_registry()
    abp = reg.get_asset_by_object_path(ABP_UNARMED + "." + ABP_UNARMED.rsplit("/", 1)[1])
    inputs.set_editor_property("assets_to_retarget", [abp])
    inputs.set_editor_property("source_mesh", eal.load_asset(MANNY))
    inputs.set_editor_property("target_mesh", eal.load_asset(GAME + "/SKM_Sasquatch"))
    inputs.set_editor_property("ik_retarget_asset", eal.load_asset(RIGS + "/RTG_Manny_To_Sasquatch"))
    inputs.set_editor_property("search", "Unarmed")
    inputs.set_editor_property("replace", "Sasquatch")
    inputs.set_editor_property("target_path", ANIMS)
    inputs.set_editor_property("use_source_path", False)
    inputs.set_editor_property("include_referenced_assets", True)
    inputs.set_editor_property("overwrite_existing_files", True)
    out = unreal.IKRetargetBatchOperation.run_batch_retarget(inputs)
    names = sorted(str(a.asset_name) for a in out)
    for a in out:
        _save(a.get_asset())
    if not eal.does_asset_exist(ANIMS + "/ABP_Sasquatch"):
        raise RuntimeError("no ABP_Sasquatch among %s" % names)
    return "%d assets: %s" % (len(names), ", ".join(names))


def blend_space_top_speed():
    """The fastest ground speed among the blend space's samples (ARRExplorer.AnimTopSpeed)."""
    bs = eal.load_asset(ANIMS + "/BS_Idle_Walk_Run")
    params = [bs.get_editor_property("blend_parameters")[i] for i in range(2)]
    names = [str(p.get_editor_property("display_name")) for p in params]
    axis = next((i for i, n in enumerate(names) if "speed" in n.lower()), 1)
    speeds = []
    for s in bs.get_editor_property("sample_data"):
        v = s.get_editor_property("sample_value")
        speeds.append(v.y if axis == 1 else v.x)
    return "axes %s; top speed %.0f cm/s" % (names, max(speeds))


def main():
    manifest = json.load(open(os.path.join(SRC, "manifest.json")))
    if not eal.does_directory_exist(GAME):
        eal.make_directory(GAME)
    if _step("import SKM_Sasquatch", import_mesh, manifest) is None:
        return
    skin = _step("M_SasquatchSkin", build_skin)
    eye = _step("M_SasquatchEye", build_eye)
    fur = _step("M_SasquatchFur", build_fur)
    _step("mesh materials", apply_mesh_materials, skin, eye)
    for part in ("Body", "Head"):
        if _step("groom %s" % part, import_groom, "GR_Sasquatch_" + part, "fur_%s.rrgroom" % part.lower(), fur):
            _step("bind %s" % part, bind_groom, "GR_Sasquatch_" + part, "GB_Sasquatch_" + part)
    _step("IK_Manny", make_ik_rig, "IK_Manny", eal.load_asset(MANNY))
    _step("IK_Sasquatch", make_ik_rig, "IK_Sasquatch", eal.load_asset(GAME + "/SKM_Sasquatch"))
    if _step("RTG_Manny_To_Sasquatch", make_retargeter, manifest):
        if _step("retarget animations", retarget_anims):
            _step("blend space", blend_space_top_speed)
    log.append("done")


try:
    main()
except Exception as exc:
    log.append("ERROR %r" % (exc,))
finally:
    with open(LOG_PATH, "w") as fh:
        fh.write("\n".join(log) + "\n")
    print("\n".join(log))
