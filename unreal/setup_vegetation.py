"""Junipers, ponderosa pines and scrub on the Red Rocks hillsides.

Run after setup_environment.py (rebuild_level.py does). Needs
`redrocks vegetation --out out` first: it writes out/vegetation/ (OBJ models,
leaf atlases and vegetation.json with every instance). Headless:
    UnrealEditor-Cmd RedRocks.uproject -run=pythonscript \
        -script=unreal/setup_vegetation.py -unattended -nosplash -stdout

What it does, each step logged to $REDROCKS_LOG_DIR (default /tmp)/redrocks_vegetation.log:
  1. Imports the four leaf atlases (T_Leaves_*) to /Game/RedRocks/Vegetation,
     keeping alpha coverage in the mips so distant cards don't thin out.
  2. Builds three procedural master materials and an instance per species:
       M_Veg_Bark    world-space plates and fissures (orange ponderosa plates,
                     shreddy grey-brown juniper, grey scrub stems)
       M_Veg_Leaves  masked, two-sided, Two Sided Foliage shading: the atlas,
                     tinted per instance, darker inside the crown
       M_Veg_Crown   opaque foliage core with a broken-up normal
  3. Imports every model part with Nanite on (no hand-made LODs needed),
     keeping the generator's bent foliage normals. Tree trunks get
     complex-as-simple collision; canopies and shrubs get none.
  4. Replaces the old sphere-clump "Scrub" actor with one "Vegetation" actor
     holding an instanced-mesh component per model part, filled from
     vegetation.json. Trunks block (BlockAll), everything else is NoCollision.
     Leaf cards stay out of distance-field lighting; shrubs cull past 2 km.
  5. Saves the level, failing loudly if a save returns False.
"""

import json
import math
import os

import unreal

OUT_DIR = os.environ.get("REDROCKS_OUT_DIR", "/Users/austin/dev/redrocks-ue/out")
VEG_DIR = os.path.join(OUT_DIR, "vegetation")
ASSET_DIR = "/Game/RedRocks/Vegetation"
LEVEL_PATH = "/Game/RedRocks/Lvl_RedRocks"
ACTOR_LABEL = "Vegetation"
OLD_LABELS = ["Scrub", "Vegetation"]
LOG_PATH = os.path.join(os.environ.get("REDROCKS_LOG_DIR", "/tmp"), "redrocks_vegetation.log")

TREES = ("JuniperRM", "JuniperOS", "Ponderosa")  # these trunks block the player
SHRUB_CULL_CM = 200000.0

# Bark per species group: (light, dark, plate size cm, vertical stretch, contrast)
BARK = {
    "Ponderosa": ("#C0703F", "#2E211B", 28.0, 0.4, 1.0),
    "Juniper": ("#8C7B6B", "#3B2F28", 9.0, 0.12, 0.8),
    "Shrub": ("#7A6E64", "#3E3630", 6.0, 0.3, 0.5),
}
# Leaf tint pair per atlas (multiplies the atlas colour; picked per instance),
# crown core colour pair, subsurface colour.
FOLIAGE = {
    "Juniper": (("#C8D8E0", "#E2E6D4"), ("#34463C", "#465446"), "#6E7F48"),
    "Pine": (("#D6DEC0", "#B8C49A"), ("#243620", "#34472A"), "#7A8A3A"),
    "Oak": (("#F0F0D0", "#D8D2A0"), ("#3A5022", "#55662C"), "#9AAA40"),
    "Mahogany": (("#E8E8DA", "#D2D0BE"), ("#4E5644", "#646A52"), "#8E9460"),
}

log = []
lib = unreal.MaterialEditingLibrary if hasattr(unreal, "MaterialEditingLibrary") else None
eal = unreal.EditorAssetLibrary if hasattr(unreal, "EditorAssetLibrary") else None


def bark_group(model_name):
    sp = model_name.rsplit("_", 1)[0]
    if sp == "Ponderosa":
        return "Ponderosa"
    if sp.startswith("Juniper"):
        return "Juniper"
    return "Shrub"


def species_of(model_name):
    return model_name.rsplit("_", 1)[0]


def part_collides(model_name, part):
    """Only tree trunks block; canopies and shrubs never do."""
    return part == "Trunk" and species_of(model_name) in TREES


def instance_rows(rows):
    """[(location, rotator kwargs, scale)] from vegetation.json rows."""
    out = []
    for x, y, z, yaw, pitch, roll, sxy, sz in rows:
        out.append(((x, y, z), {"pitch": pitch, "yaw": yaw, "roll": roll}, (sxy, sxy, sz)))
    return out


def srgb_to_linear(hex_color):
    h = hex_color.lstrip("#")
    out = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return out


# --------------------------------------------------------------- Unreal side

def _step(label, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        log.append("ok    %s" % label)
        return result
    except Exception as exc:
        log.append("FAIL  %s: %r" % (label, exc))
        return None


def _lc(hex_color):
    r, g, b = srgb_to_linear(hex_color)
    return unreal.LinearColor(r=r, g=g, b=b, a=1.0)


class Graph:
    """Minimal material-graph wiring (see setup_materials.py for the full one)."""

    def __init__(self, mat):
        self.mat = mat
        self.n = 0

    def node(self, cls_name, **props):
        self.n += 1
        node = lib.create_material_expression(self.mat, getattr(unreal, cls_name),
                                              -2000 + 220 * (self.n % 10), -800 + 160 * (self.n // 10))
        for key, value in props.items():
            node.set_editor_property(key, value)
        return node

    def _in(self, v):
        if isinstance(v, (int, float)):
            return self.node("MaterialExpressionConstant", r=float(v))
        return v

    def link(self, src, dst, pin, src_pin=""):
        if not lib.connect_material_expressions(src, src_pin, dst, pin):
            raise RuntimeError("connect failed: %s.%s -> %s.%s" % (src.get_name(), src_pin, dst.get_name(), pin))

    def link1(self, src, dst, src_pin=""):
        for pin in ("", "Input"):
            try:
                if lib.connect_material_expressions(src, src_pin, dst, pin):
                    return
            except Exception:
                pass
        raise RuntimeError("could not connect %s -> %s" % (src.get_name(), dst.get_name()))

    def bin(self, cls, a, b):
        n = self.node(cls)
        self.link(self._in(a), n, "A")
        self.link(self._in(b), n, "B")
        return n

    def mul(self, a, b):
        return self.bin("MaterialExpressionMultiply", a, b)

    def add(self, a, b):
        return self.bin("MaterialExpressionAdd", a, b)

    def sub(self, a, b):
        return self.bin("MaterialExpressionSubtract", a, b)

    def append(self, a, b):
        return self.bin("MaterialExpressionAppendVector", a, b)

    def lerp(self, a, b, t):
        n = self.node("MaterialExpressionLinearInterpolate")
        self.link(self._in(a), n, "A")
        self.link(self._in(b), n, "B")
        self.link(self._in(t), n, "Alpha")
        return n

    def sat(self, a):
        n = self.node("MaterialExpressionSaturate")
        self.link1(self._in(a), n)
        return n

    def mask(self, a, channel):
        n = self.node("MaterialExpressionComponentMask", **{c: (c == channel) for c in "rgba"})
        self.link1(a, n)
        return n

    def scalar(self, name, value):
        return self.node("MaterialExpressionScalarParameter", parameter_name=name, default_value=float(value))

    def vector(self, name, hex_color):
        return self.node("MaterialExpressionVectorParameter", parameter_name=name, default_value=_lc(hex_color))

    def noise(self, pos, size_cm, levels=2, func="SIMPLEX_TEX", lo=0.0, hi=1.0):
        n = self.node("MaterialExpressionNoise", scale=1.0 / size_cm, levels=levels, quality=1,
                      noise_function=getattr(unreal.NoiseFunction, "NOISEFUNCTION_" + func),
                      output_min=lo, output_max=hi, turbulence=False)
        self.link(pos, n, "World Position")
        return n

    def out(self, prop, node):
        if not lib.connect_material_property(self._in(node), "", getattr(unreal.MaterialProperty, prop)):
            raise RuntimeError("could not wire %s" % prop)


def _usage(mat):
    mat.set_editor_property("used_with_instanced_static_meshes", True)
    try:
        mat.set_editor_property("used_with_nanite", True)
    except Exception:
        pass


def build_bark(mat):
    lib.delete_all_material_expressions(mat)
    _usage(mat)
    g = Graph(mat)
    light, dark = g.vector("BarkLight", "#C0703F"), g.vector("BarkDark", "#2E211B")
    plate = g.scalar("PlateSize", 28.0)
    stretch = g.scalar("Stretch", 0.4)
    contrast = g.scalar("Contrast", 1.0)
    wpos = g.node("MaterialExpressionWorldPosition")
    # Plates elongated up the trunk: squash Z before the cellular noise.
    xy = g.mask(wpos, "r")
    pos = g.append(g.append(xy, g.mask(wpos, "g")), g.mul(g.mask(wpos, "b"), stretch))
    inv = g.node("MaterialExpressionDivide")
    g.link(g._in(1.0), inv, "A")
    g.link(plate, inv, "B")
    cells = g.node("MaterialExpressionNoise", scale=1.0, levels=1, quality=1,
                   noise_function=unreal.NoiseFunction.NOISEFUNCTION_VORONOI_ALU,
                   output_min=0.0, output_max=1.0, turbulence=False)
    g.link(g.mul(pos, inv), cells, "World Position")
    fine = g.noise(wpos, 3.0, levels=2)
    fissure = g.sat(g.mul(g.sub(cells, 0.45), 3.0))
    base = g.lerp(light, dark, g.mul(fissure, contrast))
    base = g.mul(base, g.lerp(0.75, 1.15, fine))
    nz = g.mask(g.node("MaterialExpressionVertexNormalWS"), "b")
    # Dust settles on upward-facing bark.
    base = g.lerp(base, g.node("MaterialExpressionConstant3Vector", constant=_lc("#9C6B4E")),
                  g.mul(g.sat(nz), 0.25))
    g.out("MP_BASE_COLOR", base)
    g.out("MP_ROUGHNESS", 0.9)


def build_leaves(mat):
    lib.delete_all_material_expressions(mat)
    _usage(mat)
    mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED)
    mat.set_editor_property("two_sided", True)
    mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_TWO_SIDED_FOLIAGE)
    mat.set_editor_property("opacity_mask_clip_value", 0.4)
    g = Graph(mat)
    tex = g.node("MaterialExpressionTextureSampleParameter2D", parameter_name="Atlas")
    tint = g.lerp(g.vector("TintA", "#FFFFFF"), g.vector("TintB", "#FFFFFF"),
                  g.node("MaterialExpressionPerInstanceRandom"))
    rgb = g.node("MaterialExpressionComponentMask", r=True, g=True, b=True, a=False)
    g.link1(tex, rgb, "RGB")
    rgb = g.mul(rgb, tint)
    wpos = g.node("MaterialExpressionWorldPosition")
    rgb = g.mul(rgb, g.noise(wpos, 400.0, levels=2, lo=0.78, hi=1.1))
    # Cheap crown occlusion: the bent normal points down inside and under the crown.
    nz = g.mask(g.node("MaterialExpressionVertexNormalWS"), "b")
    rgb = g.mul(rgb, g.lerp(0.6, 1.1, g.sat(g.add(g.mul(nz, 0.6), 0.45))))
    g.out("MP_BASE_COLOR", rgb)
    alpha = g.node("MaterialExpressionComponentMask", r=False, g=False, b=False, a=True)
    g.link1(tex, alpha, "A")
    g.out("MP_OPACITY_MASK", alpha)
    g.out("MP_SUBSURFACE_COLOR", g.mul(g.vector("Subsurface", "#7C8F4A"), 0.35))
    g.out("MP_ROUGHNESS", 0.75)
    g.out("MP_SPECULAR", 0.3)


def build_crown(mat):
    lib.delete_all_material_expressions(mat)
    _usage(mat)
    mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_TWO_SIDED_FOLIAGE)
    g = Graph(mat)
    wpos = g.node("MaterialExpressionWorldPosition")
    col = g.lerp(g.vector("CoreA", "#3A4E40"), g.vector("CoreB", "#4E5E48"),
                 g.node("MaterialExpressionPerInstanceRandom"))
    col = g.mul(col, g.noise(wpos, 35.0, levels=2, lo=0.55, hi=1.15))
    nz = g.mask(g.node("MaterialExpressionVertexNormalWS"), "b")
    col = g.mul(col, g.lerp(0.4, 1.0, g.sat(g.add(g.mul(nz, 0.6), 0.45))))
    nx = g.noise(wpos, 30.0, levels=2, lo=-1.1, hi=1.1)
    ny = g.noise(g.add(wpos, g.node("MaterialExpressionConstant3Vector",
                                    constant=unreal.LinearColor(r=317.0, g=911.0, b=53.0, a=1.0))),
                 30.0, levels=2, lo=-1.1, hi=1.1)
    g.out("MP_BASE_COLOR", col)
    g.out("MP_NORMAL", g.append(g.append(nx, ny), 1.0))
    g.out("MP_SUBSURFACE_COLOR", g.mul(g.vector("Subsurface", "#7C8F4A"), 0.35))
    g.out("MP_ROUGHNESS", 0.85)
    g.out("MP_SPECULAR", 0.25)


def _save(asset):
    if not eal.save_loaded_asset(asset, False):
        raise RuntimeError("save failed for %s (read-only file?)" % asset.get_path_name())


def _get_or_create(name, cls, factory, folder=ASSET_DIR):
    path = "%s/%s" % (folder, name)
    if eal.does_asset_exist(path):
        return eal.load_asset(path)
    return unreal.AssetToolsHelpers.get_asset_tools().create_asset(name, folder, cls, factory)


def _material(name, builder):
    mat = _get_or_create(name, unreal.Material, unreal.MaterialFactoryNew())
    builder(mat)
    lib.recompile_material(mat)
    _save(mat)
    return mat


def _instance(name, parent, scalars=None, vectors=None, textures=None):
    inst = _get_or_create(name, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    inst.set_editor_property("parent", parent)
    for k, v in (scalars or {}).items():
        lib.set_material_instance_scalar_parameter_value(inst, k, v)
    for k, v in (vectors or {}).items():
        lib.set_material_instance_vector_parameter_value(inst, k, _lc(v))
    for k, v in (textures or {}).items():
        lib.set_material_instance_texture_parameter_value(inst, k, v)
    lib.update_material_instance(inst)
    _save(inst)
    return inst


def _import(path, dest):
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", path)
    task.set_editor_property("destination_path", dest)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", True)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    name = os.path.splitext(os.path.basename(path))[0]
    asset = eal.load_asset("%s/%s" % (dest, name))
    if asset is None:
        raise RuntimeError("import produced nothing for %s" % path)
    return asset


def _textures(info):
    out = {}
    for atlas, png in sorted(info["atlases"].items()):
        tex = _import(os.path.join(VEG_DIR, png), ASSET_DIR + "/Textures")
        for key, value in (("srgb", True), ("do_scale_mips_for_alpha_coverage", True),
                           ("alpha_coverage_thresholds", unreal.Vector4(x=0.0, y=0.0, z=0.0, w=0.4))):
            try:
                tex.set_editor_property(key, value)
            except Exception as exc:
                log.append("      %s.%s not set: %r" % (atlas, key, exc))
        _save(tex)
        out[atlas] = tex
    return out


def _materials(textures):
    bark_m = _material("M_Veg_Bark", build_bark)
    leaves_m = _material("M_Veg_Leaves", build_leaves)
    crown_m = _material("M_Veg_Crown", build_crown)
    mats = {}
    for group, (light, dark, size, stretch, contrast) in BARK.items():
        mats[("Bark", group)] = _instance(
            "MI_Veg_Bark_%s" % group, bark_m,
            scalars={"PlateSize": size, "Stretch": stretch, "Contrast": contrast},
            vectors={"BarkLight": light, "BarkDark": dark})
    for atlas, ((ta, tb), (ca, cb), sss) in FOLIAGE.items():
        mats[("Leaves", atlas)] = _instance(
            "MI_Veg_Leaves_%s" % atlas, leaves_m, vectors={"TintA": ta, "TintB": tb, "Subsurface": sss},
            textures={"Atlas": textures[atlas]})
        mats[("Crown", atlas)] = _instance(
            "MI_Veg_Crown_%s" % atlas, crown_m, vectors={"CoreA": ca, "CoreB": cb, "Subsurface": sss})
    return mats


def _meshes(info, mats):
    # The subsystem can be missing in a commandlet; the older library has the same calls.
    sms = (unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
           or getattr(unreal, "EditorStaticMeshLibrary", None))
    out = {}
    for model in info["models"]:
        for part, spec in model["parts"].items():
            mesh = _import(os.path.join(VEG_DIR, spec["obj"]), ASSET_DIR + "/Meshes")
            if not isinstance(mesh, unreal.StaticMesh):
                raise RuntimeError("%s is not a static mesh" % spec["obj"])
            build = sms.get_lod_build_settings(mesh, 0)
            build.set_editor_property("recompute_normals", False)  # keep the bent foliage normals
            build.set_editor_property("recompute_tangents", True)
            sms.set_lod_build_settings(mesh, 0, build)
            nanite = mesh.get_editor_property("nanite_settings")
            nanite.set_editor_property("enabled", True)
            mesh.set_editor_property("nanite_settings", nanite)
            key = ("Bark", bark_group(model["name"])) if part == "Trunk" else (part, model["atlas"])
            mesh.set_material(0, mats[key])
            body = mesh.get_editor_property("body_setup")
            if body is not None:
                flag = (unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE if part_collides(model["name"], part)
                        else unreal.CollisionTraceFlag.CTF_USE_DEFAULT)
                body.set_editor_property("collision_trace_flag", flag)
                mesh.set_editor_property("body_setup", body)
            _save(mesh)
            out[(model["name"], part)] = mesh
    return out


def _actors():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def _place(info, meshes):
    for old in [a for a in _actors().get_all_level_actors() if a.get_actor_label() in OLD_LABELS]:
        _actors().destroy_actor(old)
    actor = _actors().spawn_actor_from_class(unreal.Actor, unreal.Vector(0.0, 0.0, 0.0))
    actor.set_actor_label(ACTOR_LABEL)
    subsystem = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
    root = subsystem.k2_gather_subobject_data_for_instance(actor)[0]
    total = 0
    for model in info["models"]:
        rows = info["instances"].get(model["name"], [])
        if not rows:
            continue
        transforms = [unreal.Transform(location=unreal.Vector(*loc), rotation=unreal.Rotator(**rot),
                                       scale=unreal.Vector(*scale))
                      for loc, rot, scale in instance_rows(rows)]
        for part in model["parts"]:
            handle, fail = subsystem.add_new_subobject(unreal.AddNewSubobjectParams(
                parent_handle=root, new_class=unreal.HierarchicalInstancedStaticMeshComponent,
                blueprint_context=None))
            if not handle or str(fail):
                raise RuntimeError("could not add a component: %s" % fail)
            comp = unreal.SubobjectDataBlueprintFunctionLibrary.get_object(
                unreal.SubobjectDataBlueprintFunctionLibrary.get_data(handle))
            comp.set_static_mesh(meshes[(model["name"], part)])
            comp.set_mobility(unreal.ComponentMobility.STATIC)
            if part_collides(model["name"], part):
                comp.set_collision_profile_name("BlockAll")
            else:
                comp.set_collision_profile_name("NoCollision")
                comp.set_editor_property("can_character_step_up_on", unreal.CanBeCharacterBase.ECB_NO)
            if part == "Leaves":
                comp.set_editor_property("affect_distance_field_lighting", False)
            if species_of(model["name"]) not in TREES:
                comp.set_editor_property("instance_end_cull_distance", int(SHRUB_CULL_CM))
                comp.set_editor_property("instance_start_cull_distance", int(SHRUB_CULL_CM * 0.8))
            comp.add_instances(transforms, False, True)
            if comp.get_instance_count() != len(transforms):
                raise RuntimeError("placed %d of %d %s %s" % (comp.get_instance_count(), len(transforms),
                                                              model["name"], part))
            total += len(transforms)
        log.append("      %-13s %5d instances" % (model["name"], len(rows)))
    return total


def _save_level():
    if not unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level():
        raise RuntimeError("save_current_level returned False (read-only .umap?)")


def main():
    path = os.path.join(VEG_DIR, "vegetation.json")
    if not os.path.exists(path):
        log.append("ABORT: %s not found; run `redrocks vegetation --out %s`" % (path, OUT_DIR))
        return
    with open(path) as fh:
        info = json.load(fh)
    if not info.get("instances"):
        log.append("ABORT: %s has no instances (was it built without a terrain?)" % path)
        return
    if not unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).load_level(LEVEL_PATH):
        log.append("ERROR: could not load %s" % LEVEL_PATH)
        return
    for folder in (ASSET_DIR, ASSET_DIR + "/Textures", ASSET_DIR + "/Meshes"):
        if not eal.does_directory_exist(folder):
            eal.make_directory(folder)
    textures = _step("import leaf atlases", _textures, info)
    mats = _step("vegetation materials", _materials, textures) if textures else None
    meshes = _step("import %d models (Nanite)" % len(info["models"]), _meshes, info, mats) if mats else None
    if meshes:
        total = _step("place instances", _place, info, meshes)
        if total:
            log.append("      %d part instances; %s" % (total, json.dumps(info.get("counts", {}))))
    _step("save %s" % LEVEL_PATH, _save_level)


if eal is not None:
    try:
        main()
    except Exception as exc:
        log.append("ERROR: %r" % (exc,))
    with open(LOG_PATH, "w") as fh:
        fh.write("\n".join(log) + "\n")
