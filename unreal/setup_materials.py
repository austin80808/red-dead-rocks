"""Create the Red Rocks sandstone material and apply it to the imported meshes.

Run headless (close the editor first):
    UnrealEditor-Cmd RedRocks.uproject -run=pythonscript \
        -script=unreal/setup_materials.py -unattended -nosplash -stdout
or inside a running editor console:
    py "/Users/austin/dev/redrocks-ue/unreal/setup_materials.py"

Creates under /Game/RedRocks/Materials:
    M_Sandstone         master material: Fountain Formation strata driven by
                        world position (bedding dips 30 deg west), rough, non-metal
    MI_Sandstone_Rock   instance for Creation_Rock and Ship_Rock
    MI_Sandstone_Ground paler, dustier, less banded, redder on steep slopes
    M_Benches           plain weathered wood (only if a "Benches" mesh exists)

Then sets slot 0 of the StaticMesh assets and saves them. Safe to re-run: the
master material graph is rebuilt, existing instances are reused. Log goes to
/tmp/redrocks_materials.log.
"""

import math

import unreal

GAME_DIR = "/Game/RedRocks"
MAT_DIR = GAME_DIR + "/Materials"
LOG_PATH = "/tmp/redrocks_materials.log"

log = []

lib = unreal.MaterialEditingLibrary
eal = unreal.EditorAssetLibrary


def _step(label, fn, *args, **kwargs):
    """Run one step, log success or failure, never raise."""
    try:
        result = fn(*args, **kwargs)
        log.append("ok    %s" % label)
        return result
    except Exception as exc:
        log.append("FAIL  %s: %r" % (label, exc))
        return None


def srgb_to_linear(hex_color):
    """'#B5532E' -> unreal.LinearColor (vector parameters are linear)."""
    h = hex_color.lstrip("#")
    out = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return unreal.LinearColor(r=out[0], g=out[1], b=out[2], a=1.0)


# Bedding dips 30 deg to the west (X east, Y north, Z up): the planes descend
# toward -X, i.e. z = x * tan30, whose upward unit normal is (-sin30, 0, cos30).
DIP = math.radians(30.0)
BED_NORMAL = (-math.sin(DIP), 0.0, math.cos(DIP))


# ---------------------------------------------------------------- graph helpers

def _node(mat, cls_name, x, y, **props):
    cls = getattr(unreal, cls_name)
    node = lib.create_material_expression(mat, cls, x, y)
    for key, value in props.items():
        node.set_editor_property(key, value)
    return node


def _link(src, dst, dst_pin, src_pin=""):
    """Connect src output to dst input; raises if Unreal refuses."""
    if not lib.connect_material_expressions(src, src_pin, dst, dst_pin):
        raise RuntimeError("connect failed: %s -> %s.%s" % (src, dst, dst_pin))


def _link_single(src, dst, src_pin=""):
    """Connect to a single-input node whose pin name may be '' or 'Input'."""
    for pin in ("", "Input"):
        try:
            if lib.connect_material_expressions(src, src_pin, dst, pin):
                return
        except Exception:
            pass
    raise RuntimeError("could not connect %s -> %s" % (src, dst))


def _scalar_param(mat, name, value, x, y):
    return _node(mat, "MaterialExpressionScalarParameter", x, y,
                 parameter_name=name, default_value=value)


def _vector_param(mat, name, hex_color, x, y):
    return _node(mat, "MaterialExpressionVectorParameter", x, y,
                 parameter_name=name, default_value=srgb_to_linear(hex_color))


def _mul(mat, a, b, x, y, const_b=None):
    n = _node(mat, "MaterialExpressionMultiply", x, y)
    _link(a, n, "A")
    if b is not None:
        _link(b, n, "B")
    else:
        n.set_editor_property("const_b", const_b)
    return n


def _add_const(mat, a, const_b, x, y):
    n = _node(mat, "MaterialExpressionAdd", x, y)
    _link(a, n, "A")
    n.set_editor_property("const_b", const_b)
    return n


def _lerp(mat, a, b, alpha, x, y):
    n = _node(mat, "MaterialExpressionLinearInterpolate", x, y)
    _link(a, n, "A")
    _link(b, n, "B")
    _link(alpha, n, "Alpha")
    return n


def _saturate(mat, a, x, y):
    n = _node(mat, "MaterialExpressionSaturate", x, y)
    _link_single(a, n)
    return n


def _band_mask(mat, band_coord, thickness_param, contrast, x, y):
    """0..1 sine mask: sin(2*pi * coord/thickness) * contrast * 0.5 + 0.5."""
    scaled = _node(mat, "MaterialExpressionDivide", x, y)
    _link(band_coord, scaled, "A")
    _link(thickness_param, scaled, "B")
    sine = _node(mat, "MaterialExpressionSine", x + 200, y, period=1.0)
    _link_single(scaled, sine)
    amp = _mul(mat, sine, contrast, x + 400, y)
    half = _mul(mat, amp, None, x + 600, y, const_b=0.5)
    shifted = _add_const(mat, half, 0.5, x + 800, y)
    return _saturate(mat, shifted, x + 1000, y)


def build_sandstone(mat):
    """Wire the whole sandstone graph into `mat`."""
    lib.delete_all_material_expressions(mat)

    # Parameters (instances override these).
    base = _vector_param(mat, "BaseColor", "#B5532E", -1400, -400)
    buff = _vector_param(mat, "BuffColor", "#D9A66F", -1400, -250)
    iron = _vector_param(mat, "IronColor", "#6E2A1A", -1400, -100)
    dust = _vector_param(mat, "DustColor", "#C98F6A", -1400, 50)
    thick_a = _scalar_param(mat, "BandThickness", 250.0, -1400, 250)  # cm
    contrast = _scalar_param(mat, "BandContrast", 1.0, -1400, 400)
    dust_amt = _scalar_param(mat, "DustAmount", 0.0, -1400, 550)
    slope_red = _scalar_param(mat, "SlopeRed", 0.0, -1400, 700)

    # Band coordinate: dot(WorldPosition, bedding normal), cm along the normal.
    wpos = _node(mat, "MaterialExpressionWorldPosition", -1400, 900)
    normal = _node(
        mat, "MaterialExpressionConstant3Vector", -1400, 1050,
        constant=unreal.LinearColor(
            r=BED_NORMAL[0], g=BED_NORMAL[1], b=BED_NORMAL[2], a=1.0),
    )
    coord = _node(mat, "MaterialExpressionDotProduct", -1100, 950)
    _link(wpos, coord, "A")
    _link(normal, coord, "B")

    # Fine band thickness = A / 3.7 so the two sine layers never line up.
    thick_b = _mul(mat, thick_a, None, -1100, 250, const_b=1.0 / 3.7)

    # Coarse bands: base red <-> pale buff. Fine bands: darker iron staining.
    coarse = _band_mask(mat, coord, thick_a, contrast, -900, -400)
    fine = _band_mask(mat, coord, thick_b, contrast, -900, 200)
    fine_soft = _mul(mat, fine, None, 300, 200, const_b=0.6)

    col = _lerp(mat, base, buff, coarse, 600, -300)
    col = _lerp(mat, col, iron, fine_soft, 800, -200)

    # Dust wash (the ground instance turns this up).
    col = _lerp(mat, col, dust, dust_amt, 1000, -100)

    # Slope: steep faces (normal.z small) go back toward the base red.
    vn = _node(mat, "MaterialExpressionVertexNormalWS", 400, 700)
    nz = _node(mat, "MaterialExpressionComponentMask", 600, 700,
               r=False, g=False, b=True, a=False)
    _link_single(vn, nz)
    flat = _saturate(mat, nz, 800, 700)
    steep = _node(mat, "MaterialExpressionOneMinus", 1000, 700)
    _link_single(flat, steep)
    steep_amt = _mul(mat, steep, slope_red, 1200, 700)
    col = _lerp(mat, col, base, _saturate(mat, steep_amt, 1400, 700), 1600, -100)

    rough = _node(mat, "MaterialExpressionConstant", 1600, 200, r=0.92)
    metal = _node(mat, "MaterialExpressionConstant", 1600, 350, r=0.0)

    mp = unreal.MaterialProperty
    lib.connect_material_property(col, "", mp.MP_BASE_COLOR)
    lib.connect_material_property(rough, "", mp.MP_ROUGHNESS)
    lib.connect_material_property(metal, "", mp.MP_METALLIC)


def build_plain(mat, hex_color, roughness):
    lib.delete_all_material_expressions(mat)
    color = _node(mat, "MaterialExpressionConstant3Vector", -400, 0,
                  constant=srgb_to_linear(hex_color))
    rough = _node(mat, "MaterialExpressionConstant", -400, 200, r=roughness)
    metal = _node(mat, "MaterialExpressionConstant", -400, 320, r=0.0)
    mp = unreal.MaterialProperty
    lib.connect_material_property(color, "", mp.MP_BASE_COLOR)
    lib.connect_material_property(rough, "", mp.MP_ROUGHNESS)
    lib.connect_material_property(metal, "", mp.MP_METALLIC)


# --------------------------------------------------------------- asset helpers

def _get_or_create(name, asset_class, factory):
    path = "%s/%s" % (MAT_DIR, name)
    if eal.does_asset_exist(path):
        log.append("reuse  %s" % path)
        return eal.load_asset(path)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    asset = tools.create_asset(name, MAT_DIR, asset_class, factory)
    log.append("create %s" % path)
    return asset


def make_material(name, builder, *args):
    mat = _get_or_create(name, unreal.Material, unreal.MaterialFactoryNew())
    if mat is None:
        raise RuntimeError("could not create %s" % name)
    _step("build graph %s" % name, builder, mat, *args)
    _step("recompile %s" % name, lib.recompile_material, mat)
    _step("save %s" % name, eal.save_loaded_asset, mat)
    return mat


def make_instance(name, parent, scalars, vectors):
    inst = _get_or_create(
        name, unreal.MaterialInstanceConstant,
        unreal.MaterialInstanceConstantFactoryNew(),
    )
    if inst is None:
        raise RuntimeError("could not create %s" % name)
    _step("parent %s" % name, inst.set_editor_property, "parent", parent)
    for key, value in scalars.items():
        _step("%s.%s" % (name, key),
              lib.set_material_instance_scalar_parameter_value, inst, key, value)
    for key, hex_color in vectors.items():
        _step("%s.%s" % (name, key),
              lib.set_material_instance_vector_parameter_value,
              inst, key, srgb_to_linear(hex_color))
    _step("update %s" % name, lib.update_material_instance, inst)
    _step("save %s" % name, eal.save_loaded_asset, inst)
    return inst


def apply_to_mesh(mesh_name, material):
    path = "%s/%s" % (GAME_DIR, mesh_name)
    if not eal.does_asset_exist(path):
        log.append("skip  %s (no such asset)" % path)
        return False
    mesh = eal.load_asset(path)
    if mesh is None or material is None:
        raise RuntimeError("could not load mesh or material for %s" % path)
    try:
        mesh.set_material(0, material)
    except Exception as exc:
        log.append("set_material failed (%r), trying static_materials" % (exc,))
        slots = list(mesh.get_editor_property("static_materials"))
        if not slots:
            raise RuntimeError("%s has no material slots" % path)
        slots[0].set_editor_property("material_interface", material)
        mesh.set_editor_property("static_materials", slots)
    _step("save mesh %s" % mesh_name, eal.save_loaded_asset, mesh)
    log.append("applied %s -> %s" % (material.get_name(), mesh_name))
    return True


def main():
    if not eal.does_directory_exist(MAT_DIR):
        eal.make_directory(MAT_DIR)

    sandstone = _step("M_Sandstone", make_material, "M_Sandstone", build_sandstone)

    rock = ground = None
    if sandstone is not None:
        rock = _step("MI_Sandstone_Rock", make_instance,
                     "MI_Sandstone_Rock", sandstone, {}, {})
        ground = _step(
            "MI_Sandstone_Ground", make_instance, "MI_Sandstone_Ground", sandstone,
            {"BandContrast": 0.35, "DustAmount": 0.45, "SlopeRed": 0.8,
             "BandThickness": 400.0},
            {"BaseColor": "#B96A45", "BuffColor": "#D8B088", "DustColor": "#CDA27E"},
        )

    for name in ("Creation_Rock", "Ship_Rock"):
        _step("apply %s" % name, apply_to_mesh, name, rock or sandstone)
    _step("apply RedRocks_Terrain", apply_to_mesh, "RedRocks_Terrain",
          ground or sandstone)

    if eal.does_asset_exist(GAME_DIR + "/Benches"):
        wood = _step("M_Benches", make_material, "M_Benches", build_plain,
                     "#6B5438", 0.85)
        _step("apply Benches", apply_to_mesh, "Benches", wood)
    else:
        log.append("skip  Benches (not in %s)" % GAME_DIR)

    log.append("done")


try:
    main()
except Exception as exc:
    log.append("ERROR: %r" % (exc,))

with open(LOG_PATH, "w") as fh:
    fh.write("\n".join(log))
