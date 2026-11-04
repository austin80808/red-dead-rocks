"""Let it snow: every Red Rocks material whitens where the ground faces up as
the MPC_Weather "Snow" parameter goes from 0 to 1 (ARRWeather drives it in game,
N toggles it).

Run after setup_materials.py, setup_vegetation.py and import_scan.py (their
materials are the ones it edits); rebuild_level.py runs it after
setup_credits.py. Headless:
    UnrealEditor-Cmd RedRocks.uproject -run=pythonscript \
        -script=unreal/setup_snow.py -unattended -nosplash -stdout

What it does, logged to /tmp/redrocks_snow.log as "ok" / "skip" / "FAIL":
  1. Creates /Game/RedRocks/Materials/MPC_Weather with one scalar, Snow (0),
     and M_Snowflake (unlit white) for the falling flakes.
  2. For each Material under /Game/RedRocks: takes whatever feeds its Base
     Color and Roughness and puts a lerp in between, toward packed snow
     (#e4e8ee, roughness 0.55), with alpha
         saturate(Snow * 1.4 - noise * 0.4) * saturate((normal.z - 0.35) * 2.5)
     so flat ground and bench tops go white first, rock faces stay red, and
     partial cover comes in patches. Material instances inherit it.
     A material that already has the snow lerp (desc "rr_snow") is skipped,
     so re-running is safe.
"""

import os

import unreal

GAME_DIR = "/Game/RedRocks"
MAT_DIR = GAME_DIR + "/Materials"
MPC_NAME = "MPC_Weather"
MPC_PATH = "%s/%s" % (MAT_DIR, MPC_NAME)
PARAM = "Snow"
MARK = "rr_snow"
SNOW_RGB = (0.78, 0.81, 0.86)  # #e4e8ee in linear
SNOW_ROUGHNESS = 0.55
LOG_PATH = os.path.join(os.environ.get("REDROCKS_LOG_DIR", "/tmp"), "redrocks_snow.log")

lib = unreal.MaterialEditingLibrary
log = []


def _step(label, fn, *args):
    try:
        result = fn(*args)
        log.append("ok    %s%s" % (label, "" if result is None else ": %s" % (result,)))
        return result
    except Exception as exc:
        log.append("FAIL  %s: %r" % (label, exc))
        return None


def _collection():
    if unreal.EditorAssetLibrary.does_asset_exist(MPC_PATH):
        mpc = unreal.EditorAssetLibrary.load_asset(MPC_PATH)
    else:
        tools = unreal.AssetToolsHelpers.get_asset_tools()
        mpc = tools.create_asset(MPC_NAME, MAT_DIR, unreal.MaterialParameterCollection,
                                 unreal.MaterialParameterCollectionFactoryNew())
    params = list(mpc.get_editor_property("scalar_parameters"))
    if not any(str(p.get_editor_property("parameter_name")) == PARAM for p in params):
        p = unreal.CollectionScalarParameter()
        p.set_editor_property("parameter_name", PARAM)
        p.set_editor_property("default_value", 0.0)
        params.append(p)
        mpc.set_editor_property("scalar_parameters", params)
    unreal.EditorAssetLibrary.save_asset(MPC_PATH)
    return mpc


class _Graph:
    def __init__(self, mat):
        self.mat = mat
        self.n = 0

    def node(self, cls_name, **props):
        self.n += 1
        node = lib.create_material_expression(self.mat, getattr(unreal, cls_name), 600 + 220 * (self.n % 6),
                                              -900 - 160 * (self.n // 6))
        for key, value in props.items():
            node.set_editor_property(key, value)
        return node

    def const(self, v):
        return self.node("MaterialExpressionConstant", r=float(v))

    def link(self, src, dst, pin, src_pin=""):
        if not lib.connect_material_expressions(src, src_pin, dst, pin):
            raise RuntimeError("connect failed -> %s" % pin)

    def link1(self, src, dst):
        for pin in ("", "Input"):
            try:
                if lib.connect_material_expressions(src, "", dst, pin):
                    return
            except Exception:
                pass
        raise RuntimeError("could not connect into %s" % dst)

    def bin(self, cls_name, a, b):
        n = self.node(cls_name)
        self.link(a if not isinstance(a, (int, float)) else self.const(a), n, "A")
        self.link(b if not isinstance(b, (int, float)) else self.const(b), n, "B")
        return n

    def sat(self, a):
        n = self.node("MaterialExpressionSaturate")
        self.link1(a, n)
        return n


def _coverage(g, mpc):
    snow = g.node("MaterialExpressionCollectionParameter", collection=mpc, parameter_name=PARAM)
    normal = g.node("MaterialExpressionVertexNormalWS")
    nz = g.node("MaterialExpressionComponentMask", r=False, g=False, b=True, a=False)
    g.link1(normal, nz)
    up = g.sat(g.bin("MaterialExpressionMultiply", g.bin("MaterialExpressionSubtract", nz, 0.35), 2.5))
    noise = g.node("MaterialExpressionNoise", scale=1.0 / 350.0, levels=2, quality=1,
                   noise_function=unreal.NoiseFunction.NOISEFUNCTION_SIMPLEX_TEX,
                   output_min=0.0, output_max=1.0, turbulence=False)
    g.link(g.node("MaterialExpressionWorldPosition"), noise, "World Position")
    patchy = g.sat(g.bin("MaterialExpressionSubtract", g.bin("MaterialExpressionMultiply", snow, 1.4),
                         g.bin("MaterialExpressionMultiply", noise, 0.4)))
    return g.bin("MaterialExpressionMultiply", patchy, up)


def _snow_material(mat, mpc):
    mp = unreal.MaterialProperty
    base = lib.get_material_property_input_node(mat, mp.MP_BASE_COLOR)
    if base is None:
        return "no base colour input"
    if str(base.get_editor_property("desc")) == MARK:
        return "already snowy"
    base_out = lib.get_material_property_input_node_output_name(mat, mp.MP_BASE_COLOR) or ""
    rough = lib.get_material_property_input_node(mat, mp.MP_ROUGHNESS)
    rough_out = lib.get_material_property_input_node_output_name(mat, mp.MP_ROUGHNESS) or ""

    g = _Graph(mat)
    alpha = _coverage(g, mpc)
    white = g.node("MaterialExpressionConstant3Vector",
                   constant=unreal.LinearColor(r=SNOW_RGB[0], g=SNOW_RGB[1], b=SNOW_RGB[2], a=1.0))
    lerp = g.node("MaterialExpressionLinearInterpolate", desc=MARK)
    g.link(base, lerp, "A", base_out)
    g.link(white, lerp, "B")
    g.link(alpha, lerp, "Alpha")
    lib.connect_material_property(lerp, "", mp.MP_BASE_COLOR)

    rlerp = g.node("MaterialExpressionLinearInterpolate", desc=MARK)
    if rough is not None:
        g.link(rough, rlerp, "A", rough_out)
    else:
        g.link(g.const(0.8), rlerp, "A")
    g.link(g.const(SNOW_ROUGHNESS), rlerp, "B")
    g.link(alpha, rlerp, "Alpha")
    lib.connect_material_property(rlerp, "", mp.MP_ROUGHNESS)

    lib.recompile_material(mat)
    unreal.EditorAssetLibrary.save_asset(mat.get_path_name().split(".")[0])
    return "snow added"


FLAKE_PATH = MAT_DIR + "/M_Snowflake"
FLAKE_NITS = 3500.0


def _snowflake():
    """M_Snowflake: unlit, softly glowing white, for ARRWeather's falling flakes."""
    if unreal.EditorAssetLibrary.does_asset_exist(FLAKE_PATH):
        unreal.EditorAssetLibrary.delete_asset(FLAKE_PATH)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    mat = tools.create_asset("M_Snowflake", MAT_DIR, unreal.Material, unreal.MaterialFactoryNew())
    mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    mat.set_editor_property("used_with_instanced_static_meshes", True)
    white = lib.create_material_expression(mat, unreal.MaterialExpressionConstant3Vector, -300, 0)
    # The level is physically exposed (EV100 11-14.2, configure_level.py), so an
    # emissive of ~1 reads as black; daylight-white snow needs thousands.
    white.set_editor_property("constant", unreal.LinearColor(r=FLAKE_NITS, g=FLAKE_NITS * 1.02, b=FLAKE_NITS * 1.06, a=1.0))
    lib.connect_material_property(white, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    lib.recompile_material(mat)
    unreal.EditorAssetLibrary.save_asset(FLAKE_PATH)
    return FLAKE_PATH


def main():
    _step("snowflake material", _snowflake)
    mpc = _step("collection %s" % MPC_PATH, _collection)
    if mpc is None:
        log.append("ERROR: no MPC_Weather")
        return
    paths = unreal.EditorAssetLibrary.list_assets(GAME_DIR, recursive=True, include_folder=False)
    count = 0
    for path in sorted(paths):
        name = path.split("/")[-1].split(".")[0]
        if not name.startswith("M_") or name == "M_Snowflake":
            continue
        mat = unreal.EditorAssetLibrary.load_asset(path.split(".")[0])
        if not isinstance(mat, unreal.Material):
            continue
        if _step("snow %s" % name, _snow_material, mat, mpc) is not None:
            count += 1
    log.append("      %d materials" % count)


if hasattr(unreal, "EditorAssetLibrary"):
    try:
        main()
    except Exception as exc:
        log.append("ERROR: %r" % (exc,))
    with open(LOG_PATH, "w") as fh:
        fh.write("\n".join(log) + "\n")
    for line in log:
        (unreal.log_error if line.startswith(("FAIL", "ERROR")) else unreal.log)(line)
