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

Edge blend (issue #17): the material is masked and dithers the scan out toward
its footprint edge, by a plan-view fade mask (assets/scan/scan_fade.png, 0 at
the edge, 1 from 30 m inside) looked up by world X/Y. tools/scan_heights.py
writes the mask and its placement (scan_fade.json) from the same footprint the
terrain blend in redrocks.scan_blend uses, so where the scan thins out the
terrain underneath already matches it. The material is rebuilt on every run so
a new mask or fade takes effect; re-run tools/scan_heights.py whenever the FIT_*
constants below change.
"""

import json
import math
import os

import unreal


def _repo_dir():
    """The checkout these scripts belong to: env override, then this file's repo."""
    env = os.environ.get("REDROCKS_REPO")
    if env:
        return env
    here = globals().get("__file__")
    if here:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(here)))
        if os.path.isdir(os.path.join(repo, "assets", "scan")):
            return repo
    return "/Users/austin/dev/redrocks-ue"


REPO_DIR = _repo_dir()
SCAN_DIR = os.path.join(REPO_DIR, "assets", "scan", "usdz_extract")
SCAN_FILE = os.path.join(SCAN_DIR, "scene.usdc")
TEX_DIR = os.path.join(SCAN_DIR, "0")
ASSET_DIR = "/Game/RedRocks/Scan"
LEVEL_PATH = "/Game/RedRocks/Lvl_RedRocks"
ACTOR_LABEL = "Scan_ml0w"
FADE_PNG = os.path.join(REPO_DIR, "assets", "scan", "scan_fade.png")
FADE_JSON = os.path.join(REPO_DIR, "assets", "scan", "scan_fade.json")
DITHER_FUNCTION = "/Engine/Functions/Engine_MaterialFunctions02/Utility/DitherTemporalAA"
# World-space noise that breaks the fade into patches. Scale is per cm, so
# 0.0008 gives blobs of roughly 12 m (the noise's unit cell), about the size of
# the scan's scrub and rock clusters; 4 levels add detail down to ~1.5 m.
EDGE_NOISE_SCALE = 0.0008
EDGE_NOISE_LEVELS = 4
# How sharply a patch ends: 4 spreads each patch border over +-0.125 of alpha,
# a metre or two of dither, so patches read as soft-edged clumps, not stipple.
EDGE_SHARPNESS = 4.0
# Tint toward the ground colour at the very edge (0..1). 0.85 rather than 1
# keeps a trace of the photo's light and dark so the last patches are not flat.
EDGE_TINT = 0.85
GROUND_MI = "/Game/RedRocks/Materials/MI_Sandstone_Ground"
ROCK_MI = "/Game/RedRocks/Materials/MI_Sandstone_Rock"
# _ground_colour() of the look pass (#41) ground, used only if it is missing:
# soil #7E4530 25 % toward dust #B27D58, 40 % grass #5F6236/#8E8058.
GROUND_COLOUR_FALLBACK = (0.238, 0.125, 0.055)
# Rock band mix of #A2482A and #C98A5E (40 % buff), used only if it is missing.
ROCK_COLOUR_FALLBACK = (0.450, 0.141, 0.059)
# Near-black albedo lift. The scan's baseColor has its 10th percentile at
# linear luminance 0.007 (baked shadow and holes in the photogrammetry), which
# reads as black under the look pass's lighting. Texels darker than DARK_LUM
# (0.025, about the 20th percentile) blend toward shaded sandstone, by up to
# DARK_LIFT at luminance 0. The target is the rock colour at DARK_SHADE of its
# brightness and DARK_DESAT toward grey, so lifted texels stay in the scan's
# shadow range and do not turn tree shade and asphalt rust-red (a first pass
# at 0.06 / 0.85 / 0.45 with no desaturation did exactly that).
DARK_LUM = 0.025
DARK_LIFT = 0.75
DARK_SHADE = 0.35
DARK_DESAT = 0.4
# Masked clip for the dithered fade: the engine default, which the
# DitherTemporalAA function is built around (alpha 0 clips every pixel).
FADE_CLIP = 0.3333
CREDIT = "Red Rocks scan: ml0w (CC BY 4.0)"

# ---- Placement, from tools/fit_scan.py --------------------------------------
# The USD carries no real-world units (Sketchfab normalised them), so scale, yaw
# and position come from tools/fit_scan.py: scale searched within +-8 % of the
# 0.516 m/unit measured in PR #33, then yaw/mirror/offset fitted to the 3DEP
# terrain heights plus landmarks (stage front edge, both rock summits). Frame:
# the OBJ/manifest terrain frame (X east, +Y north, cm), i.e. the level's world
# frame once the OBJ imports carry scale (1,-1,1). Re-run
# unreal/export_scan_points.py or pass the USD to tools/fit_scan.py to refresh.
# Residuals of this fit: median |dz| 4.4 m over 50k points; stage front edge
# 0.2 m off, Creation Rock summit 12.3 m, Ship Rock summit 5.2 m. The scale sits
# at the low edge of the search (0.475 m/unit), so rock-to-rock distances favour
# a smaller scale still; check by eye. (Old fit: 31.1814 / 292.00, which put the
# stage ~400 m north of the bowl.)
FIT_SCALE = 47.4751  # cm per scan unit
FIT_YAW_DEG = 187.34
FIT_MIRROR_X = -1
FIT_SCAN_CENTRE = (15.956, 21.755)  # scan units, removed before rotating
FIT_OFFSET_CM = (-2901.0, 4335.6, 18007.2)
# Re-import the USD only when asked; re-placing reuses the existing chunks.
# Changing any FIT_* value moves the scan: re-run tools/scan_heights.py too (it
# reads these constants) so the terrain blend and the fade mask follow.
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


def _texture(role, fname, force=False):
    """Import one scan texture unless it is already in the project (or force)."""
    asset = "%s/%s" % (ASSET_DIR, os.path.splitext(fname)[0])
    tex = None
    if not force and unreal.EditorAssetLibrary.does_asset_exist(asset):
        tex = unreal.EditorAssetLibrary.load_asset(asset)
    if tex is None:
        path = os.path.join(TEX_DIR, fname) if role != "fade" else FADE_PNG
        if not os.path.exists(path):
            log.append("MISSING texture %s" % path)
            return None
        paths = _import_task(path, ASSET_DIR)
        log.append("imported texture %s -> %s" % (fname, paths))
        if not paths:
            return None
        tex = unreal.EditorAssetLibrary.load_asset(paths[0].split(".")[0])
        if tex is None:
            return None
    if role == "normal":
        tex.set_editor_property(
            "compression_settings", unreal.TextureCompressionSettings.TC_NORMALMAP
        )
        tex.set_editor_property("srgb", False)
    elif role in ("occl", "fade"):
        tex.set_editor_property(
            "compression_settings", unreal.TextureCompressionSettings.TC_MASKS
        )
        tex.set_editor_property("srgb", False)
    if role == "fade":
        # Clamp so the mask's zero border extends past the scan; never stream
        # so a screenshot taken right after load uses the full-res mask.
        tex.set_editor_property("address_x", unreal.TextureAddress.TA_CLAMP)
        tex.set_editor_property("address_y", unreal.TextureAddress.TA_CLAMP)
        tex.set_editor_property("never_stream", True)
    _save(tex)
    return tex


def _save(asset):
    """Save a loaded asset; a failed save is an ERROR so rebuild_level.py stops."""
    if not unreal.EditorAssetLibrary.save_loaded_asset(asset, False):
        raise RuntimeError("could not save %s (read-only? chmod -R u+w Content)" % asset.get_path_name())


def _import_textures():
    """Import (or reuse) the JPEGs and the fade mask; return {role: Texture2D}."""
    files = {
        "base": "material0_baseColor.jpg",
        "normal": "material0_normal.jpg",
        "occl": "material0_metallicRoughness_occl.jpg",
    }
    out = {}
    for role, fname in files.items():
        tex = _texture(role, fname)
        if tex is not None:
            out[role] = tex
    # The fade mask is regenerated by tools/scan_heights.py, so always
    # re-import it (replace_existing keeps the asset and its references).
    tex = _texture("fade", "scan_fade.png", force=True)
    if tex is not None:
        out["fade"] = tex
    return out


def _fade_placement():
    with open(FADE_JSON) as fh:
        return json.load(fh)


def _mi_vectors(path, names):
    """{name: (r, g, b)} linear vector parameters of a material instance, or None."""
    lib = unreal.MaterialEditingLibrary
    if not unreal.EditorAssetLibrary.does_asset_exist(path):
        return None
    mi = unreal.EditorAssetLibrary.load_asset(path)
    try:
        out = {}
        for n in names:
            c = lib.get_material_instance_vector_parameter_value(mi, n)
            out[n] = (c.r, c.g, c.b)
        if not any(sum(v) > 0.0 for v in out.values()):
            return None
        return out
    except Exception as exc:
        log.append("could not read %s from %s: %r" % (names, path, exc))
        return None


def _mix(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def _ground_colour():
    """Linear RGB the terrain reads as, to tint the scan toward at its edge.

    From MI_Sandstone_Ground's own parameters (setup_materials.py runs before
    this step): its soil (SoilColor washed 25 % toward DustColor) and its grass
    (GrassColor/DryGrassColor halfway), mixed 60/40, which is about how much of
    the flat ground that graph covers with grass at GrassAmount 0.6. Falls back
    to GROUND_COLOUR_FALLBACK if the instance or its parameters are missing.
    """
    v = _mi_vectors(GROUND_MI, ("SoilColor", "DustColor", "GrassColor", "DryGrassColor"))
    if v is None:
        log.append("edge tint fallback: %s" % (GROUND_COLOUR_FALLBACK,))
        return GROUND_COLOUR_FALLBACK
    soil = _mix(v["SoilColor"], v["DustColor"], 0.25)
    grass = _mix(v["GrassColor"], v["DryGrassColor"], 0.5)
    rgb = _mix(soil, grass, 0.4)
    log.append("edge tint from %s: (%.3f, %.3f, %.3f) linear" % ((GROUND_MI,) + rgb))
    return rgb


def _rock_shade_colour():
    """Linear RGB that near-black scan texels are lifted toward: shaded sandstone.

    MI_Sandstone_Rock's BaseColor/BuffColor band mix (40 % buff), DARK_DESAT
    toward grey and at DARK_SHADE of its brightness, so a lifted texel still
    reads as rock in shadow, just warm rock rather than a black hole.
    """
    v = _mi_vectors(ROCK_MI, ("BaseColor", "BuffColor"))
    rock = _mix(v["BaseColor"], v["BuffColor"], 0.4) if v else ROCK_COLOUR_FALLBACK
    grey = 0.2126 * rock[0] + 0.7152 * rock[1] + 0.0722 * rock[2]
    rgb = tuple(DARK_SHADE * (c + (grey - c) * DARK_DESAT) for c in rock)
    log.append("dark-texel lift toward (%.3f, %.3f, %.3f) linear%s"
               % (rgb + ("" if v else " (fallback)",)))
    return rgb


def _build_material(textures):
    """M_Scan_ml0w: the scan's textures, broken up and dithered out at its edge.

    Opacity: the fade mask (1 inside, 0 at the footprint edge) is compared
    against world-space noise, so toward the edge the scan thins into ragged
    patches rather than a band, and DitherTemporalAA softens each patch's
    border. Colour: toward the edge the scan is tinted to the terrain's ground
    colour, so what remains of it matches what shows through.

    Rebuilt in place (expressions cleared) rather than deleted, so the chunks
    already using it keep their reference.
    """
    name = "M_Scan_ml0w"
    path = "%s/%s" % (ASSET_DIR, name)
    lib = unreal.MaterialEditingLibrary
    mat = None
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        mat = unreal.EditorAssetLibrary.load_asset(path)
    if mat is None:
        mat = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            name, ASSET_DIR, unreal.Material, unreal.MaterialFactoryNew()
        )
    else:
        lib.delete_all_material_expressions(mat)

    def node(cls, x, y, **props):
        n = lib.create_material_expression(mat, getattr(unreal, cls), x, y)
        for k, v in props.items():
            n.set_editor_property(k, v)
        return n

    def link(a, a_out, b, b_in):
        if not lib.connect_material_expressions(a, a_out, b, b_in):
            raise RuntimeError("could not connect %s.%s -> %s.%s" % (a.get_name(), a_out, b.get_name(), b_in))

    def sample(tex, x, y, sampler=None):
        n = node("MaterialExpressionTextureSample", x, y, texture=tex)
        if sampler is not None:
            n.set_editor_property("sampler_type", sampler)
        return n

    def const(v, x, y):
        return node("MaterialExpressionConstant", x, y, r=float(v))

    def op(cls, a, b, x, y, a_out="", b_out=""):
        n = node(cls, x, y)
        link(a, a_out, n, "A")
        link(b, b_out, n, "B")
        return n

    base = None
    if "base" in textures:
        # Lift near-black texels toward shaded sandstone:
        # col = lerp(tex, shade, saturate((DARK_LUM - lum) / DARK_LUM) * DARK_LIFT).
        tex = sample(textures["base"], -1400, 0)
        lum = node("MaterialExpressionDotProduct", -1200, -150)
        link(tex, "RGB", lum, "A")
        link(node("MaterialExpressionConstant3Vector", -1400, -200,
                  constant=unreal.LinearColor(r=0.2126, g=0.7152, b=0.0722, a=1.0)), "", lum, "B")
        dark = op("MaterialExpressionMultiply",
                  op("MaterialExpressionSubtract", const(DARK_LUM, -1200, -300), lum, -1000, -250),
                  const(DARK_LIFT / DARK_LUM, -1000, -350), -800, -250)
        dark01 = node("MaterialExpressionSaturate", -600, -250)
        link(dark, "", dark01, "")
        r, g_, b_ = _rock_shade_colour()
        shade = node("MaterialExpressionConstant3Vector", -800, -100,
                     constant=unreal.LinearColor(r=r, g=g_, b=b_, a=1.0))
        base = node("MaterialExpressionLinearInterpolate", -400, 0)
        link(tex, "RGB", base, "A")
        link(shade, "", base, "B")
        link(dark01, "", base, "Alpha")
    if "normal" in textures:
        lib.connect_material_property(
            sample(textures["normal"], -600, 300, unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL),
            "RGB",
            unreal.MaterialProperty.MP_NORMAL,
        )
    if "occl" in textures:
        lib.connect_material_property(
            sample(textures["occl"], -600, 600, unreal.MaterialSamplerType.SAMPLERTYPE_MASKS),
            "R",
            unreal.MaterialProperty.MP_AMBIENT_OCCLUSION,
        )
    lib.connect_material_property(const(0.9, -600, 900), "", unreal.MaterialProperty.MP_ROUGHNESS)

    if "fade" not in textures:
        log.append("ERROR: no fade mask (%s); run tools/scan_heights.py" % FADE_PNG)
        if base is not None:
            lib.connect_material_property(base, "", unreal.MaterialProperty.MP_BASE_COLOR)
        mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_OPAQUE)
    else:
        # Fade alpha a: uv = (world XY - (x0, y1)) * (1/width, -1/height); the
        # mask's row 0 is its north edge and level Y is north.
        place = _fade_placement()
        wp = node("MaterialExpressionWorldPosition", -2200, 1200)
        xy = node("MaterialExpressionComponentMask", -2000, 1200, r=True, g=True, b=False, a=False)
        link(wp, "", xy, "")
        origin = node("MaterialExpressionConstant2Vector", -2000, 1350,
                      r=float(place["x0_cm"]), g=float(place["y1_cm"]))
        inv = node("MaterialExpressionConstant2Vector", -1800, 1350,
                   r=1.0 / float(place["width_cm"]), g=-1.0 / float(place["height_cm"]))
        uv = op("MaterialExpressionMultiply", op("MaterialExpressionSubtract", xy, origin, -1800, 1200),
                inv, -1600, 1200)
        fade = sample(textures["fade"], -1400, 1200, unreal.MaterialSamplerType.SAMPLERTYPE_MASKS)
        link(uv, "", fade, "UVs")

        # Patchy break-up: v = saturate((a - lerp(lo, hi, saturate(noise))) * sharp + 0.5).
        # lo = 0.5 / sharp clips every pixel at a = 0; hi = 1 - 2 * lo keeps
        # every pixel at a = 1 (with room to spare: multi-level noise can
        # overshoot its output range, and the mask is 8-bit).
        noise = node("MaterialExpressionNoise", -1600, 1500, scale=EDGE_NOISE_SCALE,
                     levels=EDGE_NOISE_LEVELS, output_min=0.0, output_max=1.0, quality=1)
        noise01 = node("MaterialExpressionSaturate", -1400, 1500)
        link(noise, "", noise01, "")
        lo = 0.5 / EDGE_SHARPNESS
        thresh = node("MaterialExpressionLinearInterpolate", -1200, 1500)
        link(const(lo, -1400, 1650), "", thresh, "A")
        link(const(1.0 - 2.0 * lo, -1400, 1750), "", thresh, "B")
        link(noise01, "", thresh, "Alpha")
        diff = op("MaterialExpressionSubtract", fade, thresh, -1000, 1300, a_out="R")
        sharp = op("MaterialExpressionMultiply", diff, const(EDGE_SHARPNESS, -1000, 1450), -800, 1300)
        biased = op("MaterialExpressionAdd", sharp, const(0.5, -800, 1450), -600, 1300)
        vis = node("MaterialExpressionSaturate", -400, 1300)
        link(biased, "", vis, "")

        # DitherTemporalAA is an engine material function (not an expression
        # class). Loaded by object path: a commandlet's asset registry has not
        # scanned /Engine content, so EditorAssetLibrary.load_asset misses it.
        func = unreal.load_object(None, DITHER_FUNCTION + "." + DITHER_FUNCTION.rsplit("/", 1)[1])
        if func is None:
            raise RuntimeError("missing engine material function %s" % DITHER_FUNCTION)
        dither = node("MaterialExpressionMaterialFunctionCall", -200, 1300)
        dither.set_editor_property("material_function", func)
        link(vis, "", dither, "Alpha Threshold")
        if not lib.connect_material_property(dither, "Result", unreal.MaterialProperty.MP_OPACITY_MASK):
            raise RuntimeError("could not wire DitherTemporalAA into the opacity mask")
        mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED)
        mat.set_editor_property("opacity_mask_clip_value", FADE_CLIP)

        # Edge tint: lerp(scan colour, ground colour, (1 - a) * EDGE_TINT).
        if base is not None:
            g = _ground_colour()
            ground = node("MaterialExpressionConstant3Vector", -600, -300,
                          constant=unreal.LinearColor(r=g[0], g=g[1], b=g[2], a=1.0))
            inv_a = node("MaterialExpressionOneMinus", -1000, 1050)
            link(fade, "R", inv_a, "")
            amount = op("MaterialExpressionMultiply", inv_a, const(EDGE_TINT, -1000, 950), -800, 1000)
            tinted = node("MaterialExpressionLinearInterpolate", -300, 0)
            link(base, "", tinted, "A")
            link(ground, "", tinted, "B")
            link(amount, "", tinted, "Alpha")
            lib.connect_material_property(tinted, "", unreal.MaterialProperty.MP_BASE_COLOR)
        log.append("fade mask %s at x0=%.0f y1=%.0f size %.0f x %.0f cm, fade %.0f m"
                   % (FADE_PNG, place["x0_cm"], place["y1_cm"], place["width_cm"],
                      place["height_cm"], place["fade_m"]))

    lib.recompile_material(mat)
    _save(mat)
    log.append("built material %s (%s)" % (path, mat.get_editor_property("blend_mode")))
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
            if not unreal.EditorAssetLibrary.save_asset(mesh_path, False):
                raise RuntimeError("could not save %s" % mesh_path)
    # Always rebuilt: cheap, and picks up a new fade mask.
    material = _build_material(_import_textures())

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

    if not level_subsystem.save_current_level():
        raise RuntimeError("could not save %s (read-only? chmod -R u+w Content)" % LEVEL_PATH)
    log.append("placed %s and saved %s" % (ACTOR_LABEL, LEVEL_PATH))


try:
    main()
except Exception as exc:
    log.append("ERROR: %r" % (exc,))

with open("/tmp/redrocks_scan.log", "w") as fh:
    fh.write("\n".join(log))
