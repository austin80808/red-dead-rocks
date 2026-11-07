"""Make Lvl_RedRocks walkable: import the venue and park meshes, give every
walkable surface collision, put the player on the stage, and show the credits.

Run after import_scan.py (rebuild_level.py runs it last). Headless:
    UnrealEditor-Cmd RedRocks.uproject -run=pythonscript \
        -script=unreal/setup_gameplay.py -unattended -nosplash -stdout
or from the editor console:
    py "/Users/austin/dev/redrocks-ue/unreal/setup_gameplay.py"

What it does, each step logged to /tmp/redrocks_gameplay.log as "ok" or "FAIL":
  1. Imports the stage (Stage, the concrete, and Stage_Steel, Stage_Roof,
     Stage_Fascia and Stage_Towers, one material each), Roads, Paths, Parking
     and the rest of NEW_MESHES from out/ (skips any that are missing) and
     places them at the origin: they are already in the terrain's frame, in
     centimetres. Re-running replaces the actors, it never duplicates.
     Denver (downtown's skyline, 20 km out) gets M_DistantCity, no collision,
     no shadow, and is never culled by distance (FAR_AWAY_SETTINGS).
  2. Gives every OBJ actor scale (1, 1, 1): the OBJ importer negates Y, which
     is already the terrain -> level map (level Y = -north, level_frame.py).
     A Y scale of -1 here once built a mirror image of Red Rocks.
  3. Sets "use complex as simple" collision on the terrain, rocks, benches,
     stage and park meshes, so the player walks on the real triangles instead
     of falling through or tripping on a convex hull.
  3. Gives roads, paths, parking and the stage their materials (asphalt,
     gravel, lot asphalt; the stage's concrete, oxide-red steel, maroon roof,
     dark fascia and the towers' flagstone; built by setup_materials.py).
  4. Moves the PlayerStart onto the stage, 5 m behind the downstage edge,
     facing up the bowl, at the height of manifest.json's spawn block (the
     higher of the deck and the terrain under the player).
  5. Sets the level's game mode to the FirstPerson template's, so Play spawns a
     walking first-person character (WASD, mouse, space to jump).
  6. Recaptures the sky light (intensity 1.0, physical).
  7. When the ml0w scan is present: the scan chunks get complex collision and
     the extruded rocks are hidden in game with no collision (#16), so what
     blocks the player is the rock they see.
  8. Places the attribution as text in the world, beside the spawn, for the
     editor (CC BY 4.0 ml0w scan, ODbL OpenStreetMap). It is hidden in game:
     in play the credits are the press-C overlay (setup_credits.py), whose hint
     shows at the start, so no sign floats over the bowl in every view.
"""

import json
import math
import os
import sys

import unreal

_HERE = os.path.dirname(os.path.abspath(globals().get("__file__") or "/Users/austin/dev/redrocks-ue/unreal/x"))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import level_frame  # noqa: E402  (beside this script: terrain frame -> level frame)

# The redrocks build output; REDROCKS_OUT_DIR points a test copy at another build.
OUT_DIR = os.environ.get("REDROCKS_OUT_DIR", "/Users/austin/dev/redrocks-ue/out")
ASSET_DIR = "/Game/RedRocks"
MAT_DIR = "/Game/RedRocks/Materials"
LEVEL_PATH = "/Game/RedRocks/Lvl_RedRocks"
GAME_MODE = "/Game/FirstPerson/Blueprints/BP_FirstPersonGameMode"
LOG_PATH = os.path.join(os.environ.get("REDROCKS_LOG_DIR", "/tmp"), "redrocks_gameplay.log")

# The stage's OBJs (redrocks.stage.STAGE_MESHES): the concrete, then one per material.
STAGE_MESHES = ["Stage", "Stage_Steel", "Stage_Roof", "Stage_Fascia", "Stage_Towers"]
# Meshes placed by this script (all in the terrain frame, origin, no rotation).
NEW_MESHES = STAGE_MESHES + ["Roads", "Paths", "Parking", "Buildings", "Walls", "Rocks", "Water",
                             "Bench_Planks", "Bench_Brackets", "Handrail", "Booth", "Stairs", "Denver"]
# Seen but not stood on: the planks and brackets sit on the terraces' riser
# edges, a few cm over the treads; colliding with them would make every row a
# step too tall to walk up (redrocks.seating). Denver is 20 km away.
NO_COLLIDE = ["Bench_Planks", "Bench_Brackets", "Denver"]
# Seen only from far off: downtown Denver's skyline, ~20.5 km east-north-east
# of the stage (redrocks.denver). See FAR_AWAY_SETTINGS.
FAR_AWAY = ["Denver"]
# Every mesh the player should be able to stand on or bump into.
COLLIDE = ["RedRocks_Terrain", "Creation_Rock", "Ship_Rock", "Stage_Rock", "Benches"] + [
    m for m in NEW_MESHES if m not in NO_COLLIDE]

# Plain materials: name -> (sRGB hex, roughness), and which mesh gets which.
# setup_materials.py builds these as procedural materials (asphalt, gravel,
# concrete, parking lot, buildings, sandstone) and runs first; the flat colours
# here are only a fallback for running this script on its own.
PLAIN = {
    "M_Asphalt": ("4a4845", 0.85),
    "M_Lot": ("57544f", 0.85),
    "M_Gravel": ("9b6a4b", 0.95),
    "M_Concrete": ("8f847b", 0.85),  # weathered warm-grey concrete (stage_front photo); a8a196 read near-white in sun
    "M_Buildings": ("a66848", 0.85),
    "M_Sandstone": ("a2482a", 0.88),
    "M_Water": ("1d4b52", 0.04),  # Bear Creek: dark green-blue, glassy
    "M_BenchWood": ("a8916f", 0.8),
    "M_Steel": ("2e2d2b", 0.6),
    "M_Roof": ("7a2a22", 0.55),
    "M_Benches": ("94482c", 0.9),
    # Downtown Denver at 20 km: pale grey-beige concrete and glass, sRGB
    # (170, 170, 165); the sky atmosphere's haze does the rest.
    "M_DistantCity": ("aaaaa5", 0.6),
    "M_StageSteel": ("8e3f28", 0.65),  # the stage's painted oxide-red steel
}
MESH_MATERIAL = {
    "Roads": "M_Asphalt",
    "Paths": "M_Gravel",
    "Parking": "M_Lot",
    "Stage": "M_Concrete",
    "Stage_Steel": "M_StageSteel",  # columns, Y-arms, trusses, purlins, side roofs' frames
    "Stage_Roof": "M_Roof",  # the roof slab and side roofs: maroon painted metal, like the booth's
    "Stage_Fascia": "M_Steel",  # the roof's dark front band: near-black painted steel
    "Stage_Towers": "M_Benches",  # the towers' stacked red flagstone, like the risers
    "Buildings": "M_Buildings",
    "Walls": "M_Concrete",
    "Rocks": "MI_Sandstone_Rock",
    "Water": "M_Water",
    "Bench_Planks": "M_BenchWood",
    "Bench_Brackets": "M_Steel",
    "Handrail": "M_Steel",
    "Booth": "M_Roof",
    "Stairs": "M_Benches",  # the seating's flagstone risers and sandstone treads
    "Denver": "M_DistantCity",
}

SPAWN_BACK_M = 5.0  # how far behind the downstage edge the player starts
# The sky light stays physical (1.0, as setup_lighting.py sets it): with Lumen
# GI on (configure_level.py) shade gets the sky and the red ground's bounce,
# so first person no longer needs the old 4x boost. This step only recaptures.
SKY_LIGHT_INTENSITY = 1.0
EYE_LIFT_CM = 130.0  # PlayerStart above the ground; capsule half-height 96 + gap

SCAN_CREDIT = (
    'Red Rocks scan: "Red Rocks, Colorado" by ml0w, CC BY 4.0\n'
    "sketchfab.com/3d-models/red-rocks-colorado-9dec580ed1254c13be687a93bae55ae1"
)

log = []


def _step(label, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        log.append("ok    %s" % label)
        return result
    except Exception as exc:  # keep going; the runner stops on FAIL lines
        log.append("FAIL  %s: %r" % (label, exc))
        return None


def _actors():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()


def _by_label(label):
    return [a for a in _actors() if a.get_actor_label() == label]


def _import_mesh(name):
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", os.path.join(OUT_DIR, name + ".obj"))
    task.set_editor_property("destination_path", ASSET_DIR)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", True)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    paths = list(task.get_editor_property("imported_object_paths"))
    if not paths:
        raise RuntimeError("import produced nothing for %s" % name)
    mesh = unreal.EditorAssetLibrary.load_asset("%s/%s" % (ASSET_DIR, name))
    if isinstance(mesh, unreal.StaticMesh):
        settings = mesh.get_editor_property("nanite_settings")
        settings.set_editor_property("enabled", False)
        mesh.set_editor_property("nanite_settings", settings)
        unreal.EditorAssetLibrary.save_asset("%s/%s" % (ASSET_DIR, name))
    return paths


def _place_at_origin(name):
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for old in _by_label(name):
        actor_subsystem.destroy_actor(old)
    mesh = unreal.EditorAssetLibrary.load_asset("%s/%s" % (ASSET_DIR, name))
    if mesh is None:
        raise RuntimeError("missing asset %s/%s" % (ASSET_DIR, name))
    # spawn_actor_from_object no-ops in a headless commandlet; spawn by class.
    actor = actor_subsystem.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(0.0, 0.0, 0.0))
    actor.static_mesh_component.set_static_mesh(mesh)
    actor.set_actor_label(name)
    actor.set_mobility(unreal.ComponentMobility.STATIC)
    return actor


# Unreal's Interchange OBJ importer negates Y (an OBJ vertex at north = +y
# lands at y = -north), which is exactly the terrain -> level map: Unreal is
# left-handed, so north has to go on -Y or the level is a mirror image of Red
# Rocks (level_frame.py). Every OBJ actor therefore keeps scale (1, 1, 1); this
# step resets any flip an older build left on it.
OBJ_SCALE = level_frame.OBJ_ACTOR_SCALE


def _obj_scale(name):
    actors = _by_label(name)
    if not actors:
        raise RuntimeError("no actor labelled %s" % name)
    for actor in actors:
        actor.set_actor_scale3d(unreal.Vector(*OBJ_SCALE))
    return len(actors)


SCAN_PREFIX = "Scan_ml0w"


def _scan_collision():
    """The scan's rocks are what the player sees, so they are what blocks.

    Every scan chunk gets complex-as-simple collision (its Nanite fallback
    mesh), and the hidden extruded rocks lose theirs, so there are no invisible
    walls where the extruded outlines differ from the scanned rock. The terrain
    under the scan is blended 0.4 m below it (scan_blend.py), so walking onto
    the scan is a small step up, not a ledge.
    """
    chunks = [a for a in _actors() if a.get_actor_label().startswith(SCAN_PREFIX)
              and isinstance(a, unreal.StaticMeshActor)]
    meshes = {}
    for actor in chunks:
        mesh = actor.static_mesh_component.static_mesh
        if mesh is not None:
            meshes[mesh.get_path_name()] = mesh
    for path, mesh in meshes.items():
        body = mesh.get_editor_property("body_setup")
        if body is None:
            raise RuntimeError("%s has no body setup" % path)
        body.set_editor_property("collision_trace_flag", unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE)
        mesh.set_editor_property("body_setup", body)
        unreal.EditorAssetLibrary.save_asset(path.split(".")[0])
    for actor in chunks:
        actor.set_actor_enable_collision(True)
        actor.static_mesh_component.set_collision_profile_name("BlockAll")
    if chunks:
        for name in EXTRUDED_ROCKS:
            for actor in _by_label(name):
                actor.static_mesh_component.set_collision_profile_name("NoCollision")
                actor.set_actor_enable_collision(False)
    return "%d chunks, %d meshes" % (len(chunks), len(meshes))


EXTRUDED_ROCKS = ["Creation_Rock", "Ship_Rock", "Stage_Rock"]


def _hide_extruded_rocks():
    """Issue #16: where the scan has the real rocks, hide the extruded ones in
    game (_scan_collision also takes their collision away)."""
    scan = [a for a in _actors() if a.get_actor_label().startswith(SCAN_PREFIX)]
    if not scan:
        return "no scan in the level; extruded rocks stay visible"
    hidden = 0
    for name in EXTRUDED_ROCKS:
        for actor in _by_label(name):
            actor.set_actor_hidden_in_game(True)
            hidden += 1
    return hidden


def _save_level():
    """Save and fail loudly: a read-only .umap (Git LFS 'lockable' files are
    read-only until locked) makes save_current_level return False silently."""
    if not unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level():
        raise RuntimeError(
            "could not save %s; is Content/RedRocks/Lvl_RedRocks.umap read-only "
            "(git lfs lock it, or close the editor holding it)?" % LEVEL_PATH
        )


def _complex_collision(name):
    path = "%s/%s" % (ASSET_DIR, name)
    mesh = unreal.EditorAssetLibrary.load_asset(path)
    if not isinstance(mesh, unreal.StaticMesh):
        raise RuntimeError("no static mesh at %s" % path)
    body = mesh.get_editor_property("body_setup")
    if body is None:
        raise RuntimeError("%s has no body setup" % path)
    body.set_editor_property("collision_trace_flag", unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE)
    mesh.set_editor_property("body_setup", body)
    unreal.EditorAssetLibrary.save_asset(path)
    for actor in _by_label(name):
        actor.static_mesh_component.set_collision_profile_name("BlockAll")
    return path


def _no_collision(name):
    for actor in _by_label(name):
        actor.static_mesh_component.set_collision_profile_name("NoCollision")
        actor.set_actor_enable_collision(False)
    return name


# A mesh seen only from tens of km: no shadow (nothing near it to shade, and a
# 3 km-wide caster bloats the virtual shadow maps), never culled by distance
# (MaxDrawDistance 0, no cull-distance volume), and out of the mesh distance
# fields Lumen traces within a few hundred metres. The level has no far clip:
# the project keeps Unreal's infinite far plane (its DefaultEngine.ini sets
# only NearClipPlane), and the terrain skirt already draws out to 120 km.
FAR_AWAY_SETTINGS = {
    "cast_shadow": False,
    "ld_max_draw_distance": 0.0,
    "never_distance_cull": True,
    "allow_cull_distance_volume": False,
    "affect_distance_field_lighting": False,
}


def _far_away(name):
    actors = _by_label(name)
    if not actors:
        raise RuntimeError("no actor labelled %s" % name)
    for actor in actors:
        comp = actor.static_mesh_component
        for prop, value in FAR_AWAY_SETTINGS.items():
            comp.set_editor_property(prop, value)
    return len(actors)


def _srgb_to_linear(hex_color):
    out = []
    for i in (0, 2, 4):
        c = int(hex_color[i:i + 2], 16) / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return out


def _plain_material(name, hex_color, roughness):
    path = "%s/%s" % (MAT_DIR, name)
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        return unreal.EditorAssetLibrary.load_asset(path)
    if not unreal.EditorAssetLibrary.does_directory_exist(MAT_DIR):
        unreal.EditorAssetLibrary.make_directory(MAT_DIR)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    mat = tools.create_asset(name, MAT_DIR, unreal.Material, unreal.MaterialFactoryNew())
    lib = unreal.MaterialEditingLibrary
    r, g, b = _srgb_to_linear(hex_color)
    color = lib.create_material_expression(mat, unreal.MaterialExpressionConstant3Vector, -400, 0)
    color.set_editor_property("constant", unreal.LinearColor(r, g, b, 1.0))
    lib.connect_material_property(color, "", unreal.MaterialProperty.MP_BASE_COLOR)
    rough = lib.create_material_expression(mat, unreal.MaterialExpressionConstant, -400, 200)
    rough.set_editor_property("r", roughness)
    lib.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)
    lib.recompile_material(mat)
    unreal.EditorAssetLibrary.save_asset(path)
    return mat


def _assign_material(mesh_name, mat):
    path = "%s/%s" % (ASSET_DIR, mesh_name)
    mesh = unreal.EditorAssetLibrary.load_asset(path)
    if mesh is None:
        raise RuntimeError("missing %s" % path)
    for i in range(mesh.get_num_sections(0) or 1):
        mesh.set_material(i, mat)
    unreal.EditorAssetLibrary.save_asset(path)


def spawn_transform(stage, terrain_min_m):
    """PlayerStart location (cm) and yaw (deg): on the deck, facing up the bowl.

    Plain arithmetic so it can be tested without Unreal.
    """
    ax, ay = stage["axis_east_north"]
    norm = math.hypot(ax, ay) or 1.0
    ax, ay = ax / norm, ay / norm
    # The deck lies on the stage side of the downstage edge, i.e. against the
    # axis that runs from row 1 up to the top row.
    x, y = level_frame.to_level_cm(stage["front_edge_east_m"] - SPAWN_BACK_M * ax,
                                   stage["front_edge_north_m"] - SPAWN_BACK_M * ay)
    z = (stage["deck_top_elevation_m"] - terrain_min_m) * 100.0 + EYE_LIFT_CM
    yaw = level_frame.level_yaw(math.degrees(math.atan2(ay, ax)))
    return (x, y, z), yaw


def spawn_from_manifest(manifest):
    """PlayerStart (x, y, z) in cm and yaw, preferring the build's spawn block.

    The spawn block's ground height is the higher of the deck and the terrain
    under the capsule; without it the deck top is used, which can bury the
    start in the hillside (Unreal then drops the pawn at the origin).
    """
    terrain_min = manifest["terrain_mesh"]["min_elevation_m"]
    spawn = manifest.get("spawn")
    if spawn:
        x, y = level_frame.to_level_cm(spawn["east_m"], spawn["north_m"])
        z = (spawn["ground_elevation_m"] - terrain_min) * 100.0 + EYE_LIFT_CM
        return (x, y, z), level_frame.level_yaw(spawn["yaw_deg"])
    stage = manifest.get("stage")
    if not stage:
        raise RuntimeError("manifest has no spawn or stage block; run redrocks build")
    return spawn_transform(stage, terrain_min)


def _player_start(manifest):
    (x, y, z), yaw = spawn_from_manifest(manifest)
    starts = [a for a in _actors() if isinstance(a, unreal.PlayerStart)]
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    if not starts:
        starts = [actor_subsystem.spawn_actor_from_class(unreal.PlayerStart, unreal.Vector(x, y, z))]
    for extra in starts[1:]:
        actor_subsystem.destroy_actor(extra)
    start = starts[0]
    start.set_actor_label("PlayerStart")
    start.set_actor_location(unreal.Vector(x, y, z), False, False)
    start.set_actor_rotation(unreal.Rotator(pitch=0.0, yaw=yaw, roll=0.0), False)
    return (x, y, z, yaw)


def _game_mode():
    cls = unreal.EditorAssetLibrary.load_blueprint_class(GAME_MODE)
    if cls is None:
        raise RuntimeError("no game mode at %s" % GAME_MODE)
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    settings = world.get_world_settings()
    settings.set_editor_property("default_game_mode", cls)
    return GAME_MODE


def _first_person_light():
    lights = [a for a in _actors() if isinstance(a, unreal.SkyLight)]
    if not lights:
        raise RuntimeError("no SkyLight in the level; run setup_lighting.py")
    for light in lights:
        comp = light.get_component_by_class(unreal.SkyLightComponent)
        comp.set_editor_property("intensity", SKY_LIGHT_INTENSITY)
        comp.recapture_sky()
    return len(lights)


def credits_text(manifest):
    """The in-game credit lines: the scan first, then the manifest's list."""
    lines = [SCAN_CREDIT]
    lines.extend(manifest.get("attribution", []))
    lines.append("Rock models: Poly Haven, CC0")
    return "\n".join(lines)


def _credits(manifest, spawn):
    x, y, z, yaw = spawn
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for old in _by_label("Credits"):
        actor_subsystem.destroy_actor(old)
    # 6 m ahead of the player and 4 m to their left, at eye height, turned to
    # face back at the spawn (a TextRender reads from its +X side).
    fx, fy = level_frame.forward(yaw)
    lx, ly = level_frame.left_of(yaw)
    loc = unreal.Vector(x + 600 * fx + 400 * lx, y + 600 * fy + 400 * ly, z + 150)
    actor = actor_subsystem.spawn_actor_from_class(unreal.TextRenderActor, loc)
    actor.set_actor_label("Credits")
    actor.set_actor_rotation(unreal.Rotator(pitch=0.0, yaw=yaw + 180.0, roll=0.0), False)
    comp = actor.text_render
    comp.set_text(credits_text(manifest))
    comp.set_editor_property("world_size", 18.0)
    comp.set_editor_property("horizontal_alignment", unreal.HorizTextAligment.EHTA_CENTER)
    comp.set_editor_property("text_render_color", unreal.Color(r=255, g=244, b=225, a=255))
    actor.tags = [unreal.Name("credits")]
    # Floating text over the rows was the blockout's credit; play shows the
    # press-C overlay instead, so keep this for the editor only.
    actor.set_actor_hidden_in_game(True)
    return loc


def main():
    with open(os.path.join(OUT_DIR, "manifest.json")) as fh:
        manifest = json.load(fh)

    unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).load_level(LEVEL_PATH)

    present = [n for n in NEW_MESHES if os.path.exists(os.path.join(OUT_DIR, n + ".obj"))]
    for name in NEW_MESHES:
        if name not in present:
            log.append("skip  %s: %s.obj not in out/" % (name, name))
    for name in present:
        if _step("import %s" % name, _import_mesh, name):
            _step("place %s" % name, _place_at_origin, name)

    for name in COLLIDE:
        if _by_label(name):
            _step("scale %s" % name, _obj_scale, name)
    for name in NO_COLLIDE:
        if _by_label(name):
            _step("scale %s" % name, _obj_scale, name)
            _step("no collision %s" % name, _no_collision, name)
    for name in FAR_AWAY:
        if _by_label(name):
            _step("no shadow, never culled: %s" % name, _far_away, name)
    for name in COLLIDE:
        if unreal.EditorAssetLibrary.does_asset_exist("%s/%s" % (ASSET_DIR, name)):
            _step("collision %s" % name, _complex_collision, name)
        else:
            log.append("skip  collision %s: not imported" % name)

    mats = {}
    for name, (color, rough) in PLAIN.items():
        mats[name] = _step("material %s" % name, _plain_material, name, color, rough)
    rock_inst_path = "%s/MI_Sandstone_Rock" % MAT_DIR
    if unreal.EditorAssetLibrary.does_asset_exist(rock_inst_path):
        mats["MI_Sandstone_Rock"] = unreal.EditorAssetLibrary.load_asset(rock_inst_path)
    else:
        mats["MI_Sandstone_Rock"] = mats.get("M_Sandstone")
    for mesh_name, mat_name in MESH_MATERIAL.items():
        if mesh_name in present and mats.get(mat_name):
            _step("assign %s -> %s" % (mat_name, mesh_name), _assign_material, mesh_name, mats[mat_name])

    spawn = _step("player start on stage", _player_start, manifest)
    if spawn:
        log.append("      spawn x=%.0f y=%.0f z=%.0f yaw=%.1f" % spawn)
        _step("credits", _credits, manifest, spawn)
    _step("game mode FirstPerson", _game_mode)
    _step("sky light %.1f for first person" % SKY_LIGHT_INTENSITY, _first_person_light)

    _step("scan rocks collide; extruded rocks do not", _scan_collision)
    _step("hide extruded rocks in game (collision kept)", _hide_extruded_rocks)
    _step("save %s" % LEVEL_PATH, _save_level)


if hasattr(unreal, "EditorAssetLibrary"):
    try:
        main()
    except Exception as exc:
        log.append("ERROR: %r" % (exc,))
    with open(LOG_PATH, "w") as fh:
        fh.write("\n".join(log) + "\n")
    for line in log:
        (unreal.log_error if line.startswith(("FAIL", "ERROR")) else unreal.log)(line)
