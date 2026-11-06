"""Hide the edge of the world with a terrain skirt.

Run after setup_materials.py (rebuild_level.py does). Headless:
    UnrealEditor-Cmd RedRocks.uproject -run=pythonscript \
        -script=unreal/setup_environment.py -unattended -nosplash -stdout

The DEM terrain is a 3.6 km square. Past it the SkyAtmosphere's planet ground
shows through as a flat dark-blue plane, so from the bowl and from the air the
world visibly ends. This script reads out/RedRocks_Terrain.obj, takes its edge
vertices, and grows rings outward from them to 120 km past the edge:

* the first ring is the terrain's edge itself, so the seam is exact;
* each ring outward smooths the edge heights more and blends toward a far
  profile: to the east the real plains by distance east (``plains_m``: about
  1.78 km at Morrison, 1.65 km at 12 km, 1.61 km at downtown Denver, settling
  at 1.58 km), the foothills to the west keep rising toward the Front Range;
* broad undulations break up the outline in the west; over the plains they
  are a few metres (the plains are flat, and Denver's skyline stands on them);
* every ring past the terrain's edge drops for Earth curvature and refraction,
  d^2 / 2R x (1 - k), the same as Denver.obj (redrocks.denver), so the far
  plains fall away below downtown's towers as they do from the top row;
* the height fog (setup_lighting.py) turns it into distant haze.

The plains are measured against the terrain's Z = 0 elevation, read from the
build's manifest.json (terrain_mesh.min_elevation_m).

It is written as an OBJ under the project's Saved/RedRocksGen/, imported to
/Game/RedRocks/Terrain_Skirt with MI_Sandstone_Ground, and placed like every
other OBJ actor, with no flip: the importer's Y negation is the level frame
(level_frame.py). It has no collision (no one can walk there) and casts no
shadows.

Trees and shrubs are setup_vegetation.py's (it replaced the sphere-clump
scrub this script used to scatter). The geometry functions are plain Python, tested offline.
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
TERRAIN_OBJ = os.path.join(OUT_DIR, "RedRocks_Terrain.obj")
ASSET_DIR = "/Game/RedRocks"
LEVEL_PATH = "/Game/RedRocks/Lvl_RedRocks"
SKIRT_NAME = "Terrain_Skirt"
SKIRT_MATERIAL = "/Game/RedRocks/Materials/MI_Sandstone_Ground"
LOG_PATH = os.path.join(os.environ.get("REDROCKS_LOG_DIR", "/tmp"), "redrocks_environment.log")

# Rings, as distance past the terrain edge (cm), how many edge samples each
# side of a vertex its height is averaged over, how far it has blended to the
# far profile, and the size of the undulations.
RING_OUT_CM = [0.0, 4000.0, 15000.0, 40000.0, 100000.0, 250000.0, 600000.0,
               1500000.0, 4000000.0, 12000000.0]
RING_SMOOTH = [0, 4, 12, 30, 70, 140, 260, 400, 500, 600]
RING_BLEND = [0.0, 0.04, 0.12, 0.25, 0.45, 0.65, 0.85, 1.0, 1.0, 1.0]
RING_WAVE_CM = [0.0, 0.0, 500.0, 1500.0, 4000.0, 8000.0, 14000.0, 18000.0, 12000.0, 0.0]
PLAINS_CM = -4000.0        # far north and south, ~1.70 km (it was the far east too)
FOOTHILLS_CM = 90000.0     # far west: foothills toward the Front Range, ~2.65 km

# The plains east of the hogback by distance east of the venue, in metres
# above sea level: PLAINS_FLOOR_M + PLAINS_RISE_M * exp(-east / PLAINS_FALL_KM),
# smooth and monotone: 1.78 km at Morrison (2 km), 1.65 km at 12 km, 1.61 km at
# downtown Denver (20.6 km), 1.60 km at 24 km, 1.58 km from 60 km on. The flat
# plain it replaces (PLAINS_CM, 1.70 km) stood 100 m over downtown's ground.
PLAINS_FLOOR_M = 1580.0
PLAINS_RISE_M = 246.7
PLAINS_FALL_KM = 9.53
# The terrain's Z = 0 (the manifest's terrain_mesh.min_elevation_m), for when
# the build's manifest is not at hand.
BASE_M = 1742.098
# How far into the eastern half (dx / distance: the sine of the angle from due
# north or south) the plains take over from the north-south plain.
EAST_RAMP = 0.34
# Undulations over the plains, as a share of RING_WAVE_CM: at most ~13 m.
EAST_WAVE = 0.07
# Earth curvature and refraction, as redrocks.denver: R and k.
EARTH_RADIUS_M = 6371000.0
REFRACTION_K = 0.13

log = []


def read_obj_vertices(path):
    """[(x, y, z)] from an OBJ's 'v' lines, in file units (cm here)."""
    verts = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                verts.append((float(x), float(y), float(z)))
    return verts


def grid_border(verts):
    """Edge vertices of a square grid mesh, counter-clockwise seen from above.

    Starts at the south-west corner and walks east along the south edge, then
    north, west and south again; corners appear once.
    """
    xs = sorted({round(v[0], 3) for v in verts})
    ys = sorted({round(v[1], 3) for v in verts})
    xi = {x: i for i, x in enumerate(xs)}
    yi = {y: i for i, y in enumerate(ys)}
    nx, ny = len(xs), len(ys)
    grid = {}
    for x, y, z in verts:
        grid[(xi[round(x, 3)], yi[round(y, 3)])] = (x, y, z)
    if len(grid) != nx * ny:
        raise ValueError("terrain is not a full %dx%d grid" % (nx, ny))
    ring = [grid[(i, 0)] for i in range(nx - 1)]
    ring += [grid[(nx - 1, j)] for j in range(ny - 1)]
    ring += [grid[(i, ny - 1)] for i in range(nx - 1, 0, -1)]
    ring += [grid[(0, j)] for j in range(ny - 1, 0, -1)]
    return ring


def _smoothed(values, half_window):
    if half_window <= 0:
        return list(values)
    n = len(values)
    out = []
    for i in range(n):
        acc = 0.0
        for k in range(-half_window, half_window + 1):
            acc += values[(i + k) % n]
        out.append(acc / (2 * half_window + 1))
    return out


def plains_m(east_m):
    """Elevation (m above sea level) of the plains ``east_m`` metres east of the venue."""
    return PLAINS_FLOOR_M + PLAINS_RISE_M * math.exp(-max(0.0, east_m) / 1000.0 / PLAINS_FALL_KM)


def _east_weight(dx, dy):
    """0 on the western half and due north or south, 1 well into the eastern half."""
    s = min(1.0, max(0.0, dx / (math.hypot(dx, dy) or 1.0)) / EAST_RAMP)
    return s * s * (3.0 - 2.0 * s)


def far_height(dx, dy, base_m=BASE_M):
    """Height (cm, terrain Z) the skirt settles to at offset (dx, dy) cm from the centre.

    West: by direction, rising toward the foothills (as it always has). East:
    the plains by distance east (``plains_m``), measured against ``base_m``,
    the terrain's Z = 0. Due north and south keep the old plain, PLAINS_CM.
    """
    west = max(0.0, -dx / (math.hypot(dx, dy) or 1.0))
    old = PLAINS_CM + (FOOTHILLS_CM - PLAINS_CM) * west ** 1.5
    w = _east_weight(dx, dy)
    if w <= 0.0:
        return old
    plains = (plains_m(dx / 100.0) - base_m) * 100.0
    return old + (plains - old) * w


def wave_weight(dx, dy):
    """Share of RING_WAVE_CM the undulations get in direction (dx, dy)."""
    west = max(0.0, -dx / (math.hypot(dx, dy) or 1.0))
    old = 0.35 + 0.65 * west
    return old + (EAST_WAVE - old) * _east_weight(dx, dy)


def curvature_drop_cm(distance_cm):
    """How far ground ``distance_cm`` from the level origin sits below the flat level.

    d^2 / 2R x (1 - k), as redrocks.denver.curvature_drop_m: the Earth's fall
    below the tangent plane, less what refraction bends back.
    """
    d = distance_cm / 100.0
    return d * d / (2.0 * EARTH_RADIUS_M) * (1.0 - REFRACTION_K) * 100.0


def _wave(angle, k):
    """-1..1 ridges and saddles around the skirt; ring k shifts the phases so
    the hills also vary outward instead of running as concentric terraces."""
    return (0.35 * math.sin(3 * angle + 1.3 + 0.7 * k)
            + 0.25 * math.sin(8 * angle + 0.4 - 0.9 * k)
            + 0.2 * math.sin(19 * angle + 2.0 + 1.7 * k)
            + 0.12 * math.sin(37 * angle + 0.9 - 2.3 * k)
            + 0.08 * math.sin(71 * angle + 1.6 + 3.1 * k))


def skirt_mesh(border, base_m=BASE_M):
    """(vertices, uvs, triangles) of the skirt around a CCW square border.

    Vertex i of ring k is border vertex i pushed out from the centre so ring k
    is the square grown by RING_OUT_CM[k]; triangles wind counter-clockwise seen
    from above (normals up), matching the terrain OBJ. Ring 0 is the terrain's
    edge, exactly; every ring past it drops by curvature_drop_cm of its distance
    from the level origin. ``base_m`` is the terrain's Z = 0 elevation.
    """
    cx = sum(p[0] for p in border) / len(border)
    cy = sum(p[1] for p in border) / len(border)
    half = max(max(abs(p[0] - cx), abs(p[1] - cy)) for p in border)
    zs = [p[2] for p in border]
    verts = []
    for k, out_cm in enumerate(RING_OUT_CM):
        grow = (half + out_cm) / half
        heights = _smoothed(zs, RING_SMOOTH[k])
        for (x, y, _), h in zip(border, heights):
            dx, dy = x - cx, y - cy
            px, py = cx + dx * grow, cy + dy * grow
            z = h + (far_height(dx * grow, dy * grow, base_m) - h) * RING_BLEND[k]
            z += RING_WAVE_CM[k] * wave_weight(dx, dy) * _wave(math.atan2(dy, dx), k)
            if k:
                z -= curvature_drop_cm(math.hypot(px, py))
            verts.append((px, py, z))
    n = len(border)
    tris = []
    for k in range(len(RING_OUT_CM) - 1):
        a0, b0 = k * n, (k + 1) * n
        for i in range(n):
            j = (i + 1) % n
            tris.append((a0 + i, b0 + i, b0 + j))
            tris.append((a0 + i, b0 + j, a0 + j))
    uvs = [(x / 100000.0, y / 100000.0) for x, y, _ in verts]
    return verts, uvs, tris


def write_obj(path, verts, uvs, tris, name):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("# Red Rocks terrain skirt, generated by unreal/setup_environment.py\n")
        fh.write("o %s\n" % name)
        for x, y, z in verts:
            fh.write("v %.1f %.1f %.1f\n" % (x, y, z))
        for u, v in uvs:
            fh.write("vt %.5f %.5f\n" % (u, v))
        for a, b, c in tris:
            fh.write("f %d/%d %d/%d %d/%d\n" % (a + 1, a + 1, b + 1, b + 1, c + 1, c + 1))


# ------------------------------------------------------------------- Unreal side

def _step(label, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        log.append("ok    %s" % label)
        return result
    except Exception as exc:
        log.append("FAIL  %s: %r" % (label, exc))
        return None


def _actors():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def terrain_base_m(manifest_path):
    """The terrain's Z = 0 elevation from a build's manifest.json, else BASE_M."""
    try:
        with open(manifest_path) as fh:
            return float(json.load(fh)["terrain_mesh"]["min_elevation_m"])
    except (OSError, KeyError, TypeError, ValueError):
        return BASE_M


def _build_obj():
    saved = unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_saved_dir())
    path = os.path.join(saved, "RedRocksGen", SKIRT_NAME + ".obj")
    border = grid_border(read_obj_vertices(TERRAIN_OBJ))
    base_m = terrain_base_m(os.path.join(OUT_DIR, "manifest.json"))
    verts, uvs, tris = skirt_mesh(border, base_m)
    write_obj(path, verts, uvs, tris, SKIRT_NAME)
    log.append("      %s: %d vertices, %d triangles, plains against base %.3f m"
               % (path, len(verts), len(tris), base_m))
    return path


def _import(path):
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", path)
    task.set_editor_property("destination_path", ASSET_DIR)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", True)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    if not list(task.get_editor_property("imported_object_paths")):
        raise RuntimeError("import produced nothing for %s" % path)
    asset_path = "%s/%s" % (ASSET_DIR, SKIRT_NAME)
    mesh = unreal.EditorAssetLibrary.load_asset(asset_path)
    if not isinstance(mesh, unreal.StaticMesh):
        raise RuntimeError("no static mesh at %s" % asset_path)
    settings = mesh.get_editor_property("nanite_settings")
    settings.set_editor_property("enabled", False)
    mesh.set_editor_property("nanite_settings", settings)
    material = unreal.EditorAssetLibrary.load_asset(SKIRT_MATERIAL)
    if material is None:
        raise RuntimeError("no %s; run setup_materials.py first" % SKIRT_MATERIAL)
    mesh.set_material(0, material)
    if not unreal.EditorAssetLibrary.save_asset(asset_path, False):
        raise RuntimeError("could not save %s" % asset_path)
    return mesh


def _place(mesh):
    for old in [a for a in _actors().get_all_level_actors()
                if a.get_actor_label() == SKIRT_NAME]:
        _actors().destroy_actor(old)
    actor = _actors().spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(0.0, 0.0, 0.0))
    comp = actor.static_mesh_component
    comp.set_static_mesh(mesh)
    actor.set_actor_label(SKIRT_NAME)
    # The OBJ importer negates Y, which is the terrain -> level map (level_frame.py).
    actor.set_actor_scale3d(unreal.Vector(*level_frame.OBJ_ACTOR_SCALE))
    actor.set_mobility(unreal.ComponentMobility.STATIC)
    comp.set_collision_profile_name("NoCollision")
    actor.set_actor_enable_collision(False)
    comp.set_editor_property("cast_shadow", False)
    return actor


def _save():
    if not unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level():
        raise RuntimeError("save_current_level returned False (read-only .umap?)")


def main():
    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not level_subsystem.load_level(LEVEL_PATH):
        log.append("ERROR: could not load %s" % LEVEL_PATH)
        return
    path = _step("skirt OBJ from the terrain edge", _build_obj)
    mesh = _step("import %s" % SKIRT_NAME, _import, path) if path else None
    if mesh is not None:
        _step("place %s (no collision, no shadow)" % SKIRT_NAME, _place, mesh)
    _step("save %s" % LEVEL_PATH, _save)


if hasattr(unreal, "EditorAssetLibrary"):
    try:
        main()
    except Exception as exc:
        log.append("ERROR: %r" % (exc,))
    with open(LOG_PATH, "w") as fh:
        fh.write("\n".join(log) + "\n")
