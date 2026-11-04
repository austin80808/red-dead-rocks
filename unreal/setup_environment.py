"""Hide the edge of the world with a terrain skirt.

Run after setup_materials.py (rebuild_level.py does). Headless:
    UnrealEditor-Cmd RedRocks.uproject -run=pythonscript \
        -script=unreal/setup_environment.py -unattended -nosplash -stdout

The DEM terrain is a 2.4 km square. Past it the SkyAtmosphere's planet ground
shows through as a flat dark-blue plane, so from the bowl and from the air the
world visibly ends. This script reads out/RedRocks_Terrain.obj, takes its edge
vertices, and grows rings outward from them to 120 km past the edge:

* the first ring is the terrain's edge itself, so the seam is exact;
* each ring outward smooths the edge heights more and blends toward a far
  profile: the plains to the east settle just below the venue, the foothills
  to the west keep rising toward the Front Range;
* broad undulations break up the outline, and the height fog
  (setup_lighting.py) turns it into distant haze.

It is written as an OBJ under the project's Saved/RedRocksGen/, imported to
/Game/RedRocks/Terrain_Skirt with MI_Sandstone_Ground, and placed with the same
Y scale of -1 as every other OBJ actor. It has no collision (no one can walk
there) and casts no shadows.

Trees and shrubs are setup_vegetation.py's (it replaced the sphere-clump
scrub this script used to scatter). The geometry functions are plain Python, tested offline.
"""

import math
import os

import unreal

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
PLAINS_CM = -4000.0        # far east: the plains toward Denver, ~1.71 km
FOOTHILLS_CM = 90000.0     # far west: foothills toward the Front Range, ~2.65 km

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


def far_height(dx, dy):
    """Height the skirt settles to far out, by direction from the centre."""
    west = max(0.0, -dx / (math.hypot(dx, dy) or 1.0))
    return PLAINS_CM + (FOOTHILLS_CM - PLAINS_CM) * west ** 1.5


def _wave(angle, k):
    """-1..1 ridges and saddles around the skirt; ring k shifts the phases so
    the hills also vary outward instead of running as concentric terraces."""
    return (0.35 * math.sin(3 * angle + 1.3 + 0.7 * k)
            + 0.25 * math.sin(8 * angle + 0.4 - 0.9 * k)
            + 0.2 * math.sin(19 * angle + 2.0 + 1.7 * k)
            + 0.12 * math.sin(37 * angle + 0.9 - 2.3 * k)
            + 0.08 * math.sin(71 * angle + 1.6 + 3.1 * k))


def skirt_mesh(border):
    """(vertices, uvs, triangles) of the skirt around a CCW square border.

    Vertex i of ring k is border vertex i pushed out from the centre so ring k
    is the square grown by RING_OUT_CM[k]; triangles wind counter-clockwise seen
    from above (normals up), matching the terrain OBJ.
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
            z = h + (far_height(dx, dy) - h) * RING_BLEND[k]
            west = max(0.0, -dx / (math.hypot(dx, dy) or 1.0))
            z += RING_WAVE_CM[k] * (0.35 + 0.65 * west) * _wave(math.atan2(dy, dx), k)
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


def _build_obj():
    saved = unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_saved_dir())
    path = os.path.join(saved, "RedRocksGen", SKIRT_NAME + ".obj")
    border = grid_border(read_obj_vertices(TERRAIN_OBJ))
    verts, uvs, tris = skirt_mesh(border)
    write_obj(path, verts, uvs, tris, SKIRT_NAME)
    log.append("      %s: %d vertices, %d triangles" % (path, len(verts), len(tris)))
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
    # The OBJ importer negates Y; -1 restores north (README gotchas).
    actor.set_actor_scale3d(unreal.Vector(1.0, -1.0, 1.0))
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
