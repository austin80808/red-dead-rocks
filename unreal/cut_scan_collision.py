"""Cut the seating out of the ml0w scan's collision.

The scan is hidden over the seating (its fade mask, see import_scan.py), but
its chunks keep complex collision, so whatever the drone caught standing in the
rows (gear, a structure at the top, ground a step above our treads) would still
block the player there, unseen. This gives every chunk that reaches the
seating a collision-only copy with the triangles over the seating deleted, set
as the chunk's ``ComplexCollisionMesh``: the scan collides everywhere you can
see it and nowhere under the rows. The terraces are the floor there.

The seating comes from out/seating_cut.json (``redrocks build``): run-length
grid rows in level cm, X east, Y north. A triangle goes when its centre, a
corner or an edge midpoint is in the cut. Chunks that do not reach the seating are left alone.

Run after import_scan.py and before setup_gameplay.py (rebuild_level.py does).
Re-running replaces the copies. Log: /tmp/redrocks_scan_cut.log.
"""

import json
import math
import os

import unreal

OUT_DIR = os.environ.get("REDROCKS_OUT_DIR", "/Users/austin/dev/redrocks-ue/out")
LOG_PATH = os.path.join(os.environ.get("REDROCKS_LOG_DIR", "/tmp"), "redrocks_scan_cut.log")
COPY_DIR = "/Game/RedRocks/Scan/Collision"
SCAN_PREFIX = "Scan_ml0w"
LEVEL_PATH = "/Game/RedRocks/Lvl_RedRocks"

log = []
eal = unreal.EditorAssetLibrary


def load_cut(path):
    """(x0, y1, texel, {row: [(first, end), ...]}) from seating_cut.json."""
    with open(path) as fh:
        cut = json.load(fh)
    rows = {}
    for r, a, b in cut["runs"]:
        rows.setdefault(r, []).append((a, b))
    return float(cut["x0_cm"]), float(cut["y1_cm"]), float(cut["texel_cm"]), rows


def in_cut(x, y, cut):
    x0, y1, t, rows = cut
    runs = rows.get(int((y1 - y) // t))
    if not runs:
        return False
    j = int((x - x0) // t)
    return any(a <= j < b for a, b in runs)


def _world(transform):
    """A fast local -> world (x, y) for an actor transform (scale, yaw, location)."""
    s = transform.scale3d
    yaw = math.radians(transform.rotation.rotator().yaw)
    c, si = math.cos(yaw), math.sin(yaw)
    t = transform.translation

    def f(p):
        x, y = p.x * s.x, p.y * s.y
        return (c * x - si * y + t.x, si * x + c * y + t.y)
    return f


def cut_triangles(desc, to_world, cut):
    """Triangle ids whose centroid lies over the seating."""
    pos = {}
    doomed = []
    for k in range(desc.get_triangle_count()):
        tri = unreal.TriangleID(k)
        if not desc.is_triangle_valid(tri):
            continue
        corners = []
        # UE 5.8's binding returns the array's header too: six items, of which
        # the last three are the triangle's corners.
        for v in list(desc.get_triangle_vertices(tri))[-3:]:
            key = v.id_value
            if key not in pos:
                pos[key] = to_world(desc.get_vertex_position(v))
            corners.append(pos[key])
        # The centre, the corners and the edge midpoints: the scan is coarse at
        # the top of the bowl, and a big triangle can cross the seating's edge
        # with its centre just outside it.
        k_ = len(corners)
        probes = [(sum(c[0] for c in corners) / k_, sum(c[1] for c in corners) / k_)] + corners + [
            ((corners[i][0] + corners[(i + 1) % k_][0]) / 2.0, (corners[i][1] + corners[(i + 1) % k_][1]) / 2.0)
            for i in range(k_)]
        if any(in_cut(x, y, cut) for x, y in probes):
            doomed.append(tri)
    return doomed


def collision_copy(mesh, doomed):
    """A copy of ``mesh`` without ``doomed`` triangles, Nanite off, in COPY_DIR."""
    name = mesh.get_name() + "_NoSeating"
    path = "%s/%s" % (COPY_DIR, name)
    if eal.does_asset_exist(path):
        eal.delete_asset(path)
    copy = eal.duplicate_asset(mesh.get_path_name().split(".")[0], path)
    if copy is None:
        raise RuntimeError("could not duplicate %s" % mesh.get_path_name())
    desc = copy.get_static_mesh_description(0)
    for tri in doomed:
        desc.delete_triangle(tri)
    copy.build_from_static_mesh_descriptions([desc])
    nanite = copy.get_editor_property("nanite_settings")
    nanite.enabled = False
    copy.set_editor_property("nanite_settings", nanite)
    if not eal.save_loaded_asset(copy, False):
        raise RuntimeError("could not save %s" % path)
    return copy


def main():
    path = os.path.join(OUT_DIR, "seating_cut.json")
    if not os.path.exists(path):
        log.append("skip  no %s (redrocks build without the scan): scan collision left whole" % path)
        return
    cut = load_cut(path)
    x0, y1, t, rows = cut
    cx0, cx1 = x0 + t * min(a for runs in rows.values() for a, _ in runs), x0 + t * max(b for runs in rows.values() for _, b in runs)
    cy1, cy0 = y1 - t * min(rows), y1 - t * (max(rows) + 1)
    unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).load_level(LEVEL_PATH)
    if not eal.does_directory_exist(COPY_DIR):
        eal.make_directory(COPY_DIR)
    actors = [a for a in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
              if a.get_actor_label().startswith(SCAN_PREFIX) and isinstance(a, unreal.StaticMeshActor)]
    done = {}
    for actor in actors:
        mesh = actor.static_mesh_component.static_mesh
        if mesh is None or mesh.get_path_name() in done:
            continue
        origin, extent = actor.get_actor_bounds(False)
        if (origin.x + extent.x < cx0 or origin.x - extent.x > cx1
                or origin.y + extent.y < cy0 or origin.y - extent.y > cy1):
            done[mesh.get_path_name()] = 0  # nowhere near the seating
            continue
        desc = mesh.get_static_mesh_description(0)
        doomed = cut_triangles(desc, _world(actor.get_actor_transform()), cut)
        done[mesh.get_path_name()] = len(doomed)
        if not doomed:
            if mesh.get_editor_property("complex_collision_mesh") is not None:
                mesh.set_editor_property("complex_collision_mesh", None)
                eal.save_loaded_asset(mesh, False)
            continue
        copy = collision_copy(mesh, doomed)
        mesh.set_editor_property("complex_collision_mesh", copy)
        if not eal.save_loaded_asset(mesh, False):
            raise RuntimeError("could not save %s (read-only? chmod -R u+w Content)" % mesh.get_path_name())
        log.append("ok    %s: %d of %d triangles over the seating left out of its collision"
                   % (mesh.get_name(), len(doomed), desc.get_triangle_count()))
    cut_meshes = sum(1 for n in done.values() if n)
    log.append("ok    %d chunks, %d reach the seating, %d triangles out of the scan's collision"
               % (len(done), cut_meshes, sum(done.values())))


if hasattr(unreal, "EditorAssetLibrary"):
    try:
        main()
    except Exception as exc:
        log.append("ERROR: %r" % (exc,))
    with open(LOG_PATH, "w") as fh:
        fh.write("\n".join(log) + "\n")
    for line in log:
        (unreal.log_error if line.startswith(("FAIL", "ERROR")) else unreal.log)(line)
