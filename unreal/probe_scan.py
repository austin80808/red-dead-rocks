"""Compare the ml0w USD's own prim transforms with the chunks Unreal imported.

Run inside the editor (Cmd box), with the USD plugins enabled:
    py "/Users/austin/dev/redrocks-ue/unreal/probe_scan.py"

Writes /tmp/redrocks_probe_scan.log: the stage's up axis and units, every mesh
prim's world bounds and transform, and each imported chunk's local mesh bounds
and placed actor transform. If the imported meshes are in prim-local space, the
prim transforms are what puts the 31 chunks back together.
"""

import unreal

SCAN_FILE = "/Users/austin/dev/redrocks-ue/assets/scan/usdz_extract/scene.usdc"
ACTOR_LABEL = "Scan_ml0w"

log = []


def probe_stage():
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(SCAN_FILE)
    log.append("upAxis %s metersPerUnit %s" % (UsdGeom.GetStageUpAxis(stage), UsdGeom.GetStageMetersPerUnit(stage)))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    whole = Gf.Range3d()
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Xformable) and not prim.IsA(UsdGeom.Mesh):
            ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
            if ops:
                log.append("xform %s ops=%s" % (prim.GetPath(), [(o.GetOpName(), o.Get()) for o in ops]))
        if not prim.IsA(UsdGeom.Mesh):
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        whole.UnionWith(box)
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        local = cache.ComputeLocalBound(prim).ComputeAlignedRange()
        log.append(
            "mesh %s world min %s max %s | local min %s max %s | xlate %s"
            % (prim.GetPath(), box.GetMin(), box.GetMax(), local.GetMin(), local.GetMax(), m.ExtractTranslation())
        )
    log.append("WHOLE world min %s max %s size %s" % (whole.GetMin(), whole.GetMax(), whole.GetSize()))


def probe_level():
    for actor in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
        if not actor.get_actor_label().startswith(ACTOR_LABEL + "_"):
            continue
        mesh = actor.static_mesh_component.static_mesh
        box = mesh.get_bounding_box()
        log.append(
            "actor %s mesh %s tris %d local %s..%s loc %s scale %s visible %s mat %s"
            % (
                actor.get_actor_label(),
                mesh.get_name(),
                mesh.get_num_triangles(0),
                box.min,
                box.max,
                actor.get_actor_location(),
                actor.get_actor_scale3d(),
                not actor.is_hidden_ed(),
                [m.get_name() if m else None for m in actor.static_mesh_component.get_materials()],
            )
        )


for step in (probe_stage, probe_level):
    try:
        step()
    except Exception as exc:
        log.append("ERROR in %s: %r" % (step.__name__, exc))

with open("/tmp/redrocks_probe_scan.log", "w") as fh:
    fh.write("\n".join(str(x) for x in log))
