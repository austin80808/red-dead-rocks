"""Export a sample of the ml0w scan's points for fitting it to the DEM.

Run inside the editor (Cmd box), with the USD plugins enabled:
    py "/Users/austin/dev/redrocks-ue/unreal/export_scan_points.py"

Writes /tmp/redrocks_scan_points.csv as x,y,z in the frame Unreal imports the
scan into (USD world space with Y-up turned into Z-up: ue = (x, z, y)), so the
fitted transform applies directly to the imported chunk actors. Then run
tools/fit_scan.py.
"""

from pxr import Usd, UsdGeom

SCAN_FILE = "/Users/austin/dev/redrocks-ue/assets/scan/usdz_extract/scene.usdc"
OUT = "/tmp/redrocks_scan_points.csv"
STEP = 40  # keep every 40th vertex; ~3M triangles gives tens of thousands

stage = Usd.Stage.Open(SCAN_FILE)
rows = []
for prim in stage.Traverse():
    if not prim.IsA(UsdGeom.Mesh):
        continue
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
    for i in range(0, len(points), STEP):
        p = m.Transform(points[i])
        rows.append("%.3f,%.3f,%.3f" % (p[0], p[2], p[1]))

with open(OUT, "w") as fh:
    fh.write("\n".join(rows))
