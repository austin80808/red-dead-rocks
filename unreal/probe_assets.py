"""Report what actually landed in /Game/RedRocks."""

import unreal

lines = []

for path in unreal.EditorAssetLibrary.list_assets("/Game/RedRocks", recursive=True):
    obj = unreal.EditorAssetLibrary.load_asset(path)
    name = path.split("/")[-1].split(".")[0]
    if isinstance(obj, unreal.StaticMesh):
        tris = obj.get_num_triangles(0) if hasattr(obj, "get_num_triangles") else "?"
        verts = obj.get_num_vertices(0) if hasattr(obj, "get_num_vertices") else "?"
        lines.append("%s STATICMESH tris=%s verts=%s" % (name, tris, verts))
    elif isinstance(obj, unreal.World):
        lines.append("%s LEVEL" % name)
    else:
        lines.append("%s %s" % (name, type(obj).__name__))

with open("/tmp/redrocks_probe.log", "w") as fh:
    fh.write("\n".join(lines))
