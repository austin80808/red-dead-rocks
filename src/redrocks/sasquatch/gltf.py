"""Read and write the skinned binary glTF (.glb) the sasquatch is built from.

Only what a skinned character needs: one skin, its joint nodes, triangle
primitives with POSITION, NORMAL, TEXCOORD_0, JOINTS_0/WEIGHTS_0 (and _1),
COLOR_0, and named materials. numpy only, no glTF library.

Positions come back in the glTF scene frame (metres, Y up). `body_frame`
converts to the frame Unreal imports into (centimetres, Z up), and back.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_NORMALIZED_SCALE = {np.uint8: 255.0, np.uint16: 65535.0, np.int8: 127.0, np.int16: 32767.0}


@dataclass
class Primitive:
    """One triangle list with one material."""

    material: str
    positions: np.ndarray            # (n, 3) float, scene frame
    normals: np.ndarray              # (n, 3)
    uvs: np.ndarray                  # (n, 2)
    joints: np.ndarray               # (n, k) int, indices into Skin.joints
    weights: np.ndarray              # (n, k) float, rows sum to 1
    indices: np.ndarray              # (m, 3) int
    colors: np.ndarray | None = None  # (n, 4) float 0..1


@dataclass
class Skin:
    """The skeleton: joint nodes in skin order."""

    names: list[str]
    parents: np.ndarray          # (j,) int, -1 for the root joint
    local_matrices: np.ndarray   # (j, 4, 4) rest pose, relative to the parent joint (or the scene)
    inverse_bind: np.ndarray     # (j, 4, 4)

    def global_matrices(self) -> np.ndarray:
        """(j, 4, 4) rest pose in the scene frame."""
        out = np.zeros_like(self.local_matrices)
        for j in range(len(self.names)):  # parents come first (checked on load)
            p = self.parents[j]
            out[j] = self.local_matrices[j] if p < 0 else out[p] @ self.local_matrices[j]
        return out

    def index(self, name: str) -> int:
        return self.names.index(name)


@dataclass
class Model:
    skin: Skin
    primitives: list[Primitive]
    root_matrix: np.ndarray = field(default_factory=lambda: np.eye(4))  # scene transform above the root joint
    root_name: str = "Armature"


# ---------------------------------------------------------------------------
# Matrices


def trs_matrix(t=(0, 0, 0), r=(0, 0, 0, 1), s=(1, 1, 1)) -> np.ndarray:
    """4x4 from glTF translation, rotation (x, y, z, w) and scale."""
    x, y, z, w = r
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = rot * np.asarray(s, float)[None, :]
    m[:3, 3] = t
    return m


def matrix_to_trs(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Translation, unit quaternion (x, y, z, w) and scale of a 4x4 (no shear)."""
    t = m[:3, 3].copy()
    s = np.linalg.norm(m[:3, :3], axis=0)
    r = m[:3, :3] / s[None, :]
    tr = np.trace(r)
    if tr > 0:
        k = 0.5 / np.sqrt(tr + 1.0)
        q = np.array([(r[2, 1] - r[1, 2]) * k, (r[0, 2] - r[2, 0]) * k, (r[1, 0] - r[0, 1]) * k, 0.25 / k])
    else:
        i = int(np.argmax(np.diag(r)))
        j, k_ = (i + 1) % 3, (i + 2) % 3
        sq = np.sqrt(max(1e-12, 1.0 + r[i, i] - r[j, j] - r[k_, k_])) * 2
        q = np.zeros(4)
        q[i] = 0.25 * sq
        q[j] = (r[j, i] + r[i, j]) / sq
        q[k_] = (r[k_, i] + r[i, k_]) / sq
        q[3] = (r[k_, j] - r[j, k_]) / sq
    return t, q / np.linalg.norm(q), s


# ---------------------------------------------------------------------------
# Reading


def _split_glb(data: bytes) -> tuple[dict, bytes]:
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError("not a glTF 2 binary")
    offset, doc, binary = 12, None, b""
    while offset < length:
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8: offset + 8 + chunk_len]
        if chunk_type == 0x4E4F534A:
            doc = json.loads(chunk.decode("utf-8"))
        elif chunk_type == 0x004E4942:
            binary = chunk
        offset += 8 + chunk_len
    if doc is None:
        raise ValueError("glTF binary without a JSON chunk")
    return doc, binary


def _accessor(doc: dict, binary: bytes, index: int) -> np.ndarray:
    acc = doc["accessors"][index]
    dtype = _COMPONENT[acc["componentType"]]
    width = _WIDTH[acc["type"]]
    count = acc["count"]
    view = doc["bufferViews"][acc["bufferView"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    item = np.dtype(dtype).itemsize * width
    stride = view.get("byteStride", item)
    if stride == item:
        arr = np.frombuffer(binary, dtype=dtype, count=count * width, offset=start).reshape(count, width)
    else:
        raw = np.frombuffer(binary, dtype=np.uint8, count=stride * (count - 1) + item, offset=start)
        rows = np.lib.stride_tricks.as_strided(raw, shape=(count, item), strides=(stride, 1))
        arr = np.ascontiguousarray(rows).view(dtype).reshape(count, width)
    if acc.get("normalized") and dtype in _NORMALIZED_SCALE:
        arr = arr.astype(np.float64) / _NORMALIZED_SCALE[dtype]
    if acc["type"] == "MAT4":
        arr = arr.reshape(count, 4, 4).transpose(0, 2, 1)  # glTF matrices are column-major
    return np.array(arr)


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], float).reshape(4, 4).T
    return trs_matrix(node.get("translation", (0, 0, 0)), node.get("rotation", (0, 0, 0, 1)), node.get("scale", (1, 1, 1)))


def read_glb(path: str | Path) -> Model:
    """Load a skinned glTF binary (the first skin and every primitive of its mesh)."""
    doc, binary = _split_glb(Path(path).read_bytes())
    nodes = doc["nodes"]
    parent_of = {}
    for i, node in enumerate(nodes):
        for c in node.get("children", []):
            parent_of[c] = i
    skin_doc = doc["skins"][0]
    joint_nodes = skin_doc["joints"]
    order = {n: j for j, n in enumerate(joint_nodes)}

    def scene_matrix(n: int) -> np.ndarray:
        m = _node_matrix(nodes[n])
        while n in parent_of:
            n = parent_of[n]
            m = _node_matrix(nodes[n]) @ m
        return m

    names, parents, locals_ = [], [], []
    root_matrix, root_name = np.eye(4), "Armature"
    for j, n in enumerate(joint_nodes):
        names.append(nodes[n].get("name", "joint%d" % j))
        p = parent_of.get(n)
        while p is not None and p not in order:  # a non-joint node between joints: fold it in
            p = parent_of.get(p)
        if p is None:
            parents.append(-1)
            above = parent_of.get(n)
            root_matrix = scene_matrix(above) if above is not None else np.eye(4)
            if above is not None:
                root_name = nodes[above].get("name", root_name)
            locals_.append(_node_matrix(nodes[n]))
        else:
            if order[p] >= j:
                raise ValueError("joint %s listed before its parent" % names[-1])
            parents.append(order[p])
            locals_.append(np.linalg.inv(scene_matrix(p)) @ scene_matrix(n))
    locals_ = np.array(locals_)
    locals_[np.array(parents) < 0] = root_matrix @ locals_[np.array(parents) < 0]
    inverse_bind = (_accessor(doc, binary, skin_doc["inverseBindMatrices"])
                    if "inverseBindMatrices" in skin_doc else np.tile(np.eye(4), (len(names), 1, 1)))
    skin = Skin(names, np.array(parents), locals_, inverse_bind)

    mesh_node = next(n for n in nodes if n.get("skin") == 0 and "mesh" in n)
    prims = []
    materials = doc.get("materials", [])
    for p in doc["meshes"][mesh_node["mesh"]]["primitives"]:
        if p.get("mode", 4) != 4:
            continue
        a = p["attributes"]
        pos = _accessor(doc, binary, a["POSITION"]).astype(np.float64)
        n = len(pos)
        nrm = _accessor(doc, binary, a["NORMAL"]).astype(np.float64) if "NORMAL" in a else np.zeros((n, 3))
        uv = _accessor(doc, binary, a["TEXCOORD_0"]).astype(np.float64) if "TEXCOORD_0" in a else np.zeros((n, 2))
        joints = [_accessor(doc, binary, a[k]).astype(np.int64) for k in ("JOINTS_0", "JOINTS_1") if k in a]
        weights = [_accessor(doc, binary, a[k]).astype(np.float64) for k in ("WEIGHTS_0", "WEIGHTS_1") if k in a]
        jw = np.concatenate(joints, axis=1)
        ww = np.concatenate(weights, axis=1)
        ww = ww / np.maximum(ww.sum(axis=1, keepdims=True), 1e-12)
        idx = _accessor(doc, binary, p["indices"]).reshape(-1, 3).astype(np.int64)
        col = _accessor(doc, binary, a["COLOR_0"]).astype(np.float64) if "COLOR_0" in a else None
        mat = materials[p["material"]].get("name", "M%d" % p["material"]) if "material" in p else "Default"
        prims.append(Primitive(mat, pos, nrm, uv, jw, ww, idx, col))
    return Model(skin, prims, root_matrix=root_matrix, root_name=root_name)


# ---------------------------------------------------------------------------
# Writing


class _Packer:
    def __init__(self):
        self.blob = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def add(self, arr: np.ndarray, kind: str, target: int | None = None, minmax: bool = False) -> int:
        arr = np.ascontiguousarray(arr)
        comp = {np.dtype(np.float32): 5126, np.dtype(np.uint32): 5125, np.dtype(np.uint16): 5123,
                np.dtype(np.uint8): 5121}[arr.dtype]
        while len(self.blob) % 4:
            self.blob.append(0)
        view = {"buffer": 0, "byteOffset": len(self.blob), "byteLength": arr.nbytes}
        if target:
            view["target"] = target
        self.blob += arr.tobytes()
        self.views.append(view)
        count = arr.shape[0]
        acc = {"bufferView": len(self.views) - 1, "componentType": comp, "count": int(count), "type": kind}
        if minmax:
            acc["min"] = arr.reshape(count, -1).min(axis=0).astype(float).tolist()
            acc["max"] = arr.reshape(count, -1).max(axis=0).astype(float).tolist()
        self.accessors.append(acc)
        return len(self.accessors) - 1


def write_glb(model: Model, path: str | Path, mesh_name: str = "Sasquatch") -> None:
    """Write a skinned glTF binary: a scene root, the joint tree and one skinned mesh.

    The joints' rest pose is the bind pose (their inverse bind matrices are the
    inverses of their scene transforms), so an importer that takes either one
    gets the same skeleton.
    """
    skin = model.skin
    pk = _Packer()
    nodes: list[dict] = [{"name": model.root_name, "children": []}]
    joint_node = []
    for j, name in enumerate(skin.names):
        p = skin.parents[j]
        local = skin.local_matrices[j] if p >= 0 else np.linalg.inv(model.root_matrix) @ skin.local_matrices[j]
        t, q, s = matrix_to_trs(local)
        node = {"name": name, "translation": t.tolist(), "rotation": q.tolist()}
        if not np.allclose(s, 1.0, atol=1e-6):
            node["scale"] = s.tolist()
        nodes.append(node)
        joint_node.append(len(nodes) - 1)
        if p < 0:
            nodes[0]["children"].append(joint_node[j])
        else:
            nodes[joint_node[p]].setdefault("children", []).append(joint_node[j])
    rt, rq, rs = matrix_to_trs(model.root_matrix)
    nodes[0].update({"translation": rt.tolist(), "rotation": rq.tolist(), "scale": rs.tolist()})

    inv = np.linalg.inv(skin.global_matrices()).transpose(0, 2, 1).astype(np.float32).reshape(-1, 16)
    ibm = pk.add(inv, "MAT4")

    material_names: list[str] = []
    prims = []
    for prim in model.primitives:
        if prim.material not in material_names:
            material_names.append(prim.material)
        k = prim.joints.shape[1]
        pad = (-k) % 4
        joints = np.pad(prim.joints, ((0, 0), (0, pad))).astype(np.uint16)
        weights = np.pad(prim.weights, ((0, 0), (0, pad))).astype(np.float32)
        attrs = {
            "POSITION": pk.add(prim.positions.astype(np.float32), "VEC3", 34962, minmax=True),
            "NORMAL": pk.add(_unit(prim.normals).astype(np.float32), "VEC3", 34962),
            "TEXCOORD_0": pk.add(prim.uvs.astype(np.float32), "VEC2", 34962),
        }
        for g in range(joints.shape[1] // 4):
            attrs["JOINTS_%d" % g] = pk.add(joints[:, 4 * g:4 * g + 4], "VEC4", 34962)
            attrs["WEIGHTS_%d" % g] = pk.add(weights[:, 4 * g:4 * g + 4], "VEC4", 34962)
        if prim.colors is not None:
            attrs["COLOR_0"] = pk.add(np.clip(prim.colors, 0, 1).astype(np.float32), "VEC4", 34962)
        idx = pk.add(prim.indices.reshape(-1).astype(np.uint32), "SCALAR", 34963)
        prims.append({"attributes": attrs, "indices": idx, "material": material_names.index(prim.material), "mode": 4})

    # The skinned mesh is its own scene root (Interchange warns about a skinned
    # mesh node under another node; Epic's exporter keeps it at the root too).
    nodes.append({"name": mesh_name, "mesh": 0, "skin": 0})
    doc = {
        "asset": {"version": "2.0", "generator": "redrocks.sasquatch"},
        "scene": 0,
        "scenes": [{"nodes": [0, len(nodes) - 1]}],
        "nodes": nodes,
        "meshes": [{"name": mesh_name, "primitives": prims}],
        "skins": [{"name": mesh_name + "_Skin", "joints": joint_node, "inverseBindMatrices": ibm,
                   "skeleton": joint_node[int(np.flatnonzero(skin.parents < 0)[0])]}],
        "materials": [{"name": n, "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.8}}
                      for n in material_names],
        "accessors": pk.accessors,
        "bufferViews": pk.views,
        "buffers": [{"byteLength": len(pk.blob)}],
    }
    js = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    js += b" " * ((-len(js)) % 4)
    blob = bytes(pk.blob) + b"\0" * ((-len(pk.blob)) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, total))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A))
        fh.write(js)
        fh.write(struct.pack("<II", len(blob), 0x004E4942))
        fh.write(blob)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n > 1e-12, n, 1.0)


# ---------------------------------------------------------------------------
# Frames


def body_frame() -> np.ndarray:
    """3x3 taking glTF scene axes (metres, Y up) to Unreal's (Z up), before the x100.

    Epic's glTF exporter writes Unreal (x, y, z) as glTF (x, z, y), and the
    glTF importer undoes it, so this swap is its own inverse.
    """
    return np.array([[1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])


GLTF_TO_CM = 100.0
