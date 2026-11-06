"""Re-proportion a skinned character by scaling its bones.

Each deforming bone gets a length scale (along the bone, toward its main child)
and a girth scale (across it). The new skeleton keeps every joint's rotation and
moves its children by the parent's scaled offset; the skin follows by linear
blend skinning of the per-bone affine maps

    v' = sum_j w_j (p'_j + S_j (v - p_j)),   S_j = g_j I + (k_j - g_j) d_j d_j^T

so a lengthened forearm stretches evenly along its length, a girthier spine
widens the chest and pushes the clavicles apart, and the weights stay valid.

Works in the body frame: centimetres, Z up, as Unreal imports the mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    """Scale of one bone: `length` along it, `girth` across it."""

    length: float = 1.0
    girth: float = 1.0


# Bones whose name contains one of these follow their parent's segment (twist
# bones lie along their parent and carry its flesh).
FOLLOW_PARENT = ("twist",)
# Virtual bones: not deforming; placed on the real bone they shadow afterwards.
VIRTUAL_TARGETS = {
    "ik_foot_l": "foot_l", "ik_foot_r": "foot_r",
    "ik_hand_l": "hand_l", "ik_hand_r": "hand_r", "ik_hand_gun": "hand_r",
}


def main_child(names: list[str], parents: np.ndarray, j: int) -> int | None:
    """The child a bone points at: the one whose subtree is largest, ignoring
    twist and virtual bones (spine to spine, hand to the middle finger)."""
    kids = [c for c in range(len(names)) if parents[c] == j
            and not any(k in names[c] for k in FOLLOW_PARENT) and names[c] not in VIRTUAL_TARGETS]
    if not kids:
        return None
    for prefer in ("middle", "spine", "neck", "head", "lowerarm", "hand", "calf", "foot", "ball"):
        for c in kids:
            if names[c].startswith(prefer):
                return c

    def size(c: int) -> int:
        return 1 + sum(size(g) for g in range(len(names)) if parents[g] == c)

    return max(kids, key=size)


def bone_directions(names: list[str], parents: np.ndarray, heads: np.ndarray) -> np.ndarray:
    """(j, 3) unit direction of each bone in the body frame."""
    n = len(names)
    dirs = np.zeros((n, 3))
    for j in range(n):
        c = main_child(names, parents, j)
        if c is not None and np.linalg.norm(heads[c] - heads[j]) > 1e-6:
            dirs[j] = heads[c] - heads[j]
    for j in range(n):  # leaves and twist bones: the parent's direction
        if np.linalg.norm(dirs[j]) < 1e-9 or any(k in names[j] for k in FOLLOW_PARENT):
            p = parents[j]
            while p >= 0 and np.linalg.norm(dirs[p]) < 1e-9:
                p = parents[p]
            dirs[j] = dirs[p] if p >= 0 else (0, 0, 1)
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def segment_for(name: str, names: list[str], parents: np.ndarray, segments: dict[str, Segment]) -> Segment:
    """A bone's segment: its own entry, else (for twist bones) its parent's, else 1."""
    if name in segments:
        return segments[name]
    if any(k in name for k in FOLLOW_PARENT):
        p = parents[names.index(name)]
        if p >= 0:
            return segment_for(names[p], names, parents, segments)
    base = name[:-2] if name.endswith(("_l", "_r")) else name
    for key, seg in segments.items():  # "index_01" covers index_01_l and index_01_r
        if key == base:
            return seg
    return Segment()


@dataclass
class Reproportioned:
    heads: np.ndarray       # (j, 3) new joint positions
    affine: np.ndarray      # (j, 3, 3) S_j
    old_heads: np.ndarray   # (j, 3)


def reproportion(names: list[str], parents: np.ndarray, heads: np.ndarray,
                 segments: dict[str, Segment], scale: float = 1.0) -> Reproportioned:
    """New joint positions and per-bone linear maps for `segments` and an overall `scale`.

    Joints are visited parents first. A child's offset from its parent goes
    through the parent's map, so lengthening the upper arm carries the forearm
    and hand out with it.
    """
    n = len(names)
    dirs = bone_directions(names, parents, heads)
    affine = np.zeros((n, 3, 3))
    new = np.zeros_like(heads)
    for j in range(n):
        seg = segment_for(names[j], names, parents, segments)
        d = dirs[j][:, None]
        affine[j] = scale * (seg.girth * np.eye(3) + (seg.length - seg.girth) * (d @ d.T))
        p = parents[j]
        new[j] = scale * heads[j] if p < 0 else new[p] + affine[p] @ (heads[j] - heads[p])
    for v, real in VIRTUAL_TARGETS.items():
        if v in names and real in names:
            new[names.index(v)] = new[names.index(real)]
    return Reproportioned(new, affine, heads.copy())


def skin_points(rp: Reproportioned, points: np.ndarray, joints: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Move bind-pose points (n, 3) with their skin weights through the bone maps."""
    out = np.zeros_like(points)
    for k in range(joints.shape[1]):
        j = joints[:, k]
        w = weights[:, k][:, None]
        local = points - rp.old_heads[j]
        moved = rp.heads[j] + np.einsum("nab,nb->na", rp.affine[j], local)
        out += w * moved
    return out


def skin_normals(rp: Reproportioned, normals: np.ndarray, joints: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Normals through the inverse-transpose of the blended bone maps."""
    blended = np.zeros((len(normals), 3, 3))
    for k in range(joints.shape[1]):
        blended += weights[:, k][:, None, None] * rp.affine[joints[:, k]]
    inv_t = np.linalg.inv(blended).transpose(0, 2, 1)
    out = np.einsum("nab,nb->na", inv_t, normals)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def rebuild_locals(parents: np.ndarray, old_globals: np.ndarray, new_heads: np.ndarray) -> np.ndarray:
    """Local (parent-relative) matrices for the new heads with the old rotations."""
    g = old_globals.copy()
    g[:, :3, 3] = new_heads
    out = np.zeros_like(g)
    for j in range(len(parents)):
        p = parents[j]
        out[j] = g[j] if p < 0 else np.linalg.inv(g[p]) @ g[j]
    return out
