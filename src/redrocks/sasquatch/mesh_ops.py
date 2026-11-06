"""Triangle-mesh helpers for the sculpt: welding, normals, smoothing, adaptive
subdivision, surface sampling. numpy only.

Meshes here are welded (one vertex per position), with faces wound so the
cross product points out of the body in the body frame.
"""

from __future__ import annotations

import numpy as np


def weld(positions: np.ndarray, tolerance: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertices closer than `tolerance` (grid-rounded). Returns (keep, remap):
    `keep` indexes the first vertex of each merged group, `remap` maps every old
    vertex to its new index."""
    keys = np.round(positions / tolerance).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)            # keep the original order of first appearance
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return first[order], rank[inverse.reshape(-1)]


def drop_degenerate(faces: np.ndarray) -> np.ndarray:
    ok = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    return faces[ok]


def face_normals(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = (positions[faces[:, i]] for i in range(3))
    return np.cross(b - a, c - a)  # length = 2 x area


def vertex_normals(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    fn = face_normals(positions, faces)
    vn = np.zeros_like(positions)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn)
    n = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.where(n > 1e-12, n, 1.0)


def face_areas(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(face_normals(positions, faces), axis=1)


def edges(faces: np.ndarray) -> np.ndarray:
    """Unique undirected edges (e, 2), smaller index first."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    return np.unique(e, axis=0)


def neighbour_mean(values: np.ndarray, edge_list: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean of each vertex's neighbours' values, and the neighbour counts."""
    acc = np.zeros((n,) + values.shape[1:])
    cnt = np.zeros(n)
    np.add.at(acc, edge_list[:, 0], values[edge_list[:, 1]])
    np.add.at(acc, edge_list[:, 1], values[edge_list[:, 0]])
    np.add.at(cnt, edge_list[:, 0], 1)
    np.add.at(cnt, edge_list[:, 1], 1)
    safe = np.maximum(cnt, 1)
    return acc / safe.reshape((-1,) + (1,) * (values.ndim - 1)), cnt


def taubin_smooth(positions: np.ndarray, faces: np.ndarray, weight: np.ndarray | float = 1.0,
                  iterations: int = 10, lam: float = 0.5, mu: float = -0.53) -> np.ndarray:
    """Smooth without shrinking (Taubin's lambda|mu). `weight` (per vertex, 0..1) masks it."""
    p = positions.copy()
    e = edges(faces)
    w = np.broadcast_to(np.asarray(weight, float), (len(p),))[:, None]
    for _ in range(iterations):
        for f in (lam, mu):
            mean, cnt = neighbour_mean(p, e, len(p))
            p = p + w * f * np.where(cnt[:, None] > 0, mean - p, 0.0)
    return p


def subdivide(positions: np.ndarray, faces: np.ndarray, refine: np.ndarray,
              attributes: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """Red-green refinement: split every face in `refine` (bool per face) into
    four, and the faces bordering them into two, so there are no T-junctions.

    New vertices sit on edge midpoints; `attributes` (per-vertex arrays) are
    averaged the same way. Returns positions, faces, attributes and, per new
    face, the index of the face it came from.
    """
    attributes = attributes or {}
    m = len(faces)
    red = refine.copy()
    fe = np.stack([np.sort(faces[:, [0, 1]], axis=1), np.sort(faces[:, [1, 2]], axis=1),
                   np.sort(faces[:, [2, 0]], axis=1)], axis=1)              # (m, 3, 2)
    all_e, inv = np.unique(fe.reshape(-1, 2), axis=0, return_inverse=True)
    inv = inv.reshape(m, 3)
    while True:  # closure: a face with two split edges becomes red
        split = np.zeros(len(all_e), bool)
        split[inv[red].reshape(-1)] = True
        count = split[inv].sum(axis=1)
        grow = (~red) & (count >= 2)
        if not grow.any():
            break
        red |= grow
    split_ids = np.flatnonzero(split)
    mid_index = np.full(len(all_e), -1)
    mid_index[split_ids] = len(positions) + np.arange(len(split_ids))
    a, b = all_e[split_ids, 0], all_e[split_ids, 1]
    new_pos = np.concatenate([positions, 0.5 * (positions[a] + positions[b])])
    new_attr = {k: np.concatenate([v, 0.5 * (v[a] + v[b])]) for k, v in attributes.items()}

    out, origin = [], []
    mids = mid_index[inv]                                                  # (m, 3) midpoint of edge (0-1, 1-2, 2-0)
    for f in range(m):
        v0, v1, v2 = faces[f]
        m01, m12, m20 = mids[f]
        if red[f]:
            out += [(v0, m01, m20), (m01, v1, m12), (m20, m12, v2), (m01, m12, m20)]
            origin += [f] * 4
        elif m01 >= 0:
            out += [(v0, m01, v2), (m01, v1, v2)]
            origin += [f] * 2
        elif m12 >= 0:
            out += [(v0, v1, m12), (v0, m12, v2)]
            origin += [f] * 2
        elif m20 >= 0:
            out += [(v0, v1, m20), (m20, v1, v2)]
            origin += [f] * 2
        else:
            out.append((v0, v1, v2))
            origin.append(f)
    return new_pos, np.array(out, dtype=np.int64), new_attr, np.array(origin)


def sample_surface(positions: np.ndarray, faces: np.ndarray, density: np.ndarray,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random points with `density` (per vertex, points per cm^2, interpolated
    per face as its mean). Returns points, their face and barycentric coords."""
    area = face_areas(positions, faces)
    expected = area * density[faces].mean(axis=1)
    counts = rng.poisson(expected)
    face_of = np.repeat(np.arange(len(faces)), counts)
    r1 = np.sqrt(rng.random(len(face_of)))
    r2 = rng.random(len(face_of))
    bary = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], axis=1)
    tri = positions[faces[face_of]]                                        # (k, 3, 3)
    pts = np.einsum("ki,kij->kj", bary, tri)
    return pts, face_of, bary


def grid_nearest(points: np.ndarray, centres: np.ndarray, cell: float) -> np.ndarray:
    """Index of the nearest centre to each point, searching the 27 grid cells
    around it (centres further than `cell` may be missed; then the nearest
    found, or -1)."""
    key = np.floor(centres / cell).astype(np.int64)
    buckets: dict[tuple, list[int]] = {}
    for i, k in enumerate(map(tuple, key)):
        buckets.setdefault(k, []).append(i)
    pkey = np.floor(points / cell).astype(np.int64)
    best = np.full(len(points), -1)
    best_d = np.full(len(points), np.inf)
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    # group points by cell so each candidate list is built once
    order = np.lexsort(pkey.T[::-1])
    sk = pkey[order]
    breaks = np.flatnonzero(np.any(np.diff(sk, axis=0) != 0, axis=1)) + 1
    for grp in np.split(order, breaks):
        k = tuple(pkey[grp[0]])
        cand = [i for o in offsets for i in buckets.get((k[0] + o[0], k[1] + o[1], k[2] + o[2]), ())]
        if not cand:
            continue
        cand = np.array(cand)
        d = np.linalg.norm(points[grp][:, None, :] - centres[cand][None, :, :], axis=2)
        j = np.argmin(d, axis=1)
        best[grp] = cand[j]
        best_d[grp] = d[np.arange(len(grp)), j]
    return best
