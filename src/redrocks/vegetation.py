"""Procedural trees and shrubs for the Red Rocks foothills, and where they grow.

Nothing here is downloaded. Every model is grown from a seed in Python and
written as OBJ (centimetres, X east, Y north, Z up, base of the trunk at the
origin), and every leaf texture is painted with Pillow. ``redrocks vegetation``
writes them all to ``out/vegetation/`` together with ``vegetation.json``, the
list of instances that ``unreal/setup_vegetation.py`` places.

Species (heights are the generated model; instances scale them):

=====================  ======================  =================================
Rocky Mountain juniper  ``JuniperRM``  5.0 m    one gnarly trunk, dense irregular
                                                 conical crown, blue-green
One-seed juniper        ``JuniperOS``  3.6 m    2-4 stems from the base, broad
                                                 rounded crown to the ground
Ponderosa pine          ``Ponderosa`` 18.0 m    tall orange trunk clear for half
                                                 its height, sparse upturned
                                                 branches with needle tufts
Gambel (scrub) oak      ``Oak``        1.6 m    many thin stems, dense leafy dome
Mountain mahogany       ``Mahogany``   1.9 m    fewer stems, open grey-green
=====================  ======================  =================================

Each model has three parts, each its own OBJ so Unreal can treat them apart:

* ``Trunk``: trunk, stems and branches as closed tapered tubes. Watertight,
  outward-facing; this is the only part with collision.
* ``Crown``: the dense core of each foliage clump, a displaced icosphere. It
  gives the canopy its mass and a clean silhouette at distance.
* ``Leaves``: crossed leaf cards around every clump, UV-mapped into the
  species' 2x2 texture atlas (juniper sprays, pine needle tufts, oak leaves,
  mahogany leaves with their feathery plumes). Their normals are bent out from
  the crown so the canopy shades like a volume, not like flat cards.

Every face of every part has a UV. Variants are deterministic: the same name
and index always give the same mesh, byte for byte.

Placement reuses the masks the old scrub used (``setup_environment.py``): the
terrain grid's slope, the scanned bowl, roads, paths and parking, plus the
benches and stage. Ponderosa takes the upper and north-facing slopes, junipers
the open rocky slopes, oak and mahogany the lower, gentler ground, in thickets
by noise.
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------- meshes


@dataclass
class Mesh:
    """Indexed triangle mesh with separate position, UV and normal streams.

    Positions are welded (a seam never duplicates a position), so a closed
    surface is watertight by index; UVs and normals are per-corner streams, as
    OBJ's ``f v/vt/vn`` allows. Units are metres until :func:`write_obj`.
    """

    v: list = field(default_factory=list)
    vt: list = field(default_factory=list)
    vn: list = field(default_factory=list)
    f: list = field(default_factory=list)  # ((v0, v1, v2), (t0, t1, t2), (n0, n1, n2))

    def add_v(self, p) -> int:
        self.v.append((float(p[0]), float(p[1]), float(p[2])))
        return len(self.v) - 1

    def add_vt(self, uv) -> int:
        self.vt.append((float(uv[0]), float(uv[1])))
        return len(self.vt) - 1

    def add_vn(self, n) -> int:
        n = np.asarray(n, float)
        length = float(np.linalg.norm(n)) or 1.0
        self.vn.append(tuple(float(c) for c in n / length))
        return len(self.vn) - 1

    def tri(self, vs, ts, ns) -> None:
        self.f.append((tuple(vs), tuple(ts), tuple(ns)))

    @property
    def triangles(self) -> int:
        return len(self.f)

    def positions(self) -> np.ndarray:
        return np.asarray(self.v, float).reshape(-1, 3)

    def face_positions(self) -> np.ndarray:
        return np.asarray([fv for fv, _, _ in self.f], int).reshape(-1, 3)


def closed_and_outward(mesh: Mesh) -> bool:
    """True when every directed edge appears once and its reverse once, and the
    enclosed volume is positive (faces wind counter-clockwise seen from outside)."""
    faces = mesh.face_positions()
    if len(faces) == 0:
        return False
    directed = {}
    for a, b, c in faces:
        for e in ((a, b), (b, c), (c, a)):
            directed[e] = directed.get(e, 0) + 1
    if any(n != 1 for n in directed.values()):
        return False
    if any((b, a) not in directed for a, b in directed):
        return False
    return signed_volume(mesh) > 0.0


def signed_volume(mesh: Mesh) -> float:
    p = mesh.positions()
    f = mesh.face_positions()
    a, b, c = p[f[:, 0]], p[f[:, 1]], p[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def _perpendicular(t):
    helper = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    return _unit(np.cross(t, helper))


def add_tube(mesh: Mesh, path, radii, sides: int, gnarl: float = 0.0, rng=None,
             uv_scale: float = 1.0) -> None:
    """A closed tapered tube along a polyline, capped at both ends.

    ``path`` is (K, 3) metres, ``radii`` (K,). Rings are oriented by parallel
    transport so the tube does not twist. ``gnarl`` adds lumpy, fluted bark
    (a fraction of the radius). UV: U around the tube in metres of
    circumference, V along it in metres, both times ``uv_scale``.
    """
    path = np.asarray(path, float)
    radii = np.asarray(radii, float)
    k = len(path)
    if k < 2:
        raise ValueError("a tube needs at least two points")
    tangents = []
    for i in range(k):
        d = path[min(i + 1, k - 1)] - path[max(i - 1, 0)]
        tangents.append(_unit(d))
    normal = _perpendicular(tangents[0])
    frames = []
    for i, t in enumerate(tangents):
        if i:
            normal = normal - t * float(np.dot(normal, t))
            normal = _unit(normal) if np.linalg.norm(normal) > 1e-9 else _perpendicular(t)
        frames.append((normal, np.cross(t, normal)))
    phases = (rng.uniform(0, 2 * math.pi, 3) if rng is not None else np.zeros(3))
    ring_ids = []
    length = 0.0
    lengths = [0.0]
    for i in range(1, k):
        length += float(np.linalg.norm(path[i] - path[i - 1]))
        lengths.append(length)
    for i in range(k):
        n_vec, b_vec = frames[i]
        ids = []
        nids = []
        for j in range(sides):
            a = 2 * math.pi * j / sides
            r = radii[i]
            if gnarl:
                r *= 1.0 + gnarl * (0.6 * math.sin(3 * a + phases[0] + lengths[i] * 1.7)
                                    + 0.4 * math.sin(5 * a + phases[1] - lengths[i] * 2.3))
            radial = math.cos(a) * n_vec + math.sin(a) * b_vec
            ids.append(mesh.add_v(path[i] + r * radial))
            nids.append(mesh.add_vn(radial))
        uvs = [mesh.add_vt(((2 * math.pi * radii[i] * j / sides) * uv_scale, lengths[i] * uv_scale))
               for j in range(sides + 1)]
        ring_ids.append((ids, nids, uvs))
    for i in range(k - 1):
        (a_ids, a_n, a_uv), (b_ids, b_n, b_uv) = ring_ids[i], ring_ids[i + 1]
        for j in range(sides):
            j1 = (j + 1) % sides
            # Around the ring is +theta, along the tube is +t: theta x t = r,
            # so (a_j, a_j1, b_j1) faces outward.
            mesh.tri((a_ids[j], a_ids[j1], b_ids[j1]), (a_uv[j], a_uv[j + 1], b_uv[j + 1]),
                     (a_n[j], a_n[j1], b_n[j1]))
            mesh.tri((a_ids[j], b_ids[j1], b_ids[j]), (a_uv[j], b_uv[j + 1], b_uv[j]),
                     (a_n[j], b_n[j1], b_n[j]))
    for end, sign in ((0, -1.0), (k - 1, 1.0)):
        ids = ring_ids[end][0]
        centre = mesh.add_v(path[end])
        cn = mesh.add_vn(sign * tangents[end])
        n_vec, b_vec = frames[end]
        ct = mesh.add_vt((0.5, 0.5))
        rim = [mesh.add_vt((0.5 + 0.5 * math.cos(2 * math.pi * j / sides),
                            0.5 + 0.5 * math.sin(2 * math.pi * j / sides))) for j in range(sides)]
        for j in range(sides):
            j1 = (j + 1) % sides
            if sign < 0:
                mesh.tri((centre, ids[j1], ids[j]), (ct, rim[j1], rim[j]), (cn, cn, cn))
            else:
                mesh.tri((centre, ids[j], ids[j1]), (ct, rim[j], rim[j1]), (cn, cn, cn))


def _icosphere(subdiv: int):
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
             (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    verts = [tuple(_unit(v)) for v in verts]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
             (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
             (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(subdiv):
        cache = {}

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                verts.append(tuple(_unit(np.add(verts[a], verts[b]) / 2.0)))
                cache[key] = len(verts) - 1
            return cache[key]

        new = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new
    verts = np.asarray(verts, float)
    # Make every face wind counter-clockwise seen from outside.
    fixed = []
    for a, b, c in faces:
        n = np.cross(verts[b] - verts[a], verts[c] - verts[a])
        fixed.append((a, b, c) if np.dot(n, verts[a] + verts[b] + verts[c]) > 0 else (a, c, b))
    return verts, fixed


_ICO = {}


def add_blob(mesh: Mesh, centre, radii, rng, subdiv: int = 1, lump: float = 0.35,
             bend_to=None, bend: float = 0.5) -> None:
    """A closed, lumpy ellipsoid: the dense core of a foliage clump.

    Normals blend the blob's own with the direction from ``bend_to`` (the crown
    centre), so neighbouring clumps shade as one canopy.
    """
    if subdiv not in _ICO:
        _ICO[subdiv] = _icosphere(subdiv)
    unit, faces = _ICO[subdiv]
    centre = np.asarray(centre, float)
    radii = np.asarray(radii, float)
    waves = rng.normal(size=(4, 3))
    phases = rng.uniform(0, 2 * math.pi, 4)
    weights = np.array([0.45, 0.3, 0.15, 0.1])
    disp = (weights[None, :] * np.sin(unit @ (waves.T * 2.2) + phases[None, :])).sum(1)
    pts = centre + unit * radii * (1.0 + lump * disp)[:, None]
    ids = [mesh.add_v(p) for p in pts]
    nids = []
    for p, u in zip(pts, unit):
        n = u / radii * radii.mean()
        if bend_to is not None:
            n = (1 - bend) * _unit(n) + bend * _unit(p - np.asarray(bend_to, float))
        nids.append(mesh.add_vn(n + np.array([0.0, 0.0, 0.15])))
    tids = [mesh.add_vt((p[0] + p[1], p[2])) for p in pts]
    for a, b, c in faces:
        mesh.tri((ids[a], ids[b], ids[c]), (tids[a], tids[b], tids[c]), (nids[a], nids[b], nids[c]))


def add_card(mesh: Mesh, base, up, side, width: float, height: float, tile: int,
             normal_fn) -> None:
    """One leaf card: a quad rising from ``base`` along ``up``, ``width`` wide.

    ``tile`` picks the quarter of the 2x2 atlas (0..3); the bottom of the tile
    is the twig end, at ``base``. ``normal_fn(point)`` gives each corner's
    (bent) normal.
    """
    base, up, side = (np.asarray(x, float) for x in (base, up, side))
    corners = [base - side * width / 2, base + side * width / 2,
               base + side * width / 2 + up * height, base - side * width / 2 + up * height]
    u0, v0 = (tile % 2) * 0.5, (tile // 2) * 0.5
    pad = 0.004
    uvs = [(u0 + pad, v0 + pad), (u0 + 0.5 - pad, v0 + pad),
           (u0 + 0.5 - pad, v0 + 0.5 - pad), (u0 + pad, v0 + 0.5 - pad)]
    ids = [mesh.add_v(c) for c in corners]
    tids = [mesh.add_vt(uv) for uv in uvs]
    nids = [mesh.add_vn(normal_fn(c)) for c in corners]
    mesh.tri((ids[0], ids[1], ids[2]), (tids[0], tids[1], tids[2]), (nids[0], nids[1], nids[2]))
    mesh.tri((ids[0], ids[2], ids[3]), (tids[0], tids[2], tids[3]), (nids[0], nids[2], nids[3]))


def write_obj(mesh: Mesh, path: Path, name: str) -> Path:
    """Write ``mesh`` as OBJ in centimetres, one object, with v/vt/vn faces."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Red Rocks vegetation, generated by redrocks.vegetation (no downloaded assets)",
             "o %s" % name]
    lines += ["v %.2f %.2f %.2f" % (x * 100, y * 100, z * 100) for x, y, z in mesh.v]
    lines += ["vt %.5f %.5f" % uv for uv in mesh.vt]
    lines += ["vn %.4f %.4f %.4f" % n for n in mesh.vn]
    for fv, ft, fn in mesh.f:
        lines.append("f " + " ".join("%d/%d/%d" % (a + 1, b + 1, c + 1) for a, b, c in zip(fv, ft, fn)))
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------- models

PARTS = ("Trunk", "Crown", "Leaves")


@dataclass
class Model:
    name: str
    species: str
    atlas: str
    height_m: float
    trunk_radius_m: float
    parts: dict  # part -> Mesh


def _seed(name: str, index: int) -> int:
    return zlib.crc32(("%s/%d" % (name, index)).encode())


def _card_frame(rng, outward, tilt_up: float):
    """(up, side) for a card whose twig end points back into the crown."""
    up = _unit(_unit(outward) + np.array([0.0, 0.0, tilt_up]) + rng.normal(scale=0.35, size=3))
    side = _unit(np.cross(up, rng.normal(size=3)))
    return up, side


def _bent(crown_c, crown_r, clump_c, w_crown: float, lift: float = 0.3):
    crown_c, clump_c = np.asarray(crown_c, float), np.asarray(clump_c, float)
    crown_r = np.asarray(crown_r, float)

    def fn(p):
        n = w_crown * _unit((p - crown_c) / crown_r) + (1 - w_crown) * _unit(p - clump_c)
        return n + np.array([0.0, 0.0, lift])
    return fn


def _clump(meshes, rng, centre, core_r, n_cards, card_size, crown_c, crown_r, w_crown,
           subdiv=1, tilt_up=0.3, squash=0.8, spread=0.35):
    """Core blob plus leaf cards around it."""
    centre = np.asarray(centre, float)
    add_blob(meshes["Crown"], centre, core_r * np.array([1.0, 1.0, squash]) * rng.uniform(0.85, 1.15, 3),
             rng, subdiv=subdiv, bend_to=crown_c, bend=0.55)
    outward = centre - np.asarray(crown_c, float)
    outward[2] *= 0.5
    normal_fn = _bent(crown_c, crown_r, centre, w_crown)
    for _ in range(n_cards):
        up, side = _card_frame(rng, outward + rng.normal(scale=0.6 * np.linalg.norm(outward) + 0.1, size=3),
                               tilt_up)
        size = card_size * rng.uniform(0.8, 1.2)
        base = centre - up * size * 0.45 + rng.normal(scale=core_r * spread, size=3)
        add_card(meshes["Leaves"], base, up, side, size, size, int(rng.integers(0, 4)), normal_fn)


def _wobbly_path(rng, start, direction, length, points, wobble, curve_up=0.0, droop=0.0):
    start = np.asarray(start, float)
    d = _unit(direction)
    side = _perpendicular(d)
    other = np.cross(d, side)
    pts = []
    for i in range(points):
        s = i / (points - 1)
        p = start + d * length * s
        p = p + np.array([0.0, 0.0, length * (curve_up * s ** 2.5 - droop * s * s * (1 - s) * 2.0)])
        if 0 < i:
            p = p + (side * rng.normal() + other * rng.normal()) * wobble * s
        pts.append(p)
    return np.asarray(pts)


def _juniper(name, index, one_seed: bool) -> Model:
    rng = np.random.default_rng(_seed(name, index))
    meshes = {p: Mesh() for p in PARTS}
    if one_seed:
        height, n_stems, base_r, zb = 3.6, int(rng.integers(2, 5)), 0.10, 0.04
        radius, shape_pow, n_clumps, core, card = 0.55 * 3.6, 0.6, 56, 0.3, 0.5
    else:
        height, n_stems, base_r, zb = 5.0, int(rng.integers(1, 3)), 0.17, 0.10
        radius, shape_pow, n_clumps, core, card = 0.34 * 5.0, 0.85, 64, 0.32, 0.52
    lean = rng.normal(scale=0.12 * height, size=2)
    sin_w = rng.normal(size=(3, 3))
    sin_p = rng.uniform(0, 2 * math.pi, 3)

    def envelope(t, theta):
        shape = math.sin(math.pi * min(max(t, 0.0), 1.0) ** shape_pow) ** 0.7
        if not one_seed:
            shape *= 1.0 - 0.45 * t  # Rocky Mountain: broad base, narrow top
        lumps = sum(0.12 * math.sin(sin_w[i, 0] * 3 * math.cos(theta) + sin_w[i, 1] * 3 * math.sin(theta)
                                    + sin_w[i, 2] * 4 * t + sin_p[i]) for i in range(3))
        return radius * shape * (1.0 + lumps)

    def axis(t):
        return np.array([lean[0] * t, lean[1] * t, zb * height + t * (1 - zb) * height])

    stems = []
    for s in range(n_stems):
        az = rng.uniform(0, 2 * math.pi)
        tilt = rng.uniform(0.05, 0.25) if n_stems == 1 else rng.uniform(0.25, 0.55)
        direction = np.array([math.cos(az) * tilt + lean[0] / height, math.sin(az) * tilt + lean[1] / height, 1.0])
        start = np.array([math.cos(az), math.sin(az), 0.0]) * rng.uniform(0.0, 0.15) + [0, 0, -0.45]
        length = height * rng.uniform(0.6, 0.85)
        path = _wobbly_path(rng, start, direction, length, 9, 0.18)
        r0 = base_r * rng.uniform(0.8, 1.15)
        radii = [r0 * (1.0 - 0.8 * (i / 8)) ** 1.1 + 0.02 for i in range(9)]
        radii[0] *= 1.35  # root flare
        add_tube(meshes["Trunk"], path, radii, 9, gnarl=0.12, rng=rng, uv_scale=1.0)
        stems.append(path)

    crown_c = axis(0.5)
    crown_r = np.array([radius, radius, height * 0.5])
    placed = 0
    attempts = 0
    while placed < n_clumps and attempts < n_clumps * 20:
        attempts += 1
        t = rng.uniform(0.02, 1.0) ** 0.9
        theta = rng.uniform(0, 2 * math.pi)
        rho = 0.5 + 0.5 * math.sqrt(rng.uniform())
        r = envelope(t, theta)
        if r < 0.25:
            continue
        # Holes: a third of the crown's volume is open, so light gets in.
        hole = math.sin(2.1 * math.cos(theta) * 3 + t * 5 + sin_p[0]) * math.sin(theta * 2 + t * 3.3 + sin_p[1])
        if hole > 0.55:
            continue
        p = axis(t) + np.array([math.cos(theta) * rho * r, math.sin(theta) * rho * r, 0.0])
        scale = 0.65 + 0.35 * min(1.0, r / radius * 1.6)
        _clump(meshes, rng, p, core * scale, 16, card * scale, crown_c, crown_r, 0.6,
               subdiv=1, tilt_up=0.2, squash=0.85, spread=1.0)
        placed += 1
        if rng.uniform() < 0.55:
            stem = stems[int(rng.integers(0, len(stems)))]
            k = int(np.argmin(np.abs(stem[:, 2] - p[2] * 0.75)))
            start = stem[max(1, min(k, len(stem) - 2))]
            path = _wobbly_path(rng, start, p - start, float(np.linalg.norm(p - start)) * 0.9, 4, 0.12)
            add_tube(meshes["Trunk"], path, [0.05, 0.035, 0.022, 0.012], 5, gnarl=0.1, rng=rng)
    return Model("%s_%d" % (name, index), "juniper_one_seed" if one_seed else "juniper_rocky_mountain",
                 "Juniper", height, base_r, meshes)


def _ponderosa(name, index) -> Model:
    rng = np.random.default_rng(_seed(name, index))
    meshes = {p: Mesh() for p in PARTS}
    height = 18.0 * rng.uniform(0.92, 1.08)
    base_r = 0.38 * rng.uniform(0.9, 1.1)
    k = 15
    sway = rng.normal(scale=0.25, size=(2,))
    trunk = []
    for i in range(k):
        t = i / (k - 1)
        trunk.append([sway[0] * t * t + rng.normal(scale=0.04) * (0 < i < k - 1),
                      sway[1] * t * t + rng.normal(scale=0.04) * (0 < i < k - 1),
                      -0.5 + t * (height + 0.5)])
    trunk = np.asarray(trunk)
    radii = [base_r * (1 - i / (k - 1)) ** 0.85 + 0.035 for i in range(k)]
    radii[0] *= 1.3
    add_tube(meshes["Trunk"], trunk, radii, 14, gnarl=0.04, rng=rng, uv_scale=1.0)

    def trunk_at(z):
        return np.array([np.interp(z, trunk[:, 2], trunk[:, 0]), np.interp(z, trunk[:, 2], trunk[:, 1]), z])

    crown_base = height * rng.uniform(0.42, 0.55)
    # Dead stubs on the clear lower trunk: ponderosas self-prune.
    for _ in range(int(rng.integers(4, 8))):
        z = rng.uniform(0.2 * height, crown_base)
        az = rng.uniform(0, 2 * math.pi)
        d = np.array([math.cos(az), math.sin(az), rng.uniform(-0.5, 0.05)])
        path = _wobbly_path(rng, trunk_at(z), d, rng.uniform(0.3, 0.9), 3, 0.05)
        add_tube(meshes["Trunk"], path, [0.045, 0.03, 0.012], 5, rng=rng)

    n_branches = int(rng.integers(34, 44))
    golden = math.pi * (3 - math.sqrt(5))
    az0 = rng.uniform(0, 2 * math.pi)
    for b in range(n_branches):
        f = (b + rng.uniform(0.0, 0.8)) / n_branches
        z = crown_base + f * (height * 0.97 - crown_base)
        if rng.uniform() < 0.12:
            continue  # gaps in the whorls
        az = az0 + b * golden + rng.normal(scale=0.3)
        tt = (z - crown_base) / (height - crown_base)
        length = 3.6 * (1 - 0.7 * tt ** 1.5) * rng.uniform(0.7, 1.2)
        rise = -0.12 + 0.6 * tt + rng.normal(scale=0.08)
        d = np.array([math.cos(az), math.sin(az), rise])
        path = _wobbly_path(rng, trunk_at(z), d, length, 6, 0.18, curve_up=0.18, droop=0.35 * (1 - tt))
        r0 = 0.08 * (length / 3.6) ** 0.8 + 0.02
        add_tube(meshes["Trunk"], path, [r0 * (1 - 0.8 * i / 5) + 0.008 for i in range(6)], 6, rng=rng)
        crown_c = trunk_at(z)
        n_tufts = int(rng.integers(5, 8)) + (length > 2.5)
        for i in range(n_tufts):
            s = 0.3 + 0.7 * (i + rng.uniform(0, 0.6)) / n_tufts
            s = min(s, 1.0)
            p = path[0] + (path[-1] - path[0]) * s
            # Follow the drooping path, not the chord.
            idx = min(int(s * (len(path) - 1)), len(path) - 2)
            frac = s * (len(path) - 1) - idx
            p = path[idx] * (1 - frac) + path[idx + 1] * frac
            p = p + rng.normal(scale=0.35, size=3) + np.array([0.0, 0.0, 0.2])
            _clump(meshes, rng, p, 0.36, 7, 1.15, crown_c, np.array([1.5, 1.5, 1.0]), 0.3,
                   subdiv=1, tilt_up=0.9, squash=0.75, spread=0.6)
    top = trunk_at(height)
    for _ in range(4):
        p = top + rng.normal(scale=0.35, size=3) - np.array([0.0, 0.0, 0.4])
        _clump(meshes, rng, p, 0.3, 5, 1.0, trunk_at(height - 1.5), np.array([1.2, 1.2, 1.2]), 0.3,
               subdiv=1, tilt_up=1.2, squash=0.8)
    return Model("%s_%d" % (name, index), "ponderosa_pine", "Pine", height, base_r, meshes)


def _shrub(name, index, mahogany: bool) -> Model:
    rng = np.random.default_rng(_seed(name, index))
    meshes = {p: Mesh() for p in PARTS}
    if mahogany:
        height, n_stems, n_clumps, core, card, atlas = 1.9, int(rng.integers(3, 6)), 20, 0.15, 0.6, "Mahogany"
    else:
        height, n_stems, n_clumps, core, card, atlas = 1.6, int(rng.integers(5, 9)), 28, 0.2, 0.55, "Oak"
    stems = []
    for _ in range(n_stems):
        az = rng.uniform(0, 2 * math.pi)
        tilt = rng.uniform(0.2, 0.7)
        start = np.array([math.cos(az), math.sin(az), 0.0]) * rng.uniform(0.0, 0.25) + [0, 0, -0.2]
        d = np.array([math.cos(az) * tilt, math.sin(az) * tilt, 1.0])
        path = _wobbly_path(rng, start, d, height * rng.uniform(0.7, 1.0), 5, 0.08)
        add_tube(meshes["Trunk"], path, [0.04, 0.032, 0.024, 0.016, 0.009], 5, rng=rng)
        stems.append(path)
    crown_c = np.array([0.0, 0.0, 0.55 * height])
    crown_r = np.array([0.62, 0.58, 0.48]) * height * rng.uniform(0.9, 1.15, 3)
    for _ in range(n_clumps):
        d = _unit(rng.normal(size=3))
        d[2] = abs(d[2]) * 0.9 - 0.15
        p = crown_c + d * crown_r * rng.uniform(0.55, 1.0)
        p[2] = max(p[2], 0.25 * height)
        _clump(meshes, rng, p, core * rng.uniform(0.8, 1.2), 8, card, crown_c,
               crown_r, 0.65, subdiv=1, tilt_up=0.4, squash=0.8, spread=0.7)
    return Model("%s_%d" % (name, index), "mountain_mahogany" if mahogany else "gambel_oak",
                 atlas, height, 0.04, meshes)


# name -> (builder, variant count)
SPECIES = {
    "JuniperRM": (lambda i: _juniper("JuniperRM", i, one_seed=False), 3),
    "JuniperOS": (lambda i: _juniper("JuniperOS", i, one_seed=True), 3),
    "Ponderosa": (lambda i: _ponderosa("Ponderosa", i), 4),
    "Oak": (lambda i: _shrub("Oak", i, mahogany=False), 3),
    "Mahogany": (lambda i: _shrub("Mahogany", i, mahogany=True), 3),
}


def build_model(species: str, index: int) -> Model:
    builder, count = SPECIES[species]
    if not 0 <= index < count:
        raise ValueError("%s has %d variants" % (species, count))
    return builder(index)


def all_models() -> list[Model]:
    return [build_model(s, i) for s, (_, n) in SPECIES.items() for i in range(n)]


# ------------------------------------------------------------------- textures

ATLASES = ("Juniper", "Pine", "Oak", "Mahogany")
ATLAS_SIZE = 1024


def _hex(c, jitter, rng):
    h = c.lstrip("#")
    rgb = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)
    rgb = rgb * rng.uniform(1 - jitter, 1 + jitter) + rng.normal(scale=jitter * 30, size=3)
    return tuple(int(v) for v in np.clip(rgb, 0, 255)) + (255,)


def _paint_juniper(draw, rng, s):
    """A juniper spray: a twig of rope-like branchlets of overlapping scales."""
    def cord(p0, angle, length, width, depth):
        steps = max(3, int(length / (width * 0.7)))
        x, y = p0
        for i in range(steps):
            a = angle + rng.normal(scale=0.08)
            x += math.cos(a) * length / steps
            y -= math.sin(a) * length / steps
            r = width * (1.0 - 0.45 * i / steps)
            col = _hex(rng.choice(["#6F8A78", "#7E9886", "#5D7767", "#8DA294", "#667F62"]), 0.1, rng)
            draw.ellipse([x - r, y - r * 1.25, x + r, y + r * 1.25], fill=col)
            if depth > 0 and i % 2 == 1 and i < steps - 1:
                side = 1 if (i // 2) % 2 else -1
                cord((x, y), angle + side * rng.uniform(0.45, 0.8), length * rng.uniform(0.3, 0.42),
                     width * 0.8, depth - 1)
        return x, y

    draw.line([(s * 0.5, s * 0.99), (s * 0.5, s * 0.8)], fill=(92, 70, 50, 255), width=int(s * 0.02))
    cord((s * 0.5, s * 0.82), math.pi / 2 + rng.normal(scale=0.1), s * 0.62, s * 0.012, 3)
    for _ in range(int(rng.integers(2, 6))):
        x, y = rng.uniform(0.25, 0.75) * s, rng.uniform(0.15, 0.6) * s
        r = s * 0.016
        draw.ellipse([x - r, y - r, x + r, y + r], fill=_hex("#8EA3BF", 0.06, rng))


def _paint_pine(draw, rng, s):
    """A ponderosa tuft: a twig ending in a brush of long needles in threes."""
    tip = (s * 0.5 + rng.normal(scale=s * 0.02), s * 0.5)
    draw.line([(s * 0.5, s * 0.99), tip], fill=(96, 66, 42, 255), width=int(s * 0.03))
    for _ in range(170):
        f = rng.uniform(0.0, 1.0)
        x0 = s * 0.5 + (tip[0] - s * 0.5) * (0.35 + 0.65 * f)
        y0 = s * 0.99 + (tip[1] - s * 0.99) * (0.35 + 0.65 * f)
        a = math.pi / 2 + rng.normal(scale=0.55 + 0.25 * (1 - f))
        length = s * rng.uniform(0.3, 0.48)
        col = _hex(rng.choice(["#4C6A2E", "#5E7C35", "#6C8A3E", "#3F5A27"]), 0.1, rng)
        bend = rng.normal(scale=0.12)
        pts = [(x0 + math.cos(a + bend * t) * length * t, y0 - math.sin(a + bend * t) * length * t)
               for t in np.linspace(0, 1, 6)]
        draw.line(pts, fill=col, width=max(2, int(s * 0.007)))
    r = s * 0.02
    draw.ellipse([tip[0] - r, tip[1] - r * 1.5, tip[0] + r, tip[1] + r], fill=(130, 92, 60, 255))


def _paint_leaves(draw, rng, s, mahogany: bool):
    """Oak: lobed leaves along forked twigs. Mahogany: small narrow leaves in
    tufts and a few of its feathery, curling seed plumes."""
    twig = (104, 84, 70, 255) if mahogany else (88, 64, 46, 255)
    palette = (["#6C7556", "#7C8466", "#5D664A", "#8A8E70"] if mahogany
               else ["#4F6B2B", "#5E7A33", "#6B7F3A", "#47602A", "#7D8640"])

    def leaf(x, y, a, size):
        col = _hex(rng.choice(palette), 0.1, rng)
        if mahogany:
            pts = []
            for t in np.linspace(0, 2 * math.pi, 14):
                px, py = math.cos(t) * size, math.sin(t) * size * 0.32
                pts.append((x + px * math.cos(a) - py * math.sin(a), y - (px * math.sin(a) + py * math.cos(a))))
            draw.polygon(pts, fill=col)
        else:
            for k in range(5):  # lobes along the midrib
                t = (k + 0.5) / 5
                cx, cy = x + math.cos(a) * size * t * 1.6, y - math.sin(a) * size * t * 1.6
                r = size * (0.33 + 0.25 * math.sin(math.pi * t))
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)

    def twig_fn(p0, angle, length, width, depth):
        x, y = p0
        steps = 6
        for i in range(steps):
            a = angle + rng.normal(scale=0.12)
            nx, ny = x + math.cos(a) * length / steps, y - math.sin(a) * length / steps
            draw.line([(x, y), (nx, ny)], fill=twig, width=max(2, int(width)))
            x, y = nx, ny
            if i >= 1:
                n = 5 if mahogany else 3
                for _ in range(n):
                    leaf(x, y, a + rng.choice([-1, 1]) * rng.uniform(0.4, 1.2),
                         s * (0.07 if mahogany else 0.065) * rng.uniform(0.75, 1.2))
            if depth > 0 and i in (2, 4):
                twig_fn((x, y), angle + rng.choice([-1, 1]) * rng.uniform(0.4, 0.8), length * 0.55,
                        width * 0.7, depth - 1)

    twig_fn((s * 0.5, s * 0.99), math.pi / 2 + rng.normal(scale=0.12), s * 0.68, s * 0.014, 2)
    if mahogany:
        for _ in range(int(rng.integers(2, 5))):
            x, y = rng.uniform(0.25, 0.75) * s, rng.uniform(0.1, 0.5) * s
            pts = []
            for t in np.linspace(0, 1, 16):
                pts.append((x + s * 0.12 * t + s * 0.02 * math.sin(t * 9),
                            y - s * 0.1 * t + s * 0.02 * math.cos(t * 9)))
            draw.line(pts, fill=(226, 218, 204, 255), width=max(2, int(s * 0.006)))


def paint_atlas(atlas: str, size: int = ATLAS_SIZE, supersample: int = 2):
    """RGBA uint8 array (size, size, 4): the species' 2x2 leaf-card atlas.

    Each quarter is one spray, twig at the bottom of the tile. Transparent
    texels carry the blurred colour of their neighbours, so mips and filtering
    never bleed black into the edges.
    """
    from PIL import Image, ImageDraw, ImageFilter

    if atlas not in ATLASES:
        raise ValueError(atlas)
    rng = np.random.default_rng(zlib.crc32(atlas.encode()))
    tile = size // 2 * supersample
    img = Image.new("RGBA", (tile * 2, tile * 2), (0, 0, 0, 0))
    for t in range(4):
        layer = Image.new("RGBA", (tile, tile), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        if atlas == "Juniper":
            _paint_juniper(draw, rng, tile)
        elif atlas == "Pine":
            _paint_pine(draw, rng, tile)
        else:
            _paint_leaves(draw, rng, tile, atlas == "Mahogany")
        # Tile t sits at UV (t % 2, t // 2) quarters; image rows run top-down
        # while V runs bottom-up, so tile row 0 is the bottom half of the image.
        img.paste(layer, ((t % 2) * tile, (1 - t // 2) * tile))
    # Resample premultiplied so the transparent black never darkens the edges.
    img = img.convert("RGBa").resize((size, size), Image.LANCZOS).convert("RGBA")
    rgba = np.asarray(img).astype(np.float32)
    alpha = rgba[..., 3:4] / 255.0
    pre = Image.fromarray(np.clip(rgba[..., :3] * alpha, 0, 255).astype(np.uint8))
    a_img = Image.fromarray((alpha[..., 0] * 255).astype(np.uint8))
    blur = np.asarray(pre.filter(ImageFilter.GaussianBlur(24))).astype(np.float32)
    blur_a = np.asarray(a_img.filter(ImageFilter.GaussianBlur(24))).astype(np.float32)[..., None] / 255.0
    mean = (rgba[..., :3] * alpha).sum((0, 1)) / max(float(alpha.sum()), 1e-3)
    fill = np.where(blur_a > 1e-3, blur / np.maximum(blur_a, 1e-3), mean)
    rgb = np.where(alpha > 0.02, rgba[..., :3], fill)
    out = np.concatenate([np.clip(rgb, 0, 255), rgba[..., 3:4]], axis=2)
    return out.astype(np.uint8)


# ------------------------------------------------------------------ placement

# The masks below are the ones setup_environment.py used for the old scrub.
SCAN_CENTRE_CM = (-3784.0, 5263.0)  # the ml0w scan covers the bowl; it has its own plants
SCAN_CLEAR_CM = 33000.0
BLOCK_OBJS = {"Roads": 1, "Paths": 1, "Parking": 1, "Benches": 2, "Stage": 2}  # name -> dilation (cells)
BLOCK_CELL_CM = 800.0
SPACING_CM = 650.0
SEED = 1941  # the year Red Rocks Amphitheatre opened
MAX_TILT_DEG = 4.0

# Steepest ground (smallest normal.z) each grows on; steeper is cliff.
MIN_NZ = {"juniper": 0.76, "pine": 0.8, "shrub": 0.84}
# Instance scale ranges against the model's own height.
SCALE = {"JuniperRM": (0.62, 1.2), "JuniperOS": (0.6, 1.25), "Ponderosa": (0.58, 1.35),
         "Oak": (0.65, 1.3), "Mahogany": (0.6, 1.1)}


def read_obj_vertices(path) -> np.ndarray:
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                rows.append(line.split()[1:4])
    return np.asarray(rows, float).reshape(-1, 3)


class Ground:
    """The square terrain grid as an array: bilinear height and normals, cm."""

    def __init__(self, verts: np.ndarray):
        xs = np.unique(np.round(verts[:, 0], 3))
        ys = np.unique(np.round(verts[:, 1], 3))
        self.x0, self.y0 = xs[0], ys[0]
        self.nx, self.ny = len(xs), len(ys)
        if self.nx * self.ny != len(verts):
            raise ValueError("terrain is not a full grid")
        self.dx = (xs[-1] - xs[0]) / (self.nx - 1)
        self.dy = (ys[-1] - ys[0]) / (self.ny - 1)
        self.z = np.zeros((self.ny, self.nx))
        i = np.rint((verts[:, 0] - self.x0) / self.dx).astype(int)
        j = np.rint((verts[:, 1] - self.y0) / self.dy).astype(int)
        self.z[j, i] = verts[:, 2]

    @property
    def extent(self):
        return (self.x0, self.y0, self.x0 + self.dx * (self.nx - 1), self.y0 + self.dy * (self.ny - 1))

    def contains(self, x, y):
        x0, y0, x1, y1 = self.extent
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)

    def height(self, x, y):
        fx = np.clip((np.asarray(x) - self.x0) / self.dx, 0, self.nx - 1.000001)
        fy = np.clip((np.asarray(y) - self.y0) / self.dy, 0, self.ny - 1.000001)
        i, j = fx.astype(int), fy.astype(int)
        tx, ty = fx - i, fy - j
        z = self.z
        a = z[j, i] * (1 - tx) + z[j, i + 1] * tx
        b = z[j + 1, i] * (1 - tx) + z[j + 1, i + 1] * tx
        return a * (1 - ty) + b * ty

    def normal(self, x, y):
        gx = (self.height(x + self.dx, y) - self.height(x - self.dx, y)) / (2 * self.dx)
        gy = (self.height(x, y + self.dy) - self.height(x, y - self.dy)) / (2 * self.dy)
        inv = 1.0 / np.sqrt(1.0 + gx * gx + gy * gy)
        return -gx * inv, -gy * inv, inv


def _hash01(i, j, seed):
    h = (i.astype(np.int64) * 374761393 + j.astype(np.int64) * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFF) / float(0xFFFFFF)


def value_noise(x, y, cell, seed):
    """Smooth 0..1 value noise with features about ``cell`` across (vectorised)."""
    fx, fy = np.asarray(x, float) / cell, np.asarray(y, float) / cell
    i, j = np.floor(fx), np.floor(fy)
    tx, ty = fx - i, fy - j
    tx, ty = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
    i, j = i.astype(np.int64), j.astype(np.int64)
    a = _hash01(i, j, seed) * (1 - tx) + _hash01(i + 1, j, seed) * tx
    b = _hash01(i, j + 1, seed) * (1 - tx) + _hash01(i + 1, j + 1, seed) * tx
    return a * (1 - ty) + b * ty


def blocked_cells(verts: np.ndarray, cell: float = BLOCK_CELL_CM, dilate: int = 1) -> set:
    out = set()
    if len(verts) == 0:
        return out
    cells = np.unique(np.floor(verts[:, :2] / cell).astype(np.int64), axis=0)
    for di in range(-dilate, dilate + 1):
        for dj in range(-dilate, dilate + 1):
            out.update(map(tuple, (cells + [di, dj]).tolist()))
    return out


def _smooth(lo, hi, x):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def place(ground: Ground, blocked: set, models: dict, seed: int = SEED) -> dict:
    """Instances by model name: [[x, y, z, yaw, pitch, roll, scale_xy, scale_z], ...] (cm, deg).

    ``models`` maps model name to its Model (for trunk radius and species).
    Deterministic for a seed.
    """
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = ground.extent
    gx = np.arange(x0, x1, SPACING_CM)
    gy = np.arange(y0, y1, SPACING_CM)
    xx, yy = np.meshgrid(gx, gy)
    x = (xx + rng.uniform(0, SPACING_CM, xx.shape)).ravel()
    y = (yy + rng.uniform(0, SPACING_CM, yy.shape)).ravel()
    keep_roll = rng.uniform(size=x.shape)
    pick_roll = rng.uniform(size=x.shape)
    ok = ground.contains(x, y)
    ok &= np.hypot(x - SCAN_CENTRE_CM[0], y - SCAN_CENTRE_CM[1]) >= SCAN_CLEAR_CM
    if blocked:
        cells = np.floor(np.stack([x, y], 1) / BLOCK_CELL_CM).astype(np.int64)
        ok &= np.array([tuple(c) not in blocked for c in cells.tolist()], bool)
    z = ground.height(x, y)
    nx, ny, nz = ground.normal(x, y)
    zmin, zmax = float(ground.z.min()), float(ground.z.max())
    e = (z - zmin) / max(zmax - zmin, 1.0)
    slope = np.hypot(nx, ny)
    north = np.where(slope > 1e-6, ny / np.maximum(slope, 1e-6), 0.0) * np.clip(slope / 0.3, 0, 1)
    thicket = 0.75 * value_noise(x, y, 9000.0, seed) + 0.25 * value_noise(x, y, 2200.0, seed + 1)
    meadow = value_noise(x, y, 25000.0, seed + 2)
    density = np.clip((thicket - 0.25) * 2.0, 0, 1) * (0.45 + 0.55 * _smooth(0.2, 0.55, meadow))

    rocky = _smooth(0.97, 0.84, nz)
    w_pine = (_smooth(0.3, 0.55, e) * (1 + 1.4 * north) + 0.5 * _smooth(0.12, 0.3, e) * np.clip(north, 0, 1))
    w_pine = np.clip(w_pine, 0, None) * 0.55
    w_jun = (1 - 0.65 * _smooth(0.5, 0.8, e)) * (1 - 0.4 * north) * (0.6 + 0.9 * rocky)
    w_shrub = (1.5 - 1.2 * _smooth(0.15, 0.45, e)) * (0.6 + 0.7 * _smooth(0.86, 0.97, nz)) * (1 + 0.3 * north)
    w_pine = np.where(nz >= MIN_NZ["pine"], w_pine, 0)
    w_jun = np.where(nz >= MIN_NZ["juniper"], w_jun, 0)
    w_shrub = np.where(nz >= MIN_NZ["shrub"], w_shrub, 0)
    total = w_pine + w_jun + w_shrub
    ok &= total > 0
    ok &= keep_roll < density * 0.85
    p_pine = w_pine / np.maximum(total, 1e-9)
    p_jun = w_jun / np.maximum(total, 1e-9)

    by_species = {}
    for name in sorted(models):
        by_species.setdefault(name.rsplit("_", 1)[0], []).append(name)
    out = {name: [] for name in models}

    def add(model_name, px, py, tilt):
        m = models[model_name]
        lo, hi = SCALE[model_name.rsplit("_", 1)[0]]
        s = lo + (hi - lo) * rng.beta(2.0, 2.0)
        n = ground.normal(np.array([px]), np.array([py]))
        steep = math.sqrt(max(0.0, 1 - float(n[2][0]) ** 2)) / max(float(n[2][0]), 0.3)
        pz = float(ground.height(np.array([px]), np.array([py]))[0])
        sink = 12.0 + m.trunk_radius_m * 100.0 * s * steep * 1.3
        out[model_name].append([round(px, 1), round(py, 1), round(pz - sink, 1),
                                round(float(rng.uniform(0, 360)), 2),
                                round(float(rng.normal(scale=tilt)), 2), round(float(rng.normal(scale=tilt)), 2),
                                round(s * rng.uniform(0.92, 1.08), 3), round(s, 3)])

    for k in np.flatnonzero(ok):
        r = pick_roll[k]
        if r < p_pine[k]:
            add(by_species["Ponderosa"][int(rng.integers(len(by_species["Ponderosa"])))], x[k], y[k], 0.8)
        elif r < p_pine[k] + p_jun[k]:
            rm = rng.uniform() < 0.2 + 0.6 * _smooth(0.15, 0.5, e[k])
            sp = "JuniperRM" if rm else "JuniperOS"
            add(by_species[sp][int(rng.integers(len(by_species[sp])))], x[k], y[k], MAX_TILT_DEG)
        else:
            sp = "Mahogany" if rng.uniform() < 0.25 + 0.5 * rocky[k] else "Oak"
            # Scrub oak grows in clonal thickets: a few stems around the spot.
            for _ in range(int(rng.integers(1, 4)) if sp == "Oak" else int(rng.integers(1, 3))):
                px = x[k] + rng.normal(scale=180.0)
                py = y[k] + rng.normal(scale=180.0)
                if not bool(ground.contains(np.array([px]), np.array([py]))[0]):
                    continue
                add(by_species[sp][int(rng.integers(len(by_species[sp])))], px, py, MAX_TILT_DEG)
    return {name: rows for name, rows in out.items()}


# ---------------------------------------------------------------------- build

def export(out_dir: Path, terrain_obj: Path | None = None, avoid_dir: Path | None = None,
           seed: int = SEED, textures: bool = True) -> dict:
    """Write every model part, the atlases and vegetation.json to ``out_dir``.

    With ``terrain_obj``, instances are placed on it, avoiding the OBJs named in
    ``BLOCK_OBJS`` found in ``avoid_dir``. Returns the JSON written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = {m.name: m for m in all_models()}
    info = {"generator": "redrocks.vegetation", "units": "cm, X east, Y north, Z up",
            "models": [], "atlases": {}, "instances": {}, "counts": {}}
    for m in models.values():
        parts = {}
        for part, mesh in m.parts.items():
            stem = "Veg_%s_%s" % (m.name, part)
            write_obj(mesh, out_dir / (stem + ".obj"), stem)
            parts[part] = {"obj": stem + ".obj", "triangles": mesh.triangles}
        info["models"].append({"name": m.name, "species": m.species, "atlas": m.atlas,
                               "height_m": round(m.height_m, 3), "parts": parts})
    if textures:
        from PIL import Image
        for atlas in ATLASES:
            name = "T_Leaves_%s" % atlas
            Image.fromarray(paint_atlas(atlas, ATLAS_SIZE), "RGBA").save(out_dir / (name + ".png"))
            info["atlases"][atlas] = name + ".png"
    if terrain_obj is not None:
        ground = Ground(read_obj_vertices(terrain_obj))
        blocked = set()
        for name, dilate in BLOCK_OBJS.items():
            path = Path(avoid_dir or Path(terrain_obj).parent) / (name + ".obj")
            if path.exists():
                blocked |= blocked_cells(read_obj_vertices(path), dilate=dilate)
        info["instances"] = place(ground, blocked, models, seed=seed)
        counts = {}
        for name, rows in info["instances"].items():
            sp = name.rsplit("_", 1)[0]
            counts[sp] = counts.get(sp, 0) + len(rows)
        info["counts"] = counts
    (out_dir / "vegetation.json").write_text(json.dumps(info) + "\n")
    return info
