"""Can you see Denver from the top row? Line of sight, and a check picture.

Reads a build (``manifest.json``, ``RedRocks_Terrain.obj`` and ``Denver.obj``
in ``--out``) and, from the top walkway's eye (the manifest's
``stage.walkway_*``, 1.7 m up):

* rebuilds the terrain skirt with ``unreal/setup_environment.py``'s own
  functions (``grid_border``, ``skirt_mesh``), so it checks the skirt the next
  ``rebuild_level.py`` will import rather than an old one;
* fetches once (cached in ``<out>/cache``) USGS 3DEP DEMs of the real ground
  around the venue and from it to past Denver (``dem.cached_dem``);
* marches the sight line from the eye to each of the tallest towers over the
  game's ground (the terrain mesh, then the skirt) and over the real ground (the
  DEMs, every point dropped for curvature and refraction as Denver.obj is) and
  prints how many metres of each tower stand clear of the
  ground in front of it, and how far its top stands above (or below) the
  horizon behind it: in the game that is the skirt or, past it, the limb of the
  SkyAtmosphere's planet, whose top is the level's Z = 0;
* renders what that eye sees looking at the stage, 60 degrees wide: the game
  above, the real ground below, each with a 6-degree zoom on downtown, to
  ``<out>/check_denver.png``.

    PYTHONPATH=src .venv/bin/python tools/check_denver.py --out out

Exits 1 when the tallest tower shows less than ``--min-visible-m`` in the game.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from redrocks import denver  # noqa: E402
from redrocks.dem import BBox, cached_dem  # noqa: E402
from redrocks.geo import WGS84_A, WGS84_E2, LocalProjection, metres_per_degree  # noqa: E402

EYE_M = 1.7
# Real ground for the picture and the sight lines, from 3DEP: 14 km around the
# venue at ~10 m (the hogback, Green Mountain), and the 60-degree view from the
# top row out to 80 km at ~50 m, past the real horizon over the plains (~70 km).
REAL_NEAR_BBOX = BBox(west=-105.2875, south=39.6025, east=-105.1235, north=39.7285)
REAL_NEAR_SIZE = 1400
REAL_BBOX = BBox(west=-105.235, south=39.585, east=-104.270, north=40.250)
REAL_DEM_SIZE = 2048
REAL_FAR_M = 80000.0
GAME_FAR_M = 125000.0  # the skirt's outer ring is ~122 km out
# Unreal's SkyAtmosphere as setup_lighting.py leaves it: Earth's 6,360 km
# radius, the planet's top at the level's Z = 0 (the terrain minimum). Past the
# skirt its ground fills the view up to its limb.
SKY_PLANET_RADIUS_M = 6360000.0
HFOV_DEG = 60.0
ZOOM_FOV_DEG = 6.0
TOWERS = 6

SKY_TOP = np.array([104, 150, 206], dtype=float)
SKY_LOW = np.array([196, 214, 230], dtype=float)
HAZE = np.array([186, 196, 206], dtype=float)
NEAR_GROUND = np.array([150, 96, 70], dtype=float)   # red soil, sandstone
FAR_GROUND = np.array([150, 140, 104], dtype=float)  # olive-tan plains
CITY = np.array([170, 170, 165], dtype=float)        # M_DistantCity's base colour
PLANET = np.array([160, 172, 186], dtype=float)      # the SkyAtmosphere's hazy planet ground
SUN = (130.0, 58.0)  # setup_lighting.py: compass bearing, elevation


def load_env():
    """unreal/setup_environment.py with a stub ``unreal``: its skirt functions, no editor."""
    saved = sys.modules.get("unreal")
    sys.modules["unreal"] = types.ModuleType("unreal")
    try:
        spec = importlib.util.spec_from_file_location("setup_environment_check",
                                                      ROOT / "unreal" / "setup_environment.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if saved is None:
            del sys.modules["unreal"]
        else:
            sys.modules["unreal"] = saved


# ---- the game's ground ----------------------------------------------------------------

class Terrain:
    """RedRocks_Terrain.obj as a height function (absolute metres), NaN outside it."""

    def __init__(self, verts_cm: np.ndarray, base_m: float):
        xs = np.unique(np.round(verts_cm[:, 0], 2))
        n = len(xs)
        if n * n != len(verts_cm):
            raise ValueError("terrain OBJ is not a square grid")
        grid = verts_cm.reshape(n, n, 3)  # row 0 north, columns west to east (export_terrain_obj)
        self.h = grid[:, :, 2] / 100.0 + base_m
        self.half = float(grid[0, -1, 0]) / 100.0
        self.n = n

    def __call__(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        n, half = self.n, self.half
        fc = (x + half) / (2 * half) * (n - 1)
        fr = (half - y) / (2 * half) * (n - 1)
        inside = (fc >= 0) & (fc <= n - 1) & (fr >= 0) & (fr <= n - 1)
        c0 = np.clip(fc.astype(int), 0, n - 2)
        r0 = np.clip(fr.astype(int), 0, n - 2)
        tx, ty = np.clip(fc - c0, 0, 1), np.clip(fr - r0, 0, 1)
        a, b = self.h[r0, c0], self.h[r0, c0 + 1]
        d, e = self.h[r0 + 1, c0], self.h[r0 + 1, c0 + 1]
        z = np.where(ty >= tx, a + (e - d) * tx + (d - a) * ty, a + (b - a) * tx + (e - b) * ty)
        return np.where(inside, z, np.nan)


class Skirt:
    """setup_environment.skirt_mesh's surface as a height function (absolute metres).

    Every ring is the terrain's border scaled about its centre, so a point's
    Chebyshev distance gives its ring and its scaled-back position on the border
    gives the segment; then the quad's two triangles as the mesh splits them.
    """

    def __init__(self, border_cm, verts_cm, rings: int, base_m: float):
        b = np.asarray(border_cm, dtype=float)[:, :2] / 100.0
        v = np.asarray(verts_cm, dtype=float)
        self.n = n = len(b)
        self.xy = v[:, :2].reshape(rings, n, 2) / 100.0
        self.z = v[:, 2].reshape(rings, n) / 100.0 + base_m
        self.c = b.mean(axis=0)
        rel = b - self.c
        self.half = float(np.max(np.abs(rel)))
        self.grow = np.abs(self.xy[:, 0, :] - self.c).max(axis=1) / self.half
        self.s = self._perimeter(rel)
        if np.any(np.diff(self.s) <= 0):
            raise ValueError("border is not counter-clockwise from the south-west corner")

    def _perimeter(self, rel):
        h = self.half
        x, y = rel[:, 0], rel[:, 1]
        ax, ay = np.abs(x), np.abs(y)
        s = np.where((ay >= ax) & (y < 0), x + h, 0.0)
        s = np.where((ax > ay) & (x > 0), 2 * h + (y + h), s)
        s = np.where((ay >= ax) & (y > 0), 4 * h + (h - x), s)
        s = np.where((ax > ay) & (x < 0), 6 * h + (h - y), s)
        return s

    def __call__(self, x, y):
        q = np.stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)], axis=-1) - self.c
        shape = q.shape[:-1]
        q = q.reshape(-1, 2)
        g = np.abs(q).max(axis=1) / self.half
        k = np.searchsorted(self.grow, g, side="right") - 1
        ok = (k >= 0) & (k < len(self.grow) - 1) & (g > 0)
        k = np.clip(k, 0, len(self.grow) - 2)
        p = q / np.where(g > 0, g, 1.0)[:, None]
        s = self._perimeter(p)
        ext = np.concatenate([self.s, [8 * self.half]])
        i = np.clip(np.searchsorted(ext, s, side="right") - 1, 0, self.n - 1)
        j = (i + 1) % self.n
        A, B = self.xy[k, i], self.xy[k + 1, i]
        C, D = self.xy[k + 1, j], self.xy[k, j]
        zA, zB, zC, zD = self.z[k, i], self.z[k + 1, i], self.z[k + 1, j], self.z[k, j]
        qq = q + self.c
        # Which side of the A-C diagonal: triangle (A, B, C) or (A, C, D).
        side = (C[:, 0] - A[:, 0]) * (qq[:, 1] - A[:, 1]) - (C[:, 1] - A[:, 1]) * (qq[:, 0] - A[:, 0])
        side_b = (C[:, 0] - A[:, 0]) * (B[:, 1] - A[:, 1]) - (C[:, 1] - A[:, 1]) * (B[:, 0] - A[:, 0])
        use_b = np.sign(side) == np.sign(side_b)
        P = np.where(use_b[:, None], B, D)
        zP = np.where(use_b, zB, zD)
        z = _barycentric(qq, A, P, C, zA, zP, zC)
        return np.where(ok, z, np.nan).reshape(shape)


def _barycentric(q, a, b, c, za, zb, zc):
    v0, v1, v2 = b - a, c - a, q - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return za + v * (zb - za) + w * (zc - za)


# ---- the real ground --------------------------------------------------------------------

def to_latlon(proj: LocalProjection, east, north):
    """Invert proj.to_local(lat, lon, 0) for arrays: a few Newton steps from the flat guess."""
    east, north = np.asarray(east, dtype=float), np.asarray(north, dtype=float)
    mlon, mlat = metres_per_degree(proj.lat0)
    lat = proj.lat0 + north / mlat
    lon = proj.lon0 + east / mlon
    for _ in range(3):
        e, n = _to_local(proj, lat, lon)
        lat = lat + (north - n) / mlat
        lon = lon + (east - e) / mlon
    return lat, lon


def _to_local(proj: LocalProjection, lat, lon):
    phi, lam = np.radians(lat), np.radians(lon)
    s = np.sin(phi)
    nrad = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s)
    x = nrad * np.cos(phi) * np.cos(lam) - proj._origin[0]
    y = nrad * np.cos(phi) * np.sin(lam) - proj._origin[1]
    z = nrad * (1.0 - WGS84_E2) * s - proj._origin[2]
    e = x * proj._east[0] + y * proj._east[1] + z * proj._east[2]
    n = x * proj._north[0] + y * proj._north[1] + z * proj._north[2]
    return e, n


def sample_dem(dem, lat, lon):
    """Bilinear elevations of a Dem at arrays of lat/lon, NaN outside it."""
    bb, data = dem.bbox, dem.data
    rows, cols = data.shape
    fx = (lon - bb.west) / bb.width_deg * (cols - 1)
    fy = (bb.north - lat) / bb.height_deg * (rows - 1)
    inside = (fx >= 0) & (fx <= cols - 1) & (fy >= 0) & (fy <= rows - 1)
    x0 = np.clip(fx.astype(int), 0, cols - 2)
    y0 = np.clip(fy.astype(int), 0, rows - 2)
    tx, ty = np.clip(fx - x0, 0, 1), np.clip(fy - y0, 0, 1)
    z = (data[y0, x0] * (1 - tx) + data[y0, x0 + 1] * tx) * (1 - ty) + \
        (data[y0 + 1, x0] * (1 - tx) + data[y0 + 1, x0 + 1] * tx) * ty
    return np.where(inside, z, np.nan)


class RealGround:
    """Absolute elevation of the real ground from 3DEP (the fine DEM near the venue,
    the coarse one beyond), NaN outside both. Not the build's terrain: the park
    build's DEM comes back stretched north-south (see dem.fetch_dem)."""

    def __init__(self, near, far, proj: LocalProjection):
        self.near, self.far, self.proj = near, far, proj

    def __call__(self, x, y):
        lat, lon = to_latlon(self.proj, x, y)
        z = sample_dem(self.near, lat, lon)
        return np.where(np.isfinite(z), z, sample_dem(self.far, lat, lon))


# ---- sight lines -------------------------------------------------------------------------

def march(eye_xy, bearing_deg, distances, ground, drop: bool):
    """Ground elevations along a bearing from the eye (curvature-dropped when ``drop``)."""
    b = math.radians(bearing_deg)
    x = eye_xy[0] + distances * math.sin(b)
    y = eye_xy[1] + distances * math.cos(b)
    h = ground(x, y)
    if drop:
        h = h - denver.curvature_drop_m(np.hypot(x, y))
    return h


def tower_view(eye_xy, eye_z, block, ground, drop_ground: bool, far_m: float,
               horizon_floor: float = -math.inf):
    """(metres of the tower clear of the ground in front of it,
    metres of its top above the horizon behind it (negative: below), bearing, distance).

    ``horizon_floor`` is the lowest the horizon behind can be, as rise over run
    (the SkyAtmosphere's limb in the game).
    """
    cx, cy = block.plan.centroid.x, block.plan.centroid.y
    bearing, dist = denver.bearing_distance(eye_xy, (cx, cy))
    lowered = denver.curvature_drop_m(math.hypot(cx, cy))
    base, top = block.ground_m - lowered, block.top_m - lowered
    d = np.unique(np.concatenate([np.arange(2.0, 200.0, 1.0), np.geomspace(200.0, far_m, 6000)]))
    h = march(eye_xy, bearing, d, ground, drop_ground)
    clear = denver.visible_height_m(eye_z, d, h, dist, base, top)
    behind = (d > dist + 300.0) & np.isfinite(h)
    skyline = float(np.max((h[behind] - eye_z) / d[behind])) if behind.any() else -math.inf
    skyline = max(skyline, horizon_floor)
    return clear, top - (eye_z + skyline * dist), bearing, dist


def sky_limb(eye_z: float, planet_top_m: float) -> float:
    """Rise over run of the SkyAtmosphere planet's limb seen from ``eye_z``."""
    h = max(eye_z - planet_top_m, 0.0)
    return -math.tan(math.acos(SKY_PLANET_RADIUS_M / (SKY_PLANET_RADIUS_M + h)))


# ---- picture ----------------------------------------------------------------------------

def render(eye_xy, eye_z, yaw_deg, hfov_deg, size, horizon_row, ground, drop_ground, far_m,
           tris, tri_colour, near_box, limb=None):
    """RGB image (H, W, 3) of ground and Denver from a level pinhole camera.

    ``limb`` (rise over run) paints the SkyAtmosphere's planet ground below it
    where no geometry is, as the game's sky does past the skirt.
    """
    w, h = size
    f = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    cols = np.arange(w) + 0.5
    daz = np.arctan((cols - w / 2.0) / f)
    az = math.radians(yaw_deg) + daz
    d = np.geomspace(3.0, far_m, 2600)
    X = eye_xy[0] + d[:, None] * np.sin(az)[None, :]
    Y = eye_xy[1] + d[:, None] * np.cos(az)[None, :]
    Z = ground(X, Y)
    if drop_ground:
        Z = Z - denver.curvature_drop_m(np.hypot(X, Y))
    rows = horizon_row - f * (Z - eye_z) / (d[:, None] * np.cos(daz)[None, :])
    rows = np.where(np.isfinite(rows), rows, np.inf)
    top = np.minimum.accumulate(rows, axis=0)
    slope = np.gradient(np.nan_to_num(Z, nan=0.0), d, axis=0)
    ys = np.arange(h) + 0.5
    img = np.empty((h, w, 3))
    depth = np.full((h, w), np.inf)
    sky = SKY_TOP[None, :] + (SKY_LOW - SKY_TOP)[None, :] * np.clip(ys / max(horizon_row, 1), 0, 1)[:, None]
    for c in range(w):
        idx = np.searchsorted(-top[:, c], -ys, side="left")
        hit = idx < len(d)
        img[:, c] = sky
        if limb is not None:
            img[ys > horizon_row - f * limb / math.cos(daz[c]), c] = PLANET
        if not hit.any():
            continue
        s = idx[hit]
        dist = d[s]
        inside = (np.abs(X[s, c]) <= near_box) & (np.abs(Y[s, c]) <= near_box)
        base = np.where(inside[:, None], NEAR_GROUND[None, :], FAR_GROUND[None, :])
        shade = np.clip(0.8 + 4.0 * slope[s, c], 0.55, 1.2)[:, None]
        haze = (1.0 - np.exp(-dist / 30000.0))[:, None]
        img[hit, c] = (base * shade) * (1 - haze) + HAZE[None, :] * haze
        depth[hit, c] = dist

    if tris is not None and len(tris):
        yaw = math.radians(yaw_deg)
        fwd = np.array([math.sin(yaw), math.cos(yaw)])
        right = np.array([math.cos(yaw), -math.sin(yaw)])
        rel = tris - np.array([eye_xy[0], eye_xy[1], eye_z])
        F = rel[..., :2] @ fwd
        L = rel[..., :2] @ right
        u = w / 2.0 + f * L / F
        v = horizon_row - f * rel[..., 2] / F
        dist = np.hypot(rel[..., 0], rel[..., 1]).mean(axis=1)
        order = np.argsort(-dist)
        index = Image.new("I", (w, h), 0)
        draw = ImageDraw.Draw(index)
        for t in order:
            if np.any(F[t] <= 1.0):
                continue
            pts = list(zip(u[t], v[t]))
            if max(p[0] for p in pts) < 0 or min(p[0] for p in pts) > w:
                continue
            draw.polygon(pts, fill=int(t) + 1)
        idx = np.asarray(index, dtype=np.int64)
        mask = idx > 0
        tri = idx[mask] - 1
        nearer = dist[tri] < depth[mask]
        ys_, xs_ = np.nonzero(mask)
        ys_, xs_, tri = ys_[nearer], xs_[nearer], tri[nearer]
        haze = (1.0 - np.exp(-dist[tri] / 30000.0))[:, None]
        img[ys_, xs_] = tri_colour[tri] * (1 - haze) + HAZE[None, :] * haze
    return np.clip(img, 0, 255).astype(np.uint8)


def load_obj_triangles(path: Path, base_m: float) -> np.ndarray:
    """(T, 3, 3) triangles of an OBJ in absolute metres (east, north, elevation)."""
    verts, faces = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    v = np.asarray(verts) / 100.0
    v[:, 2] += base_m
    return v[np.asarray(faces)]


def city_colours(tris: np.ndarray) -> np.ndarray:
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    az, el = map(math.radians, SUN)
    sun = np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)])
    lambert = np.clip(n @ sun, 0.0, 1.0)
    return CITY[None, :] * (0.45 + 0.6 * lambert)[:, None]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--cache", type=Path, help="where the real-ground DEM is cached (default <out>/cache)")
    ap.add_argument("--png", type=Path, help="default <out>/check_denver.png")
    ap.add_argument("--min-visible-m", type=float, default=50.0)
    args = ap.parse_args(argv)
    out = args.out
    manifest = json.loads((out / "manifest.json").read_text())
    if "denver" not in manifest:
        print(f"{out}/manifest.json has no denver block: run `redrocks build --out {out}`")
        return 1
    base = manifest["terrain_mesh"]["min_elevation_m"]
    stage = manifest["stage"]
    eye_xy = (stage["walkway_east_m"], stage["walkway_north_m"])
    eye_z = stage["walkway_elevation_m"] + EYE_M
    front = (stage["front_edge_east_m"], stage["front_edge_north_m"])
    stage_bearing, stage_dist = denver.bearing_distance(eye_xy, front)
    proj = LocalProjection(manifest["origin"]["lat"], manifest["origin"]["lon"])
    cache = args.cache or out / "cache"

    env = load_env()
    tverts = env.read_obj_vertices(str(out / "RedRocks_Terrain.obj"))
    terrain = Terrain(np.asarray(tverts), base)
    border = env.grid_border(tverts)
    sverts, _, _ = env.skirt_mesh(border, base)
    skirt = Skirt(border, sverts, len(env.RING_OUT_CM), base)

    def game(x, y):
        z = terrain(x, y)
        return np.where(np.isfinite(z), z, skirt(x, y))

    real = RealGround(cached_dem(REAL_NEAR_BBOX, REAL_NEAR_SIZE, cache),
                      cached_dem(REAL_BBOX, REAL_DEM_SIZE, cache), proj)
    osm, ground = denver.fetch_downtown(cache)
    blocks = denver.merge_blocks(denver.building_blocks(osm, proj, ground)[0])
    towers, seen = [], set()
    for b in sorted(blocks, key=lambda b: -(b.top_m - b.ground_m)):
        key = b.name or b.osm
        if key in seen:
            continue
        seen.add(key)
        towers.append(b)
        if len(towers) == TOWERS:
            break

    print(f"eye: top walkway E{eye_xy[0]:.1f} N{eye_xy[1]:.1f}, {eye_z:.1f} m "
          f"(walkway {stage['walkway_elevation_m']:.1f} + {EYE_M} m)")
    print(f"looking at the stage front: bearing {stage_bearing:.1f} deg, {stage_dist:.0f} m away")
    d = manifest["denver"]
    print(f"Denver.obj: {d['buildings']} buildings, {d['prisms']} prisms, {d['triangles']:,} triangles; "
          f"skyline {d['distance_km']:.2f} km at {d['bearing_deg']:.1f} deg from the stage front")
    print("\nground under the sight line to the tallest tower (m above sea level; real dropped for curvature):")
    t0 = towers[0]
    b0, d0 = denver.bearing_distance(eye_xy, (t0.plan.centroid.x, t0.plan.centroid.y))
    samples = np.array([1000, 1900, 2500, 3500, 5000, 8000, 12000, 16000, 19000, 20000, 25000, 40000, 60000.0])
    g_line = march(eye_xy, b0, samples, game, False)
    r_line = march(eye_xy, b0, samples, real, True)
    for s, g, r in zip(samples, g_line, r_line):
        print(f"  {s / 1000:5.1f} km   game {g:7.1f}   real {r:7.1f}")

    limb = sky_limb(eye_z, base)
    print(f"  the SkyAtmosphere's planet (top at Z = 0, {base:.0f} m) has its limb {math.degrees(-limb):.2f} deg "
          f"below level from the eye")
    print(f"\n{'tower':40s} {'h m':>5s} {'brg':>6s} {'km':>6s}   {'game: clear  top vs horizon':>27s}"
          f"   {'real: clear  top vs horizon':>27s}")
    rows = []
    for t in towers:
        g_clear, g_top, brg, dist = tower_view(eye_xy, eye_z, t, game, False, GAME_FAR_M, horizon_floor=limb)
        r_clear, r_top, _, _ = tower_view(eye_xy, eye_z, t, real, True, REAL_FAR_M)
        height = t.top_m - t.ground_m
        rows.append((t, height, brg, dist, g_clear, g_top, r_clear, r_top))
        print(f"{(t.name or t.osm)[:40]:40s} {height:5.0f} {brg:6.1f} {dist / 1000:6.2f}   "
              f"{g_clear:11.0f} m {g_top:+11.0f} m   {r_clear:11.0f} m {r_top:+11.0f} m")
    print("(clear: metres of the tower above the ground in front of it; top vs horizon: metres its top stands\n"
          " above (+) or below (-) the horizon behind it, i.e. how much of it is against the sky)")

    # ---- picture
    tris = load_obj_triangles(out / "Denver.obj", base)
    colours = city_colours(tris)
    near_box = terrain.half
    W, H, HR = 1600, 470, 150.0
    ZW, ZH = 640, 210
    zoom_yaw = b0
    lowered = denver.curvature_drop_m(math.hypot(t0.plan.centroid.x, t0.plan.centroid.y))
    mid = ((t0.ground_m + t0.top_m) / 2.0 - lowered - eye_z) / d0
    fz = (ZW / 2.0) / math.tan(math.radians(ZOOM_FOV_DEG) / 2.0)
    zoom_hr = ZH / 2.0 + fz * mid
    panels = []
    for label, ground_fn, drop, far, clear_i, top_i, panel_limb in (
            ("IN THE GAME: terrain, then the skirt as setup_environment.py builds it, then the sky's planet",
             game, False, GAME_FAR_M, 4, 5, limb),
            ("REAL GROUND: USGS 3DEP with Earth curvature and refraction", real, True,
             REAL_FAR_M, 6, 7, None)):
        img = Image.fromarray(render(eye_xy, eye_z, stage_bearing, HFOV_DEG, (W, H), HR, ground_fn, drop,
                                     far, tris, colours, near_box, panel_limb))
        zoom = Image.fromarray(render(eye_xy, eye_z, zoom_yaw, ZOOM_FOV_DEG, (ZW, ZH), zoom_hr, ground_fn,
                                      drop, far, tris, colours, near_box, panel_limb))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=15)
        small = ImageFont.load_default(size=13)
        f = (W / 2.0) / math.tan(math.radians(HFOV_DEG) / 2.0)
        x_stage = W / 2.0
        x_tower = W / 2.0 + f * math.tan(math.radians(b0 - stage_bearing))
        draw.line([(x_stage, HR - 40), (x_stage, H)], fill=(255, 255, 255), width=1)
        draw.text((x_stage + 4, H - 20), f"stage front {stage_bearing:.1f} deg", fill=(255, 255, 255), font=small)
        draw.line([(x_tower, HR - 70), (x_tower, HR - 20)], fill=(255, 230, 120), width=2)
        tag = f"{t0.name or 'tallest'} {b0:.1f} deg, {d0 / 1000:.1f} km"
        draw.text((x_tower - 6 - draw.textlength(tag, font=small), HR - 76), tag, fill=(255, 240, 160), font=small)
        r = rows[0]
        draw.rectangle([(0, 0), (W, 24)], fill=(20, 20, 24))
        where = "above" if r[top_i] >= 0 else "below"
        draw.text((8, 4), f"{label}.  Eye {eye_z:.0f} m.  {t0.name}: {r[clear_i]:.0f} of {r[1]:.0f} m clear "
                          f"of the ground in front, top {abs(r[top_i]):.0f} m {where} the horizon",
                  fill=(240, 240, 240), font=font)
        zx, zy = W - ZW - 10, 32
        img.paste(zoom, (zx, zy))
        draw.rectangle([(zx - 1, zy - 1), (zx + ZW, zy + ZH)], outline=(255, 230, 120), width=2)
        draw.text((zx + 6, zy + ZH - 18), f"zoom: {ZOOM_FOV_DEG:.0f} deg on downtown", fill=(30, 30, 30),
                  font=small)
        panels.append(img)
    sheet = Image.new("RGB", (W, H * 2 + 4), (0, 0, 0))
    sheet.paste(panels[0], (0, 0))
    sheet.paste(panels[1], (0, H + 4))
    png = args.png or out / "check_denver.png"
    sheet.save(png)
    print(f"\nwrote {png}")
    worst = rows[0][4]
    if worst < args.min_visible_m:
        print(f"FAIL: {t0.name} shows only {worst:.0f} m in the game (< {args.min_visible_m:.0f} m)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
