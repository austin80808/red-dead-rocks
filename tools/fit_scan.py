"""Fit the ml0w scan onto the real terrain: scale, yaw, mirror and offset.

Frames. Scan points are in the frame unreal/export_scan_points.py writes
(USD world space with Y-up turned Z-up: p = (x, z_usd, y_usd), scan units).
The terrain is out/RedRocks_Terrain.obj in metres (X east, +Y north, Z above the
mesh base), the same frame as out/manifest.json. The transform is

    world_xy = R(yaw) . diag(mirror * s, s) . (p_xy - centre) + (tx, ty)
    world_z  = s * p_z + tz

which is what unreal/import_scan.py applies to the imported chunks.

Method (issue #14). The scale is not free: bench rise, roof height and the
rock-top difference put it near 0.516 m per scan unit (PR #33,
tools/measure_stage_scan.py), so it is searched only within +-8 % of that.
Heights alone cannot place the scan (the cost is nearly flat in scale and
the terrain has many look-alike slopes), so the cost combines

  * median |scan height - terrain height| (tz is the median difference), and
  * landmark errors in metres, mean of: stage front edge (manifest
    stage.front_edge_*), and the summit centroids of both monoliths (the
    terrain's top 8 m of each rock footprint vs the scan's top 8 m).

A coarse grid over yaw, mirror and a translation anchored on the stage front
edge seeds a local pattern search. Prints the constants for import_scan.py,
residuals and landmark errors, and writes a top-down PNG.

    .venv/bin/python tools/fit_scan.py --scan assets/scan/usdz_extract/scene.usdc \\
        --png OUTDIR/scan_fit.png

Needs numpy and, to read the USD (or a cache), usd-core (not a project
dependency; `pip install usd-core` into the venv). matplotlib for --png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TERRAIN = ROOT / "out" / "RedRocks_Terrain.obj"
MANIFEST = ROOT / "out" / "manifest.json"
SCAN = ROOT / "assets" / "scan" / "usdz_extract" / "scene.usdc"

SCALE0 = 0.516  # m per scan unit, from PR #33's measurements
SCALE_TOL = 0.08
STRIDE = 40  # keep every 40th scan vertex
SUMMIT_DROP_M = 8.0

# Scan-frame landmarks (scan units, export frame x, y), measured in PR #33:
# deck front lip at x 113.75, stage centre z 101 (roof z 75.5..126.5).
SCAN_LIP = np.array([113.75, 101.0])
# Search boxes (x0, x1, y0, y1) around each rock's summit in the scan frame.
SCAN_ROCK_BOX = {
    "Creation Rock": (-140.0, 10.0, -90.0, 40.0),
    "Ship Rock": (-140.0, -30.0, 190.0, 300.0),
}


# ---- inputs ------------------------------------------------------------------
def load_scan_points(path: Path, stride: int = STRIDE) -> np.ndarray:
    """Scan vertices as (N, 3) in the export frame (x, z_usd, y_usd)."""
    if path.suffix == ".npz":  # cache of USD-world points, key P
        p = np.load(path)["P"][::stride]
        return np.column_stack([p[:, 0], p[:, 2], p[:, 1]]).astype(float)
    if path.suffix == ".csv":
        return np.loadtxt(path, delimiter=",")
    from pxr import Usd, UsdGeom  # usd-core

    stage = Usd.Stage.Open(str(path))
    rows = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
        for i in range(0, len(pts), stride):
            p = m.Transform(pts[i])
            rows.append((p[0], p[2], p[1]))
    return np.array(rows, dtype=float)


def load_terrain(path: Path):
    """Terrain as a (rows, cols) height grid in metres, row 0 at the north edge."""
    verts = np.array(
        [[float(v) for v in line.split()[1:4]] for line in path.open() if line.startswith("v ")]
    ) / 100.0
    n = int(round(np.sqrt(len(verts))))
    return verts[:, 2].reshape(n, n), verts[:, 0].max()


def sampler(grid: np.ndarray, half: float):
    n = grid.shape[0]

    def sample(x, y):
        col = (x + half) / (2 * half) * (n - 1)
        row = (half - y) / (2 * half) * (n - 1)
        inside = (col >= 0) & (col <= n - 1) & (row >= 0) & (row <= n - 1)
        c0 = np.clip(np.floor(col).astype(int), 0, n - 2)
        r0 = np.clip(np.floor(row).astype(int), 0, n - 2)
        tc, tr = np.clip(col - c0, 0, 1), np.clip(row - r0, 0, 1)
        top = grid[r0, c0] * (1 - tc) + grid[r0, c0 + 1] * tc
        bot = grid[r0 + 1, c0] * (1 - tc) + grid[r0 + 1, c0 + 1] * tc
        return np.where(inside, top * (1 - tr) + bot * tr, np.nan)

    return sample


def obj_bbox(path: Path):
    v = np.array([[float(t) for t in l.split()[1:4]] for l in path.open() if l.startswith("v ")]) / 100.0
    return v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()


def terrain_summit(grid, half, box, drop=SUMMIT_DROP_M):
    """Centroid (E, N) of the cells within `drop` m of the highest one inside box."""
    n = grid.shape[0]
    xs = np.linspace(-half, half, n)
    ys = np.linspace(half, -half, n)
    xx, yy = np.meshgrid(xs, ys)
    inb = (xx >= box[0]) & (xx <= box[1]) & (yy >= box[2]) & (yy <= box[3])
    g = np.where(inb, grid, -np.inf)
    top = g >= g.max() - drop
    return np.array([xx[top].mean(), yy[top].mean()])


# ---- transform and cost ------------------------------------------------------
def to_world(xy, s, yaw, mirror, t, centre):
    """Scan (x, y) -> terrain (E, N) in metres."""
    x = mirror * s * (xy[:, 0] - centre[0])
    y = s * (xy[:, 1] - centre[1])
    c, si = np.cos(yaw), np.sin(yaw)
    return c * x - si * y + t[0], si * x + c * y + t[1]


def height_residual(pts, sample, s, yaw, mirror, t, centre):
    """(median |dz| in m, tz); inf when much of the scan falls off the terrain."""
    wx, wy = to_world(pts[:, :2], s, yaw, mirror, t, centre)
    ground = sample(wx, wy)
    ok = ~np.isnan(ground)
    if ok.mean() < 0.9:
        return np.inf, 0.0
    r = ground[ok] - s * pts[ok, 2]
    tz = float(np.median(r))
    return float(np.median(np.abs(r - tz))), tz


class Landmarks:
    """Real positions (terrain frame) and how to find the same features in the scan."""

    def __init__(self, manifest, grid, half, terrain_dir: Path, pts):
        st = manifest["stage"]
        self.front_edge = np.array([st["front_edge_east_m"], st["front_edge_north_m"]])
        self.rocks = {}
        for rock in manifest["monoliths"]:
            key = "Creation Rock" if rock["name"].startswith("Creation") else "Ship Rock"
            box = obj_bbox(terrain_dir / rock["obj"])
            sp = pts[_in_box(pts, SCAN_ROCK_BOX[key])]
            self.rocks[key] = dict(
                centroid=np.array([rock["centroid_east"], rock["centroid_north"]]),
                summit=terrain_summit(grid, half, box),
                scan_pts=sp,
            )

    def scan_summit(self, key, s, yaw, mirror, t, centre):
        q = self.rocks[key]["scan_pts"]
        top = q[:, 2] >= q[:, 2].max() - SUMMIT_DROP_M / s
        wx, wy = to_world(q[top, :2], s, yaw, mirror, t, centre)
        return np.array([wx.mean(), wy.mean()])

    def errors(self, s, yaw, mirror, t, centre):
        """Landmark position errors in metres."""
        wx, wy = to_world(SCAN_LIP[None], s, yaw, mirror, t, centre)
        e = {"stage front edge": float(np.hypot(wx[0] - self.front_edge[0], wy[0] - self.front_edge[1]))}
        for key, r in self.rocks.items():
            e[key + " summit"] = float(np.linalg.norm(self.scan_summit(key, s, yaw, mirror, t, centre) - r["summit"]))
        return e


def _in_box(p, box):
    return (p[:, 0] > box[0]) & (p[:, 0] < box[1]) & (p[:, 1] > box[2]) & (p[:, 1] < box[3])


def make_cost(pts, sample, lm, mirror, centre, weight=0.25):
    def cost(x):
        s, yaw_deg, tx, ty = x
        if abs(s / SCALE0 - 1) > SCALE_TOL:
            return 1e3 + abs(s - SCALE0) * 1e3
        yaw = np.radians(yaw_deg)
        e, _ = height_residual(pts, sample, s, yaw, mirror, (tx, ty), centre)
        if not np.isfinite(e):
            return 1e3
        return e + weight * float(np.mean(list(lm.errors(s, yaw, mirror, (tx, ty), centre).values())))

    return cost


def pattern_search(cost, x0, steps, rounds=9):
    """Deterministic coordinate search with step halving."""
    x, steps = np.array(x0, float), np.array(steps, float)
    cur = cost(x)
    for _ in range(rounds):
        improved = True
        while improved:
            improved = False
            for i in range(len(x)):
                for sign in (1, -1):
                    y = x.copy()
                    y[i] += sign * steps[i]
                    v = cost(y)
                    if v < cur - 1e-9:
                        x, cur, improved = y, v, True
        steps = steps / 2
    return cur, x


def fit(pts, sample, lm, centre, seed=0, n_search=3000):
    """Return the best (cost, mirror, x=(s, yaw_deg, tx, ty))."""
    rng = np.random.default_rng(seed)
    sub = pts[rng.choice(len(pts), min(n_search, len(pts)), replace=False)]
    seeds = []
    for mirror in (1, -1):
        for yaw_deg in range(0, 360, 5):
            for s in (SCALE0 * (1 - SCALE_TOL), SCALE0, SCALE0 * (1 + SCALE_TOL)):
                yaw = np.radians(yaw_deg)
                lx, ly = to_world(SCAN_LIP[None], s, yaw, mirror, (0, 0), centre)
                t0 = lm.front_edge - np.array([lx[0], ly[0]])  # lip on the front edge
                for dx in range(-64, 65, 16):
                    for dy in range(-64, 65, 16):
                        t = (t0[0] + dx, t0[1] + dy)
                        e, _ = height_residual(sub, sample, s, yaw, mirror, t, centre)
                        seeds.append((e, mirror, s, yaw_deg, t[0], t[1]))
    seeds.sort()
    results = []
    for _, mirror, s, yaw_deg, tx, ty in seeds[:8]:
        cost = make_cost(sub, sample, lm, mirror, centre)
        c, x = pattern_search(cost, (s, yaw_deg, tx, ty), (0.01, 4.0, 8.0, 8.0))
        results.append((c, mirror, x))
    results.sort(key=lambda r: r[0])
    # re-score the winner on every point
    c, mirror, x = results[0]
    c, x = pattern_search(make_cost(pts, sample, lm, mirror, centre), x, (0.004, 1.0, 2.0, 2.0), rounds=6)
    return c, mirror, x


# ---- outputs -----------------------------------------------------------------
def plot(path, pts, grid, half, sample, lm, s, yaw, mirror, t, tz, centre, manifest):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    n = grid.shape[0]
    xs, ys = np.linspace(-half, half, n), np.linspace(half, -half, n)
    r = 450
    sx, sy = (xs > -r) & (xs < r), (ys > -r) & (ys < r)
    g = grid[np.ix_(sy, sx)]
    dx = 2 * half / (n - 1)
    shade = LightSource(315, 45).hillshade(g, vert_exag=1, dx=dx, dy=dx)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(shade, extent=(-r, r, -r, r), cmap="gray")
    wx, wy = to_world(pts[:, :2], s, yaw, mirror, t, centre)
    dz = sample(wx, wy) - (s * pts[:, 2] + tz)
    sc = ax.scatter(wx, wy, c=dz, s=1.2, cmap="coolwarm", vmin=-15, vmax=15, alpha=0.55)
    plt.colorbar(sc, ax=ax, shrink=0.6, label="terrain - scan height (m)")
    for key, rk in lm.rocks.items():
        ax.plot(*rk["centroid"], "k^", ms=11, label=key + " centroid (manifest)")
        ax.plot(*rk["summit"], "g*", ms=14, label=key + " summit (terrain)")
        ax.plot(*lm.scan_summit(key, s, yaw, mirror, t, centre), "m+", ms=14, mew=2, label=key + " summit (scan)")
    ax.plot(*lm.front_edge, "yo", ms=10, mec="k", label="stage front edge (manifest)")
    ax.plot(*to_world(SCAN_LIP[None], s, yaw, mirror, t, centre), "rx", ms=12, mew=2, label="stage front lip (scan)")
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title("ml0w scan on the terrain: %.4f m/unit, yaw %.1f deg, mirror %+d" % (s, np.degrees(yaw) % 360, mirror))
    ax.legend(loc="lower left", fontsize=7)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scan", type=Path, default=SCAN, help="scene.usdc, a .npz cache (key P) or a points .csv")
    ap.add_argument("--terrain", type=Path, default=TERRAIN)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--png", type=Path, help="write a top-down check image here")
    args = ap.parse_args(argv)

    pts = load_scan_points(args.scan)
    centre = tuple(np.round(pts[:, :2].mean(axis=0), 3))
    grid, half = load_terrain(args.terrain)
    sample = sampler(grid, half)
    manifest = json.loads(args.manifest.read_text())
    lm = Landmarks(manifest, grid, half, args.manifest.parent, pts)
    print(f"{len(pts)} scan points; terrain +-{half:.0f} m")

    cost, mirror, (s, yaw_deg, tx, ty) = fit(pts, sample, lm, centre)
    yaw = np.radians(yaw_deg)
    med, tz = height_residual(pts, sample, s, yaw, mirror, (tx, ty), centre)
    errs = lm.errors(s, yaw, mirror, (tx, ty), centre)
    print(f"\ncost {cost:.2f}   median |dz| {med:.2f} m over {len(pts)} points")
    print(f"scale {s:.4f} m/unit (search {SCALE0 * (1 - SCALE_TOL):.3f}..{SCALE0 * (1 + SCALE_TOL):.3f})")
    for k, v in errs.items():
        print(f"landmark error   {k:<22s}{v:6.1f} m")
    for key, r in lm.rocks.items():
        c = lm.scan_summit(key, s, yaw, mirror, (tx, ty), centre)
        print(f"  {key}: manifest centroid {np.round(r['centroid'], 1)}, terrain summit {np.round(r['summit'], 1)}, scan summit {np.round(c, 1)}")
    print("\n# for unreal/import_scan.py (cm; scan XY centre removed before rotation)")
    print(f"FIT_SCALE = {s * 100.0:.4f}")
    print(f"FIT_YAW_DEG = {np.degrees(yaw) % 360:.2f}")
    print(f"FIT_MIRROR_X = {mirror}")
    print(f"FIT_SCAN_CENTRE = ({centre[0]:.3f}, {centre[1]:.3f})")
    print(f"FIT_OFFSET_CM = ({tx * 100:.1f}, {ty * 100:.1f}, {tz * 100:.1f})")
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        plot(args.png, pts, grid, half, sample, lm, s, yaw, mirror, (tx, ty), tz, centre, manifest)
        print("wrote", args.png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
