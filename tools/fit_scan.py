"""Fit the ml0w scan onto the DEM terrain: scale, yaw, mirror and offset.

The scan's USD carries no real-world units (Sketchfab normalised them), so its
size and placement are recovered by matching its surface heights to the 3DEP
terrain. For each candidate transform

    world_xy = R(yaw) . diag(mirror * s, s) . scan_xy + (tx, ty)
    world_z  = s * scan_z + tz

it samples the terrain under every scan point; tz is the median height
difference and the cost is the median absolute residual, which shrugs off trees,
buildings and the scan's ragged edges. A coarse grid over all parameters is
refined twice around the best candidates.

Inputs: /tmp/redrocks_scan_points.csv (from unreal/export_scan_points.py) and
out/RedRocks_Terrain.obj. Prints the transform in Unreal units (cm) for
unreal/import_scan.py.

    .venv/bin/python tools/fit_scan.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
POINTS = Path("/tmp/redrocks_scan_points.csv")
TERRAIN = ROOT / "out" / "RedRocks_Terrain.obj"
SAMPLE = 800
rng = np.random.default_rng(0)


def load_terrain(path: Path):
    """Terrain as a (rows, cols) height grid in metres, row 0 at the north edge."""
    verts = np.array(
        [[float(v) for v in line.split()[1:4]] for line in path.open() if line.startswith("v ")]
    ) / 100.0
    n = int(round(np.sqrt(len(verts))))
    grid = verts[:, 2].reshape(n, n)
    half = verts[:, 0].max()
    return grid, half


def sampler(grid: np.ndarray, half: float):
    n = grid.shape[0]

    def sample(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        col = (x + half) / (2 * half) * (n - 1)
        row = (half - y) / (2 * half) * (n - 1)
        inside = (col >= 0) & (col <= n - 1) & (row >= 0) & (row <= n - 1)
        c0 = np.clip(np.floor(col).astype(int), 0, n - 2)
        r0 = np.clip(np.floor(row).astype(int), 0, n - 2)
        tc = np.clip(col - c0, 0, 1)
        tr = np.clip(row - r0, 0, 1)
        top = grid[r0, c0] * (1 - tc) + grid[r0, c0 + 1] * tc
        bot = grid[r0 + 1, c0] * (1 - tc) + grid[r0 + 1, c0 + 1] * tc
        z = top * (1 - tr) + bot * tr
        return np.where(inside, z, np.nan)

    return sample


def cost(pts, sample, s, yaw, mirror, tx, ty):
    """1 - correlation between scan and terrain heights, and the matching tz.

    A plain height residual favours small scales, because a smaller patch has
    less relief to get wrong. Correlation is blind to that: a wrong placement
    scores near 0 however small it is.
    """
    c, si = np.cos(yaw), np.sin(yaw)
    x = mirror * s * pts[:, 0]
    y = s * pts[:, 1]
    wx = c * x - si * y + tx
    wy = si * x + c * y + ty
    ground = sample(wx, wy)
    if np.isnan(ground).mean() > 0.05:
        return np.inf, 0.0
    r = ground - s * pts[:, 2]
    tz = np.median(r)
    # Compare local shape only: any tilted patch correlates with a tilted
    # scan, so the best-fit plane is removed from both before comparing.
    a = np.column_stack([wx, wy, np.ones_like(wx)])
    g = ground - a @ np.linalg.lstsq(a, ground, rcond=None)[0]
    z = pts[:, 2] - a @ np.linalg.lstsq(a, pts[:, 2], rcond=None)[0]
    corr = np.corrcoef(g, z)[0, 1]
    return float(1.0 - corr), float(tz)


def residual(pts, sample, s, yaw, mirror, tx, ty, tz):
    """Median absolute height error in metres for a finished fit."""
    c, si = np.cos(yaw), np.sin(yaw)
    x = mirror * s * pts[:, 0]
    y = s * pts[:, 1]
    ground = sample(c * x - si * y + tx, si * x + c * y + ty)
    return float(np.nanmedian(np.abs(ground - (s * pts[:, 2] + tz))))


def search(pts, sample, scales, yaws, mirrors, txs, tys):
    results = []
    for s in scales:
        for yaw in yaws:
            for mirror in mirrors:
                for tx in txs:
                    for ty in tys:
                        e, tz = cost(pts, sample, s, yaw, mirror, tx, ty)
                        if np.isfinite(e):
                            results.append((e, s, yaw, mirror, tx, ty, tz))
    results.sort(key=lambda r: r[0])
    return results


def main() -> int:
    if not POINTS.exists():
        print(f"missing {POINTS}: run unreal/export_scan_points.py in the editor first")
        return 1
    pts = np.loadtxt(POINTS, delimiter=",")
    pts = pts - np.array([*pts[:, :2].mean(axis=0), 0.0])  # centre in XY
    width = np.ptp(pts[:, :2], axis=0).max()
    grid, half = load_terrain(TERRAIN)
    sample = sampler(grid, half)
    print(f"{len(pts)} scan points, {width:.0f} units across; terrain +-{half:.0f} m")

    coarse = pts[rng.choice(len(pts), SAMPLE, replace=False)]
    # Scan widths from 200 m up to what fits inside the terrain box.
    scales = np.geomspace(200.0 / width, 1.9 * half / width, 24)
    yaws = np.radians(np.arange(0, 360, 10))
    best = search(coarse, sample, scales, yaws, (1, -1), np.arange(-500, 501, 100), np.arange(-500, 501, 100))
    print("coarse best:", ["%.2f" % b[0] for b in best[:5]])

    for step_s, step_yaw, step_t in ((0.04, 4, 30), (0.01, 1, 8)):
        fine = []
        for e, s, yaw, mirror, tx, ty, _ in best[:6]:
            fine += search(
                coarse,
                sample,
                s * (1 + np.arange(-3, 4) * step_s),
                yaw + np.radians(np.arange(-2, 3) * step_yaw),
                (mirror,),
                tx + np.arange(-3, 4) * step_t,
                ty + np.arange(-3, 4) * step_t,
            )
        best = sorted(fine, key=lambda r: r[0])
        print("refined best:", ["%.2f" % b[0] for b in best[:5]])

    e, s, yaw, mirror, tx, ty, _ = best[0]
    e_all, tz = cost(pts, sample, s, yaw, mirror, tx, ty)
    centre = np.loadtxt(POINTS, delimiter=",")[:, :2].mean(axis=0)
    print()
    print(f"correlation      {1 - e_all:.3f} over all {len(pts)} points")
    print(f"median residual  {residual(pts, sample, s, yaw, mirror, tx, ty, tz):.2f} m")
    for alt in best[1:4]:
        print(
            "  runner-up      corr %.3f  width %.0f m  yaw %.0f  mirror %+d  at (%.0f, %.0f)"
            % (1 - alt[0], alt[1] * width, np.degrees(alt[2]) % 360, alt[3], alt[4], alt[5])
        )
    print(f"scan width       {s * width:.0f} m  ({s:.4f} m per scan unit)")
    print(f"yaw              {np.degrees(yaw) % 360:.1f} deg   mirror {mirror:+d}")
    print(f"offset           tx {tx:.1f} m  ty {ty:.1f} m  tz {tz:.1f} m (above terrain base)")
    print()
    print("# for unreal/import_scan.py (cm; scan XY centre removed before rotation)")
    print(f"FIT_SCALE = {s * 100.0:.4f}")
    print(f"FIT_YAW_DEG = {np.degrees(yaw) % 360:.2f}")
    print(f"FIT_MIRROR_X = {mirror}")
    print(f"FIT_SCAN_CENTRE = ({centre[0]:.3f}, {centre[1]:.3f})")
    print(f"FIT_OFFSET_CM = ({tx * 100:.1f}, {ty * 100:.1f}, {tz * 100:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
