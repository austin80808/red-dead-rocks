"""Check that nothing pokes through the seating in a `redrocks build` output.

Reads out/Benches.obj (the terraces' treads), the other meshes, the vegetation
placement, the scan's cut fade mask and its top raster, and reports:

* any vertex of the terrain, OSM rocks, walls, buildings, paths, roads, lots or
  water standing above a tread inside the seating;
* any tree or shrub within 1 m of the seating;
* any scan still visible inside the seating;
* any hidden scan standing more than a step (0.55 m) above a tread, which its
  collision would turn into an invisible ledge.

Stage.obj is left out: the sound booth stands on the rows on purpose.

    .venv/bin/python tools/check_seating_clear.py [--out out]

Exits 1 when anything pokes through.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import shapely
import shapely.prepared
from PIL import Image
from shapely.geometry import Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from redrocks.scan_blend import ScanHeights  # noqa: E402

MESHES = ("RedRocks_Terrain", "Rocks", "Walls", "Buildings", "Paths", "Roads", "Parking", "Water")
STEP_M = 0.55


def _verts(path: Path) -> np.ndarray:
    return np.array([list(map(float, ln.split()[1:4])) for ln in path.open() if ln.startswith("v ")]) / 100.0


def treads(benches: Path) -> list[tuple[float, Polygon]]:
    """(tread top z in metres over the base, plan) for every flat upward face."""
    lines = benches.read_text().splitlines()
    v = np.array([list(map(float, ln.split()[1:4])) for ln in lines if ln.startswith("v ")]) / 100.0
    by_level: dict[float, list[Polygon]] = {}
    for ln in lines:
        if not ln.startswith("f "):
            continue
        f = [int(t.split("/")[0]) - 1 for t in ln.split()[1:]]
        a, b, c = v[f]
        n = np.cross(b - a, c - a)
        if n[2] > 0.999 * np.linalg.norm(n):
            by_level.setdefault(round(float(a[2]), 3), []).append(Polygon(v[f][:, :2]))
    return [(z, unary_union(p)) for z, p in by_level.items()]


def check(out: Path, scan_dir: Path) -> list[str]:
    levels = treads(out / "Benches.obj")
    seat = unary_union([p for _, p in levels])
    prepared = [(z, shapely.prepared.prep(p)) for z, p in levels]

    def tread_at(x, y):
        hits = [z for z, p in prepared if p.contains(shapely.Point(x, y))]
        return max(hits) if hits else None

    problems: list[str] = []
    for name in MESHES:
        path = out / f"{name}.obj"
        if not path.exists():
            continue
        v = _verts(path)
        inside = v[shapely.contains_xy(seat, v[:, 0], v[:, 1])]
        above = [(x, y, z - t) for x, y, z in inside if (t := tread_at(x, y)) is not None and z > t + 1e-3]
        if above:
            x, y, d = max(above, key=lambda p: p[2])
            problems.append(f"{name}: {len(above)} vertices above a tread (worst {d:.2f} m at E{x:.1f} N{y:.1f})")

    veg = out / "vegetation" / "vegetation.json"
    if veg.exists():
        near = []
        for model, insts in json.loads(veg.read_text())["instances"].items():
            for inst in insts:
                p = shapely.Point(inst[0] / 100.0, inst[1] / 100.0)
                if seat.distance(p) < 1.0:
                    near.append((model, p.x, p.y))
        if near:
            problems.append(f"vegetation: {len(near)} plants in or within 1 m of the seating, e.g. {near[0]}")

    manifest = json.loads((out / "manifest.json").read_text())
    base = manifest["benches"]["base_elevation_m"]
    fade_png, fade_json, top_npz = out / "scan_fade.png", out / "scan_fade.json", scan_dir / "scan_top.npz"
    if fade_png.exists() and top_npz.exists():
        place = json.loads(fade_json.read_text())
        alpha = np.asarray(Image.open(fade_png).convert("L"), float) / 255.0
        top = ScanHeights.load(top_npz)
        nx, ny = place["texels"]
        tx, ty = place["width_cm"] / 100.0 / nx, place["height_cm"] / 100.0 / ny
        e, n = np.meshgrid(place["x0_cm"] / 100.0 + (np.arange(nx) + 0.5) * tx,
                           place["y1_cm"] / 100.0 - (np.arange(ny) + 0.5) * ty)
        inside = shapely.contains_xy(seat, e, n)
        visible = int((inside & (alpha > 0.05)).sum())
        if visible:
            problems.append(f"scan: visible over {visible * tx * ty:.1f} m^2 inside the seating")
        ledges = 0
        worst = 0.0
        for i, j in zip(*np.nonzero(inside)):
            ti, tj = int((n[i, j] - top.y0) // top.cell), int((e[i, j] - top.x0) // top.cell)
            if not (0 <= ti < top.shape[0] and 0 <= tj < top.shape[1]):
                continue
            z = top.heights[ti, tj] - base
            t = tread_at(e[i, j], n[i, j])
            if t is not None and np.isfinite(z) and z - t > STEP_M:
                ledges += 1
                worst = max(worst, z - t)
        if ledges:
            problems.append(f"scan: {ledges} hidden cells stand over a step above a tread (worst {worst:.2f} m)")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--scan-dir", type=Path, default=ROOT / "assets" / "scan")
    args = ap.parse_args(argv)
    problems = check(args.out, args.scan_dir)
    for p in problems:
        print("POKES THROUGH  " + p)
    if not problems:
        print("clear: nothing stands above a tread, no plants and no visible scan in the seating, "
              f"no hidden scan more than {STEP_M} m over a tread")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
