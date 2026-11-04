"""Rasterise the placed ml0w scan for the terrain blend and the scan's edge fade.

Writes, from the scan USD and the placement constants in unreal/import_scan.py:

  assets/scan/scan_heights.npz   the scan's top surface, 2 m cells, absolute
                                 elevation (redrocks.scan_blend.ScanHeights);
                                 `redrocks build` conforms the terrain to it.
  assets/scan/scan_fade.png      the scan's opacity in plan view: 0 at its
                                 footprint edge, 1 from FADE_M inside (row 0 =
                                 north). import_scan.py imports it and the scan
                                 material dithers by it.
  assets/scan/scan_fade.json     where that image sits in the level (cm), read
                                 by import_scan.py.

Re-run it whenever the FIT_* constants in unreal/import_scan.py change, then
rebuild the package (`redrocks build`) and the level (rebuild_level.py):

    .venv/bin/python tools/scan_heights.py --scan assets/scan/usdz_extract/scene.usdc

Frames: scan points go through the same transform as tools/fit_scan.py and the
Unreal actors (world = R(yaw) . diag(mirror*s, s) . (p - centre) + offset), which
lands them in the terrain mesh frame: X east, Y north (m from the bbox centre),
Z above the mesh base. The base (the 377-grid DEM minimum) comes from the
manifest, so the heights are absolute elevations like the DEM.

Needs numpy and pillow; reading the USD needs usd-core (`pip install usd-core`),
or pass a points cache (.npz with key P, USD world points) as --scan.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from redrocks.scan_blend import CELL_M, CLOSE_M, FADE_M, ScanHeights, fade_alpha, footprint  # noqa: E402

IMPORT_SCAN = ROOT / "unreal" / "import_scan.py"
SCAN = ROOT / "assets" / "scan" / "usdz_extract" / "scene.usdc"
MANIFEST = ROOT / "out" / "manifest.json"
OUT_DIR = ROOT / "assets" / "scan"


def fit_constants(src: str) -> dict:
    """The FIT_* placement constants from unreal/import_scan.py's source."""

    def num(name):
        m = re.search(r"^%s = (-?[\d.]+)" % name, src, re.M)
        if not m:
            raise ValueError("%s not found in import_scan.py" % name)
        return float(m.group(1))

    def tup(name):
        m = re.search(r"^%s = \(([^)]*)\)" % name, src, re.M)
        if not m:
            raise ValueError("%s not found in import_scan.py" % name)
        return tuple(float(v) for v in m.group(1).split(","))

    return {
        "scale_m": num("FIT_SCALE") / 100.0,
        "yaw_deg": num("FIT_YAW_DEG"),
        "mirror": int(num("FIT_MIRROR_X")),
        "centre": tup("FIT_SCAN_CENTRE"),
        "offset_m": tuple(v / 100.0 for v in tup("FIT_OFFSET_CM")),
    }


def _fit_scan_module():
    spec = importlib.util.spec_from_file_location("fit_scan", ROOT / "tools" / "fit_scan.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def place(points: np.ndarray, fit: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scan points (export frame) -> (east, north, z above the mesh base), metres."""
    fs = _fit_scan_module()
    s = fit["scale_m"]
    tx, ty, tz = fit["offset_m"]
    e, n = fs.to_world(points[:, :2], s, np.radians(fit["yaw_deg"]), fit["mirror"], (tx, ty), fit["centre"])
    return e, n, s * points[:, 2] + tz


def write_fade(scan: ScanHeights, out_dir: Path, fade_m: float = FADE_M) -> tuple[Path, Path]:
    """The fade mask PNG (row 0 = north) and its placement JSON."""
    alpha = fade_alpha(scan, fade_m)
    img = np.flipud(np.round(alpha * 255.0).astype(np.uint8))
    png = out_dir / "scan_fade.png"
    Image.fromarray(np.dstack([img, img, img]), "RGB").save(png)
    ny, nx = alpha.shape
    meta = {
        "about": "ml0w scan opacity in plan view, written by tools/scan_heights.py; "
        "u = (X - x0_cm) / width_cm, v = (y1_cm - Y) / height_cm in level cm (X east, Y north)",
        "x0_cm": round(scan.x0 * 100.0, 1),
        "y1_cm": round((scan.y0 + ny * scan.cell) * 100.0, 1),
        "width_cm": round(nx * scan.cell * 100.0, 1),
        "height_cm": round(ny * scan.cell * 100.0, 1),
        "texels": [nx, ny],
        "fade_m": fade_m,
        "close_m": CLOSE_M,
    }
    js = out_dir / "scan_fade.json"
    js.write_text(json.dumps(meta, indent=2) + "\n")
    return png, js


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scan", type=Path, default=SCAN, help="scene.usdc, or a .npz points cache (key P)")
    ap.add_argument("--manifest", type=Path, default=MANIFEST, help="for the terrain mesh base elevation")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--png", type=Path, help="optional diagnostic plot")
    args = ap.parse_args(argv)

    fit = fit_constants(IMPORT_SCAN.read_text())
    base = json.loads(args.manifest.read_text())["terrain_mesh"]["min_elevation_m"]
    pts = _fit_scan_module().load_scan_points(args.scan, stride=1)
    east, north, z = place(pts, fit)
    # Pad by two cells so the footprint never touches the raster border.
    scan = ScanHeights.from_points(east, north, z + base, CELL_M).padded(2 * CELL_M)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scan.save(args.out_dir / "scan_heights.npz")
    png, js = write_fade(scan, args.out_dir)
    inside = footprint(scan)
    print("scan points %d, raster %dx%d at %.1f m, footprint %.0f m^2"
          % (len(pts), scan.shape[1], scan.shape[0], scan.cell, inside.sum() * scan.cell ** 2))
    print("wrote %s, %s, %s" % (args.out_dir / "scan_heights.npz", png, js))

    if args.png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ext = (scan.x0, scan.x0 + scan.shape[1] * scan.cell, scan.y0, scan.y0 + scan.shape[0] * scan.cell)
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        ax[0].imshow(scan.heights, origin="lower", extent=ext)
        ax[0].set_title("scan top elevation (m)")
        ax[1].imshow(fade_alpha(scan), origin="lower", extent=ext, cmap="gray")
        ax[1].set_title("fade alpha")
        fig.savefig(args.png, dpi=80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
