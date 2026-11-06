"""Measure the stage's two stone towers and side roofs on the ml0w scan's top surface.

Reads the scan's highest point per 1 m cell (``assets/scan/scan_top.npz``,
``redrocks.scan_blend.ScanHeights``, already placed in the level frame by
``tools/scan_heights.py``) and the build's stage frame (``out/manifest.json``:
the deck's front-edge centre, the stage axis and the deck top), and prints the
numbers behind the ``tower_*`` and ``roof_wing_*`` entries of
``redrocks.stage.DIMENSIONS``.

Method:

* every 1 m cell is split into 4 x 4 samples and carried into stage
  coordinates (x across the stage, + to the audience's left; y along the axis,
  + toward the audience; 0 at the deck's front-edge centre);
* the scanned deck is the median of the samples within 1.5 m of the deck top
  where the roof has holes; heights below are above that scanned deck;
* the scan's own roof (samples 12.3 to 13.6 m over the deck top) gives its own
  frame: the turn whose quantile rectangle best overlaps the roof (it is turned
  about 11 degrees and sits about 1.3 m off the blockout's, because the scan's
  placement disagrees with the bowl the blockout is built on), with its front
  edge's centre as the origin;
* a tower is the largest connected patch within 1.25 m of the 99th-percentile
  height in a window either side of that roof; its extents are the 5-95 %
  quantiles across and along the roof frame, its top the median height;
* the scan as placed is 0.4748 m per scan unit (``FIT_SCALE`` in
  ``unreal/import_scan.py``) where the stage's other measured sizes use
  0.516 (``tools/measure_stage_scan.py``, which matches the published roof grid
  height), so lengths and heights are scaled by 0.516 / 0.4748.

Needs numpy and scipy (scipy is not a project dependency; it is in the dev venv).

    PYTHONPATH=src .venv/bin/python tools/measure_stage_towers.py [--out out]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from redrocks.scan_blend import ScanHeights  # noqa: E402

SCALE_MEASURED = 0.516  # m per scan unit behind the stage's other measured sizes
SUB = 4  # samples per cell side
ROOF_BAND = (12.3, 13.6)  # m over the deck top: the scan's roof surface
TOWER_DROP = 1.25  # m under the window's 99th percentile still counts as the top


def fit_scale_m(import_scan: Path) -> float:
    m = re.search(r"^FIT_SCALE = (-?[\d.]+)", import_scan.read_text(), re.M)
    if not m:
        raise ValueError("FIT_SCALE not found in %s" % import_scan)
    return float(m.group(1)) / 100.0


def stage_samples(top: ScanHeights, stage: dict):
    """(x, y, h): every sub-sample in stage coordinates, h over the deck top."""
    a = np.array(stage["axis_east_north"], float)
    l = np.array([a[1], -a[0]])
    o = np.array([stage["front_edge_east_m"], stage["front_edge_north_m"]])
    ny, nx = top.shape
    jj, ii = np.meshgrid(np.arange(nx * SUB), np.arange(ny * SUB))
    e = top.x0 + (jj + 0.5) / SUB * top.cell
    n = top.y0 + (ii + 0.5) / SUB * top.cell
    h = np.repeat(np.repeat(top.heights, SUB, 0), SUB, 1) - stage["deck_top_elevation_m"]
    return (e - o[0]) * l[0] + (n - o[1]) * l[1], (e - o[0]) * a[0] + (n - o[1]) * a[1], h


def largest(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    if n == 0:
        raise ValueError("nothing in the window")
    return lab == 1 + int(np.argmax(ndimage.sum(mask, lab, range(1, n + 1))))


def roof_frame(x, y, h):
    """(front-edge centre, across unit, toward-audience unit, width, depth) of the scan's roof."""
    from scipy import ndimage

    box = (x > -20) & (x < 14) & (y > -26) & (y < 4)
    roof = largest(box & (h > ROOF_BAND[0]) & (h < ROOF_BAND[1]))
    roof = ndimage.binary_closing(roof, iterations=2) & box | roof
    px, py, inside = x[box], y[box], roof[box]
    best = None
    for th in np.arange(-30.0, 30.0, 0.25):
        t = np.radians(th)
        u, v = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
        pu, pv = px * u[0] + py * u[1], px * v[0] + py * v[1]
        for q in (0.0, 0.01, 0.02, 0.04):
            u0, u1 = np.quantile(pu[inside], [q, 1 - q])
            v0, v1 = np.quantile(pv[inside], [q, 1 - q])
            r = (pu >= u0) & (pu <= u1) & (pv >= v0) & (pv <= v1)
            iou = (r & inside).sum() / (r | inside).sum()
            if best is None or iou > best[0]:
                best = (iou, th, u, v, u0, u1, v0, v1)
    iou, th, u, v, u0, u1, v0, v1 = best
    front = u * (u0 + u1) / 2 + v * v1
    return front, u, v, u1 - u0, v1 - v0, th, iou


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "out", help="redrocks build output (manifest.json)")
    ap.add_argument("--scan-top", type=Path, default=ROOT / "assets" / "scan" / "scan_top.npz")
    args = ap.parse_args(argv)

    stage = json.loads((args.out / "manifest.json").read_text())["stage"]
    top = ScanHeights.load(args.scan_top)
    s = SCALE_MEASURED / fit_scale_m(ROOT / "unreal" / "import_scan.py")
    x, y, h = stage_samples(top, stage)

    under = (np.abs(x) < 8) & (y < -3) & (y > -18) & (np.abs(h) < 1.5)
    deck = float(np.nanmedian(h[under]))
    front, u, v, w, d, th, iou = roof_frame(x, y, h)
    print("scanned deck: %+.2f m against the blockout deck top" % deck)
    print("scan roof: %.1f x %.1f m as placed (x %.3f = %.1f x %.1f m), turned %+.1f deg, "
          "front-edge centre at x %+.2f y %+.2f, IoU %.2f" % (w, d, s, w * s, d * s, th, *front, iou))

    rx = (x - front[0]) * u[0] + (y - front[1]) * u[1]  # across, in the scan roof's frame
    ry = (x - front[0]) * v[0] + (y - front[1]) * v[1]  # along, + toward the audience
    out = {}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        win = (rx * sign > 15) & (rx * sign < 32) & (ry > -12) & (ry < 8)
        peak = float(np.nanpercentile(np.where(win, h, np.nan), 99))
        blob = largest(win & (np.nan_to_num(h, nan=-99.0) > peak - TOWER_DROP))
        across = np.quantile(rx[blob], [0.05, 0.95]) * s
        along = np.quantile(ry[blob], [0.05, 0.95]) * s
        height = (float(np.median(h[blob])) - deck) * s
        inner, outer = sorted(np.abs(across))
        out[side] = dict(inner=inner, outer=outer, front=along[1], back=-along[0], height=height,
                         area=blob.sum() / SUB ** 2 * s * s)
        print("%s tower: inner face %.2f m, outer face %.2f m from the centre line (%.2f wide); "
              "%.2f m in front of to %.2f m behind the front edge (%.2f deep); top %.2f m over the deck; "
              "%.1f m^2" % (side, inner, outer, outer - inner, along[1], -along[0], along[1] - along[0],
                            height, out[side]["area"]))

    # The right-hand (south) side roof: the only one the scan reconstructs.
    half = w / 2
    wing = (rx < -(half - 1.0)) & (rx > -(half + 5.0)) & (ry > -9) & (ry < 2)
    band = wing & (h > 10.8) & (h < 12.3)
    if band.any():
        blob = largest(band)
        print("right side roof: top %.2f m over the deck, %.2f m deep (5-95 %%) at %.2f..%.2f across"
              % ((float(np.median(h[blob])) - deck) * s, float(np.ptp(np.quantile(ry[blob], [0.05, 0.95]))) * s,
                 *(np.quantile(rx[blob], [0.05, 0.95]) * s)))
    print(json.dumps({k: {kk: round(vv, 2) for kk, vv in val.items()} for k, val in out.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
