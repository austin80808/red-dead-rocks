"""Measure the Red Rocks stage on the ml0w photogrammetry scan.

The scan (assets/scan/usdz_extract/scene.usdc, CC BY 4.0) has no real-world
units, and the fit in ``unreal/import_scan.py`` is unverified (issue #14). So
this script works in raw scan units (USD world space, Y up, including the 2.02
scale on /scene/Meshes), finds its own metres-per-unit from several published
figures, and only then converts the stage measurements.

Method, per measurement:

* bowl profile: median height along the stage axis (scan x) through the bowl
  centre line; the seating is a straight ramp, the top row is where the ramp
  meets the flat behind it, the deck is the flat at the bottom;
* deck level and roof top: peaks of the height histogram inside the stage
  footprint (the roof is only partly reconstructed, so the deck shows through);
* roof plan: extents of the connected up-facing surface near the roof-top
  height; roof underside: peak of the down-facing surface heights;
* deck front lip / rear and the wall behind: along-axis profile on a clean band.

Needs ``usd-core`` (``pip install usd-core`` into a venv, it is not a project
dependency), numpy, scipy and matplotlib. Keep the cache outside the repo.

    .venv/bin/python tools/measure_stage_scan.py --cache $SCRATCH/scan.npz --out $SCRATCH/diag

Raw scan axes used below: x runs from the top of the bowl (small x) toward the
stage (large x); z is across the stage; ``h`` is height.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCAN = ROOT / "assets" / "scan" / "usdz_extract" / "scene.usdc"

# Published figures used for the scale check.
OFFICIAL_RISE_M = 100 * 0.3048  # stage to top row, Red Rocks history page
ROOF_GRID_M = (40 + 2 / 12.0) * 0.3048  # SkyDeck I-beam grid above the stage floor, IA Stage
CREATION_TOP_M, SHIP_TOP_M = 2023.8, 1993.4  # monolith top elevations in this repo (manifest)

# The unverified fit in unreal/import_scan.py (cm per scan unit etc.).
FIT_SCALE_CM = 31.1814
FIT_YAW_DEG, FIT_MIRROR_X, FIT_SCAN_CENTRE = 292.00, 1, (15.967, 21.806)
FIT_OFFSET_CM = (-7000.0, 40000.0, 21459.2)

# Search windows in raw scan units, read off a hillshade of the scan (see --out).
BOWL_Z = (95.0, 125.0)  # band along the bowl centre line
STAGE_BOX = (90.0, 215.0, 40.0, 160.0)  # x0, x1, z0, z1 around the stage
INTERIOR = (112.0, 156.0, 80.0, 120.0)  # under the roof, deck visible
ROOF_SEARCH = (95.0, 175.0, 55.0, 150.0)
REAR_BAND_Z = (80.0, 92.0)  # clean strip for the rear of the deck and the wall
ROCKS = {"south": (-140.0, 10.0, -90.0, 40.0), "north": (-140.0, -30.0, 190.0, 300.0)}

# Estimates the blockout used before the scan, for the overlay (metres).
OLD = {"stage_width": 24.4, "stage_depth": 21.3, "stage_front_setback": 3.0, "roof_thickness": 1.5,
       "retaining_wall_gap": 2.0, "retaining_wall_height": 6.0}


def load_scan(cache: Path | None):
    """Return (points[N,3] float32, faces[M,3] int32) in USD world space."""
    if cache and cache.exists():
        d = np.load(cache)
        return d["P"], d["F"]
    from pxr import Usd, UsdGeom  # usd-core

    stage = Usd.Stage.Open(str(SCAN))
    pts, faces, off = [], [], 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        p = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64) @ m[:3, :3] + m[3, :3]
        idx = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        pts.append(p.astype(np.float32))
        faces.append(idx + off)
        off += len(p)
    P, F = np.vstack(pts), np.vstack(faces).astype(np.int32)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, P=P, F=F)
    return P, F


def sample_surface(P, F, box, per_unit2=16, seed=1):
    """Uniform surface samples (and normals) of the triangles whose centroid is in box (x0, x1, z0, z1)."""
    c = P[F].astype(np.float64).mean(1)
    sel = (c[:, 0] > box[0]) & (c[:, 0] < box[1]) & (c[:, 2] > box[2]) & (c[:, 2] < box[3])
    A, B, C = (P[F[sel, i]].astype(np.float64) for i in range(3))
    cross = np.cross(B - A, C - A)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    k = np.maximum(1, np.ceil(area * per_unit2).astype(int))
    idx = np.repeat(np.arange(len(A)), k)
    rng = np.random.default_rng(seed)
    r1, r2 = rng.random(len(idx)), rng.random(len(idx))
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    Q = A[idx] + (B[idx] - A[idx]) * r1[:, None] + (C[idx] - A[idx]) * r2[:, None]
    N = cross[idx] / np.maximum(np.linalg.norm(cross[idx], axis=1, keepdims=True), 1e-12)
    return Q.astype(np.float32), N.astype(np.float32)


def binned_median(x, h, lo, hi, bw, q=50):
    edges = np.arange(lo, hi + bw, bw)
    ib = np.digitize(x, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    order = np.argsort(ib)
    ib_s, h_s = ib[order], h[order]
    uniq, start = np.unique(ib_s, return_index=True)
    for u, s0, e0 in zip(uniq, start, list(start[1:]) + [len(h_s)]):
        if 0 <= u < len(out):
            out[u] = np.percentile(h_s[s0:e0], q)
    return edges[:-1] + bw / 2, out


def hist_mode(vals, lo, hi, bw=0.25):
    h, e = np.histogram(vals, bins=np.arange(lo, hi + bw, bw))
    i = int(np.argmax(h))
    return float(e[i] + bw / 2)


def in_box(Q, box):
    return (Q[:, 0] > box[0]) & (Q[:, 0] < box[1]) & (Q[:, 2] > box[2]) & (Q[:, 2] < box[3])


def measure(P, F):
    """All measurements in scan units; returns a dict."""
    from scipy import ndimage

    Q, N = sample_surface(P, F, (-110, 230, 10, 200))
    r = {}

    # Bowl profile along x, then the ramp fit and the deck/knee positions.
    m = (Q[:, 2] > BOWL_Z[0]) & (Q[:, 2] < BOWL_Z[1]) & (Q[:, 0] > -100) & (Q[:, 0] < 125)
    xs, prof = binned_median(Q[m, 0], Q[m, 1], -100, 125, 0.25)
    ok = np.isfinite(prof)
    xs, prof = xs[ok], prof[ok]
    fit = (xs > -50) & (xs < 95)
    slope, icpt = np.polyfit(xs[fit], prof[fit], 1)
    plateau = float(np.median(prof[(xs > -90) & (xs < -72)]))
    knee = float((plateau - icpt) / slope)  # where the ramp line reaches the flat behind the top row
    r["ramp_slope_deg"] = math.degrees(math.atan(-slope))
    r["top_plateau_h"] = plateau

    # Deck level and the roof.
    deck_vals = Q[in_box(Q, INTERIOR) & (Q[:, 1] < 5), 1]
    mode = hist_mode(deck_vals, -6, 2, 0.5)
    deck_h = float(np.median(deck_vals[np.abs(deck_vals - mode) < 0.5]))
    r["deck_h"] = deck_h
    top_tread = float(slope * (knee + 1.0) + icpt)  # tread just below the top-row riser
    r["top_row_tread_h"] = top_tread
    r["rise_units"] = top_tread - deck_h
    r["top_row_x"], r["knee_x"] = knee + 1.0, knee

    rm = in_box(Q, ROOF_SEARCH) & (Q[:, 1] > 23.5) & (Q[:, 1] < 27.5) & (np.abs(N[:, 1]) > 0.8)
    R = Q[rm]
    res = 0.5
    xa, za = ROOF_SEARCH[0], ROOF_SEARCH[2]
    g = np.zeros((int((ROOF_SEARCH[3] - za) / res) + 1, int((ROOF_SEARCH[1] - xa) / res) + 1))
    np.add.at(g, (((R[:, 2] - za) / res).astype(int), ((R[:, 0] - xa) / res).astype(int)), 1)
    lab, n = ndimage.label(ndimage.binary_closing(g > 0, iterations=3))
    big = 1 + int(np.argmax(ndimage.sum(lab > 0, lab, range(1, n + 1))))
    ys, xs_ = np.nonzero(lab == big)
    roof = dict(x0=xa + xs_.min() * res, x1=xa + xs_.max() * res, z0=za + ys.min() * res, z1=za + ys.max() * res)
    r["roof"] = roof
    r["roof_top_h"] = hist_mode(R[(R[:, 0] > roof["x0"]) & (R[:, 0] < roof["x1"]) & (R[:, 2] > roof["z0"]) & (R[:, 2] < roof["z1"]), 1], 23.5, 27.5)
    foot = (Q[:, 0] > roof["x0"]) & (Q[:, 0] < roof["x1"]) & (Q[:, 2] > roof["z0"]) & (Q[:, 2] < roof["z1"])
    under = Q[foot & (N[:, 1] < -0.7) & (Q[:, 1] > deck_h + 12)]
    r["roof_under_h"] = hist_mode(under[:, 1], 18, 23, 0.5)

    # Deck front lip, rear and the wall behind, on clean strips.
    mc = in_box(Q, (95, 125, 95, 105)) | in_box(Q, (125, 190, 95, 105))
    xs_c, prof_c = binned_median(Q[mc, 0], Q[mc, 1], 95, 190, 0.5, q=5)
    gutter = xs_c[(xs_c > 100) & (xs_c < 120)][np.nanargmin(prof_c[(xs_c > 100) & (xs_c < 120)])]
    lip = next(x for x, h in zip(xs_c, prof_c) if x > gutter and abs(h - deck_h) < 0.8)
    mr = (Q[:, 2] > REAR_BAND_Z[0]) & (Q[:, 2] < REAR_BAND_Z[1]) & (Q[:, 0] > 100) & (Q[:, 0] < 215)
    xs_r, prof_r = binned_median(Q[mr, 0], Q[mr, 1], 100, 215, 0.5, q=5)
    ok = np.isfinite(prof_r)
    xs_r, prof_r = xs_r[ok], prof_r[ok]
    rear = float(xs_r[(xs_r > lip) & (prof_r > deck_h + 1.5)][0])
    wall = (xs_r > rear) & (xs_r < rear + 25)
    r["deck_lip_x"], r["deck_rear_x"] = float(lip), rear
    r["wall_top_h"] = float(prof_r[wall].max())
    r["wall_face_x"] = float(xs_r[wall][np.argmax(prof_r[wall] > deck_h + 0.5 * (prof_r[wall].max() - deck_h))])

    # Flat deck seen from above (lower bound on the deck width: the roof hides the rest).
    dm = in_box(Q, (60, 215, 30, 180)) & (np.abs(Q[:, 1] - deck_h) < 0.9) & (np.abs(N[:, 1]) > 0.9)
    gg = np.zeros((300, 310)); np.add.at(gg, (((Q[dm, 2] - 30) / 0.5).astype(int), ((Q[dm, 0] - 60) / 0.5).astype(int)), 1)
    mk = ndimage.binary_opening(ndimage.binary_closing(gg > 2, iterations=2), iterations=3)
    lab, n = ndimage.label(mk)
    comp = lab == lab[int((100 - 30) / 0.5), int((135 - 60) / 0.5)]
    rows = np.nonzero(comp[:, int((140 - 60) / 0.5)])[0]
    r["deck_flat_z"] = (30 + rows.min() * 0.5, 30 + rows.max() * 0.5)

    # Monolith peaks (for the scale cross-check).
    top = {}
    for k, b in ROCKS.items():
        s = in_box(P, b)
        top[k] = float(np.percentile(P[s, 1], 99.95))
    r["rock_tops"] = top
    return r, Q, N


def scale_checks(r):
    """Independent metres-per-unit estimates; returns dict name -> (m_per_unit, note)."""
    est = {}
    est["bench rise 30.48 m"] = OFFICIAL_RISE_M / r["rise_units"]
    est["roof underside = SkyDeck grid 12.24 m"] = ROOF_GRID_M / (r["roof_under_h"] - r["deck_h"])
    hi, lo = sorted(r["rock_tops"].values(), reverse=True)
    est["rock top difference 30.4 m"] = (CREATION_TOP_M - SHIP_TOP_M) / (hi - lo)
    return est


def fit_stage_world(r):
    """Where the unverified import_scan fit puts the stage centre, in terrain metres (E, N)."""
    x = (r["deck_lip_x"] + r["roof"]["x1"]) / 2
    z = (r["roof"]["z0"] + r["roof"]["z1"]) / 2
    s = FIT_SCALE_CM / 100.0
    yaw = math.radians(FIT_YAW_DEG)
    ux, uy = FIT_MIRROR_X * s * (x - FIT_SCAN_CENTRE[0]), s * (z - FIT_SCAN_CENTRE[1])
    return (math.cos(yaw) * ux - math.sin(yaw) * uy + FIT_OFFSET_CM[0] / 100,
            math.sin(yaw) * ux + math.cos(yaw) * uy + FIT_OFFSET_CM[1] / 100)


def convert(r, s):
    """Scan units to the stage-blockout dimensions, in metres, at scale s (m per unit)."""
    lip, roof = r["deck_lip_x"], r["roof"]
    d = {
        "stage_width": (roof["z1"] - roof["z0"]) * s,
        "stage_depth": (r["deck_rear_x"] - lip) * s,
        "roof_front_inset": (roof["x0"] - lip) * s,
        "roof_depth": (roof["x1"] - roof["x0"]) * s,
        "roof_top_above_deck": (r["roof_top_h"] - r["deck_h"]) * s,
        "roof_underside_above_deck": (r["roof_under_h"] - r["deck_h"]) * s,
        "roof_thickness": (r["roof_top_h"] - r["roof_under_h"]) * s,
        "retaining_wall_gap": (r["wall_face_x"] - r["deck_rear_x"]) * s,
        "retaining_wall_height": (r["wall_top_h"] - r["deck_h"]) * s,
        "deck_flat_width_lower_bound": (r["deck_flat_z"][1] - r["deck_flat_z"][0]) * s,
        "walkway_above_deck": (r["top_plateau_h"] - r["deck_h"]) * s,
        "rise_deck_to_top_row": r["rise_units"] * s,
    }
    d["roof_front_edge_to_wall_face"] = d["roof_front_inset"] + d["roof_depth"]
    return d


def diagnostics(P, F, Q, N, r, s, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from redrocks.stage import dim

    out.mkdir(parents=True, exist_ok=True)
    lip, zc = r["deck_lip_x"], (r["roof"]["z0"] + r["roof"]["z1"]) / 2
    xa, xb, za, zb = 90, 215, 40, 165
    res = 0.25
    m = in_box(Q, (xa, xb, za, zb))
    nx, nz = int((xb - xa) / res), int((zb - za) / res)
    top = np.full(nx * nz, -np.inf)
    np.maximum.at(top, ((Q[m, 2] - za) / res).astype(int) * nx + ((Q[m, 0] - xa) / res).astype(int), Q[m, 1])
    top = top.reshape(nz, nx)
    top[~np.isfinite(top)] = np.nan
    h_rel = (top - r["deck_h"]) * s

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(h_rel, origin="lower", extent=(xa, xb, za, zb), cmap="nipy_spectral", vmin=-3, vmax=16)
    plt.colorbar(im, ax=ax, shrink=0.7, label="height above deck (m), max per 0.25-unit cell")
    # new blockout: deck and roof (solid); the old estimate (dashed)
    hw = dim("stage_width") / s / 2
    ax.add_patch(Rectangle((lip, zc - hw), dim("stage_depth") / s, 2 * hw, fill=False, ec="white", lw=2, label="blockout deck"))
    ax.add_patch(Rectangle((lip, zc - hw), dim("roof_depth") / s, 2 * hw, fill=False, ec="magenta", lw=2, label="blockout roof"))
    ax.add_patch(Rectangle((lip, zc - OLD["stage_width"] / s / 2), OLD["stage_depth"] / s, OLD["stage_width"] / s,
                           fill=False, ec="black", lw=2, ls="--", label="old estimate (24.4 x 21.3 m)"))
    wx = lip + (dim("stage_depth") + dim("retaining_wall_gap")) / s
    ax.plot([wx, wx], [zc - 30, zc + 30], color="cyan", lw=2, label="blockout retaining wall face")
    ax.set_title("Stage area on the ml0w scan at %.3f m/unit (x toward stage, z across); outlines are the blockout" % s)
    ax.set_xlabel("scan x (units)"); ax.set_ylabel("scan z (units)")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out / "stage_topdown.png", dpi=90, bbox_inches="tight")
    plt.close(fig)

    # Cross-section along the stage axis (scan x) through the stage centre.
    fig, ax = plt.subplots(figsize=(15, 6))
    band = in_box(Q, (60, 215, zc - 2, zc + 2))
    ax.scatter((lip - Q[band, 0]) * s, (Q[band, 1] - r["deck_h"]) * s, s=0.4, c="0.3", label="scan, z = stage centre +-2 units")
    band2 = in_box(Q, (60, 215, 82, 90))
    ax.scatter((lip - Q[band2, 0]) * s, (Q[band2, 1] - r["deck_h"]) * s, s=0.4, c="tab:blue", label="scan, clean rear strip (z 82-90)")
    D, W = dim("stage_depth"), dim("stage_width")
    ax.add_patch(Rectangle((-D, -dim("deck_thickness")), D, dim("deck_thickness"), fc="none", ec="tab:red", lw=2, label="blockout deck"))
    gtop = ROOF_GRID_M
    ax.add_patch(Rectangle((-dim("roof_depth"), gtop), dim("roof_depth"), dim("roof_thickness"), fc="none", ec="magenta", lw=2, label="blockout roof"))
    ax.add_patch(Rectangle((-D - dim("retaining_wall_gap") - dim("retaining_wall_thickness"), -dim("retaining_wall_buried")),
                           dim("retaining_wall_thickness"), dim("retaining_wall_buried") + dim("retaining_wall_height"), fc="none", ec="cyan", lw=2, label="blockout retaining wall"))
    ax.set_xlim(-48, 62); ax.set_ylim(-4, 36); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("m along the stage axis, 0 = deck front lip, + toward the audience"); ax.set_ylabel("m above deck")
    ax.set_title("Cross-section along the stage axis (%.3f m/unit)" % s)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(out / "stage_section.png", dpi=90, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, help="npz cache of the scan points/faces (outside the repo)")
    ap.add_argument("--out", type=Path, help="directory for diagnostic PNGs")
    ap.add_argument("--scale", type=float, help="metres per scan unit; default is the mean of the independent checks")
    args = ap.parse_args()

    P, F = load_scan(args.cache)
    r, Q, N = measure(P, F)
    checks = scale_checks(r)
    s = args.scale or float(np.mean(list(checks.values())))
    print("scan units: deck h %.2f, roof top h %.2f, roof underside h %.2f, rise deck->top row %.1f" % (
        r["deck_h"], r["roof_top_h"], r["roof_under_h"], r["rise_units"]))
    print("ramp slope %.1f deg; stage x: lip %.1f, roof %.1f..%.1f, deck rear %.1f, wall face %.1f, wall top h %.1f" % (
        r["ramp_slope_deg"], r["deck_lip_x"], r["roof"]["x0"], r["roof"]["x1"], r["deck_rear_x"], r["wall_face_x"], r["wall_top_h"]))
    print("roof z %.1f..%.1f; flat deck seen z %.1f..%.1f" % (r["roof"]["z0"], r["roof"]["z1"], *r["deck_flat_z"]))
    print("scale checks (m per unit):")
    for k, v in checks.items():
        print("  %-42s %.3f" % (k, v))
    print("  %-42s %.3f   (unreal/import_scan.py FIT_SCALE)" % ("fit", FIT_SCALE_CM / 100))
    print("using %.3f m/unit; the fit scale is %.0f%% of that" % (s, 100 * FIT_SCALE_CM / 100 / s))
    e, n = fit_stage_world(r)
    print("stage centre under the import_scan fit: E %.0f m, N %.0f m (terrain frame; the bowl is at the origin)" % (e, n))
    dims = convert(r, s)
    print(json.dumps({k: round(v, 2) for k, v in dims.items()}, indent=2))
    if args.out:
        diagnostics(P, F, Q, N, r, s, args.out)
        print("wrote diagnostics to", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
