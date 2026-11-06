"""Render a quick look at the stage from the top of the bowl, flat-shaded by material.

A plain numpy z-buffer over the build's OBJs (no Unreal, no GPU), for checking
the stage's shapes and arrangement against the photo board's ``postcard`` view
(Commons "Red Rocks Amphitheater.JPG", Scapler, CC BY-SA 3.0) while working on
them. Each OBJ gets one flat colour (tops and sides apart where the material
does that in Unreal), lit by a late-morning sun. The scan is not drawn; the
terrain under it, which follows the scan's floor, stands in for its rock.

    PYTHONPATH=src .venv/bin/python tools/render_stage_check.py [--out out] \\
        [--png out/check_stage.png] [--photo out/refs/postcard.jpg]

The camera stands at the top-row walkway's centre (``manifest.json``), 1.7 m up,
looking at the stage's front edge, 50 degrees across. With ``--photo`` it also
writes ``<png>_compare.png``: the render and the photo side by side.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent

# OBJ -> (side colour, top colour) in sRGB; top is for faces whose normal is mostly up.
COLOURS = {
    "RedRocks_Terrain": ("#8F5A40", "#9C6E52"),
    "Rocks": ("#A2482A", "#B07050"),
    "Walls": ("#B9A79B", "#C4B4A8"),
    "Buildings": ("#B67856", "#5A534D"),
    "Benches": ("#7B4632", "#C2B1A6"),
    "Bench_Planks": ("#9E8A70", "#B9A383"),
    "Bench_Brackets": ("#2E2D2B", "#2E2D2B"),
    "Stairs": ("#7B4632", "#C2B1A6"),
    "Handrail": ("#2E2D2B", "#2E2D2B"),
    "Booth": ("#6E2620", "#CD8F9A"),
    "Stage": ("#B9A79B", "#C9B5A8"),  # pale pinkish concrete
    "Stage_Steel": ("#A0482E", "#A0482E"),  # oxide-red paint
    "Stage_Roof": ("#6E2620", "#CD8F9A"),  # maroon metal; its top in sun reads pink
    "Stage_Fascia": ("#2A2422", "#2A2422"),
    "Stage_Towers": ("#8F5A44", "#C2B1A6"),  # stacked flagstone; cap-coloured tops
}
SKY_TOP, SKY_HORIZON = np.array([0x8E, 0xB8, 0xE6]), np.array([0xD9, 0xE6, 0xF0])
SUN_AZIMUTH_DEG, SUN_ELEVATION_DEG = 135.0, 55.0  # late morning, south-east
AMBIENT = 0.5
TERRAIN_RADIUS_M = 600.0
EYE_M = 1.7
NEAR_M = 0.3


def hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(vertices[N, 3] metres, triangles[M, 3] vertex indices) of an OBJ in cm."""
    verts, tris = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append(line.split()[1:4])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                tris += [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
    return np.asarray(verts, float).reshape(-1, 3) / 100.0, np.asarray(tris, int).reshape(-1, 3)


class Camera:
    """A pinhole camera at ``eye`` looking at ``target`` (metres), ``hfov_deg`` across a ``size`` image."""

    def __init__(self, eye, target, hfov_deg: float, size: tuple[int, int]):
        self.eye = np.asarray(eye, float)
        f = np.asarray(target, float) - self.eye
        self.f = f / np.linalg.norm(f)
        r = np.cross(self.f, [0.0, 0.0, 1.0])
        self.r = r / np.linalg.norm(r)
        self.u = np.cross(self.r, self.f)
        self.w, self.h = size
        self.focal = (self.w / 2) / math.tan(math.radians(hfov_deg) / 2)

    def project(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x pixel, y pixel, depth m) of points p[N, 3]."""
        d = np.asarray(p, float) - self.eye
        z = d @ self.f
        zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
        return (self.w / 2 + self.focal * (d @ self.r) / zs, self.h / 2 - self.focal * (d @ self.u) / zs, z)


def rasterise(cam: Camera, meshes: list[tuple[np.ndarray, np.ndarray, str]]) -> np.ndarray:
    """RGB image: every mesh's triangles flat-shaded, nearest wins."""
    w, h = cam.w, cam.h
    rows = np.linspace(0.0, 1.0, h)[:, None, None]
    img = (SKY_TOP * (1 - rows) + SKY_HORIZON * rows) * np.ones((1, w, 1))
    depth = np.full((h, w), np.inf)
    az, el = math.radians(SUN_AZIMUTH_DEG), math.radians(SUN_ELEVATION_DEG)
    sun = np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)])
    for verts, tris, name in meshes:
        side, top = (hex_rgb(c) for c in COLOURS[name])
        x, y, z = cam.project(verts)
        a, b, c = (verts[tris[:, k]] for k in range(3))
        n = np.cross(b - a, c - a)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        lit = AMBIENT + (1 - AMBIENT) * np.clip(n @ sun, 0.0, 1.0)
        base = np.where((n[:, 2] > 0.7)[:, None], top, side)
        colour = np.clip(base * lit[:, None], 0, 255)
        for k, t in enumerate(tris):
            zt = z[t]
            if zt.min() < NEAR_M:
                continue
            xt, yt = x[t], y[t]
            x0, x1 = max(int(math.floor(xt.min())), 0), min(int(math.ceil(xt.max())), w - 1)
            y0, y1 = max(int(math.floor(yt.min())), 0), min(int(math.ceil(yt.max())), h - 1)
            if x0 > x1 or y0 > y1:
                continue
            den = (yt[1] - yt[2]) * (xt[0] - xt[2]) + (xt[2] - xt[1]) * (yt[0] - yt[2])
            if abs(den) < 1e-12:
                continue
            px, py = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            l0 = ((yt[1] - yt[2]) * (px - xt[2]) + (xt[2] - xt[1]) * (py - yt[2])) / den
            l1 = ((yt[2] - yt[0]) * (px - xt[2]) + (xt[0] - xt[2]) * (py - yt[2])) / den
            l2 = 1 - l0 - l1
            inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
            if not inside.any():
                continue
            # Perspective-correct depth: interpolate 1/z.
            zz = 1.0 / (l0 / zt[0] + l1 / zt[1] + l2 / zt[2])
            sub = depth[y0:y1 + 1, x0:x1 + 1]
            win = inside & (zz < sub)
            sub[win] = zz[win]
            img[y0:y1 + 1, x0:x1 + 1][win] = colour[k]
    return img.astype(np.uint8)


def scene(out: Path, centre: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, str]]:
    meshes = []
    for name in COLOURS:
        path = out / f"{name}.obj"
        if not path.exists():
            continue
        verts, tris = load_obj(path)
        if name in ("RedRocks_Terrain", "Rocks", "Walls", "Buildings"):
            keep = np.linalg.norm(verts[:, :2] - centre, axis=1) < TERRAIN_RADIUS_M
            tris = tris[keep[tris].all(axis=1)]
        meshes.append((verts, tris, name))
    return meshes


def stage_camera(manifest: dict, hfov_deg: float, size: tuple[int, int]) -> Camera:
    """The top-row walkway's centre, eye height, looking at the stage's front edge."""
    st = manifest["stage"]
    base = st["base_elevation_m"]
    eye = (st["walkway_east_m"], st["walkway_north_m"], st["walkway_elevation_m"] - base + EYE_M)
    axis = np.array(st["axis_east_north"], float)
    front = np.array([st["front_edge_east_m"], st["front_edge_north_m"]]) - axis * 4.0
    target = (front[0], front[1], st["deck_top_elevation_m"] - base + 2.0)
    return Camera(eye, target, hfov_deg, size)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--png", type=Path, help="default: <out>/check_stage.png")
    ap.add_argument("--photo", type=Path, help="a photo to put beside the render (e.g. out/refs/postcard.jpg)")
    ap.add_argument("--fov", type=float, default=50.0, help="horizontal field of view, degrees")
    ap.add_argument("--size", type=int, nargs=2, default=(1280, 960))
    args = ap.parse_args(argv)

    manifest = json.loads((args.out / "manifest.json").read_text())
    cam = stage_camera(manifest, args.fov, tuple(args.size))
    st = manifest["stage"]
    img = Image.fromarray(rasterise(cam, scene(args.out, np.array([st["front_edge_east_m"], st["front_edge_north_m"]]))))
    png = args.png or args.out / "check_stage.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    img.save(png)
    print(f"wrote {png}")
    if args.photo:
        photo = Image.open(args.photo).convert("RGB").resize(img.size)
        both = Image.new("RGB", (img.width * 2, img.height))
        both.paste(img, (0, 0))
        both.paste(photo, (img.width, 0))
        d = ImageDraw.Draw(both)
        d.text((10, 10), f"build (flat colours, {args.fov:.0f} deg)", fill=(0, 0, 0))
        d.text((img.width + 10, 10), f"photo: {args.photo.name}", fill=(255, 255, 255))
        compare = png.with_name(png.stem + "_compare.png")
        both.save(compare)
        print(f"wrote {compare}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
