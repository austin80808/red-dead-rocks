"""Before | after | photo composites for the vegetation, and their colour error.

For each view in docs/media/vegetation-real/views.json, reads
``<shots>/before_<name>.png`` and ``<shots>/after_<name>.png`` (screenshots from
unreal/shoot_vegetation.py), fetches the view's Wikimedia Commons photo into
``--cache`` (never committed), and writes ``<out>/<name>.jpg``: three panels
cropped to the view's aspect, captioned "Before", "After" and the photo's
title, author and licence. Each view's ``samples`` pair a pixel (``photo_xy``/``game_xy``)
or a region (``photo_box``/``game_box`` with ``kind`` "green" or "brown") in the
photo with the same feature in the after shot; their median colours and CIE76 ΔE go
to ``<out>/delta_e.json`` and a printed table.

    .venv/bin/python tools/make_reference_compare.py --shots out/shots \
        --out docs/media/vegetation-real --cache out/refs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import requests  # noqa: F401  (fetch_photo's HTTP client; tests patch rc.requests.get)
from PIL import Image

# Shared with tools/make_board_compare.py (the photo board), not copied.
from redrocks.board_images import compose, fetch_photo  # noqa: F401
from redrocks.board_images import cover as _cover  # noqa: F401  (the old name)
from redrocks.board_images import photo_caption as _photo_caption
from redrocks.vegetation_metrics import delta_e, median_hex

REPO = Path(__file__).resolve().parent.parent
VIEWS = REPO / "docs" / "media" / "vegetation-real" / "views.json"


def load_views(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text())["views"]


def region_hex(img, box, kind: str) -> str:
    """Median colour, as hex, of the pixels in ``box`` (x0, y0, x1, y1) that look like
    ``kind``: "green" foliage (not sky) or "brown" bark."""
    x0, y0, x1, y1 = (int(v) for v in box)
    r = np.asarray(img, float)[y0:y1, x0:x1, :3].reshape(-1, 3)
    red, green, blue = r[:, 0], r[:, 1], r[:, 2]
    if kind == "green":
        keep = (green >= red * 0.95) & (green > blue * 1.1) & (blue < 200)
    elif kind == "brown":
        keep = (red > green) & (green > blue) & (red - blue > 25) & (red < 200)
    else:
        raise ValueError(kind)
    sel = r[keep] if keep.any() else r
    return "#%02X%02X%02X" % tuple(int(round(v)) for v in np.median(sel, 0))


def _sample(img, s: dict, which: str) -> str:
    if "%s_box" % which in s:
        return region_hex(img, s["%s_box" % which], s["kind"])
    return median_hex(img, s["%s_xy" % which])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shots", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--views", type=Path, default=VIEWS)
    ap.add_argument("--panel-width", type=int, default=640)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    report, missing = {}, []
    for view in load_views(args.views):
        name = view["name"]
        shots = [args.shots / ("%s_%s.png" % (k, name)) for k in ("before", "after")]
        if not all(p.exists() for p in shots):
            missing.append(name)
            continue
        before, after = (Image.open(p) for p in shots)
        photo = Image.open(fetch_photo(view["photo"], args.cache))
        compose(before, after, photo, ("Before", "After", _photo_caption(view["photo"])), tuple(view["aspect"]),
                args.panel_width).save(args.out / ("%s.jpg" % name), quality=88)
        rows = {}
        for s in view.get("samples", []):
            p_hex = _sample(np.asarray(photo.convert("RGB")), s, "photo")
            g_hex = _sample(np.asarray(after.convert("RGB")), s, "game")
            rows[s["label"]] = {"photo": p_hex, "game": g_hex, "delta_e": round(delta_e(p_hex, g_hex), 1)}
            print("%-10s %-10s photo %s  game %s  ΔE %5.1f" % (name, s["label"], p_hex, g_hex, rows[s["label"]]["delta_e"]))
        report[name] = rows
    (args.out / "delta_e.json").write_text(json.dumps(report, indent=1) + "\n")
    if missing:
        print("missing shots for: %s" % ", ".join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
