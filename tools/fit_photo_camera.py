"""Fit each photo board view's game camera from its landmarks.

For every view in ``docs/media/photo-board/views.json`` (or one, with ``--only``),
looks up the view's landmarks in the build (``redrocks.landmarks.catalog`` over
``<out>/manifest.json`` and ``assets/scan/scan_top.npz``), fits a pinhole camera to
where they appear in the photo (``redrocks.photo_camera.fit_camera``, starting
from the view's ``camera_guess``) and writes ``camera`` and ``fit_error`` (mean
landmark error as a fraction of the photo's width, rounded to 4 places) back
into the file.

A fit with a mean error over 1.5 % of the width, a landmark behind the camera
or fewer than four landmarks is refused: the tool prints
``REFUSED <view>: <reason>``, leaves that view's ``camera`` and ``fit_error`` as
they were and exits 1. A refused view means its landmarks are wrong: fix them in
the views file and run it again.

    .venv/bin/python tools/fit_photo_camera.py [--views PATH] [--out out] [--only NAME]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from redrocks.landmarks import catalog, to_level_cm  # noqa: E402
from redrocks.photo_board import load_board  # noqa: E402
from redrocks.photo_camera import Camera, FitRefused, fit_camera  # noqa: E402

VIEWS = ROOT / "docs" / "media" / "photo-board" / "views.json"
SCAN_TOP = ROOT / "assets" / "scan" / "scan_top.npz"


def load_points(out_dir: Path) -> dict[str, tuple[float, float, float]]:
    """Every catalog landmark in level-frame centimetres."""
    return to_level_cm(catalog(Path(out_dir), SCAN_TOP))


def rounded(cam: Camera) -> dict:
    """The camera as written to the views file: cm to 0.1, degrees to 0.001."""
    d = cam.to_dict()
    return {k: round(v, 1 if k in ("x", "y", "z") else 3) for k, v in d.items()}


def fit_view(view: dict, points: dict) -> tuple[dict, float]:
    """(camera dict, mean error fraction) for one view; raises FitRefused."""
    marks = view["landmarks"]
    missing = [m["name"] for m in marks if m["name"] not in points]
    if missing:
        raise FitRefused("the build has no landmark " + ", ".join(missing))
    pts = np.array([points[m["name"]] for m in marks], dtype=float)
    pix = np.array([m["photo_xy"] for m in marks], dtype=float)
    size = tuple(view["photo"]["size"])
    cam, err = fit_camera(pts, pix, size, Camera.from_dict(view["camera_guess"]))
    return rounded(cam), round(float(err), 4)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--views", type=Path, default=VIEWS, help="board views file (rewritten in place)")
    ap.add_argument("--out", type=Path, default=ROOT / "out", help="build output with manifest.json")
    ap.add_argument("--only", help="fit only this view")
    args = ap.parse_args(argv)

    data = json.loads(args.views.read_text())
    views = load_board(args.views)
    if args.only and args.only not in {v["name"] for v in views}:
        print(f"no view named {args.only} in {args.views}", file=sys.stderr)
        return 2
    points = load_points(args.out)

    refused = 0
    for view in views:
        if args.only and view["name"] != args.only:
            continue
        try:
            cam, err = fit_view(view, points)
        except FitRefused as exc:
            refused += 1
            print(f"REFUSED {view['name']}: {exc}")
            continue
        view["camera"], view["fit_error"] = cam, err
        print(f"ok      {view['name']}: mean error {err * 100:.2f} % of the width, "
              f"{len(view['landmarks'])} landmarks, fov {cam['fov']:.1f}")

    data["views"] = views
    args.views.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
