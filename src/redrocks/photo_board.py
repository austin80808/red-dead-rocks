"""The photo board: real Commons photos of Red Rocks, each with a game camera to match.

``docs/media/photo-board/views.json`` holds one entry per view::

    {"name", "photo": {"id", "title", "page", "file_url", "author", "licence", "size"},
     "aspect": [w, h], "camera_guess": {...}, "camera": {...} | null,
     "fit_error": float | null, "fit_note": str (optional),
     "sun": {"azimuth_deg", "elevation_deg"}, "sun_note": str,
     "landmarks": [{"name", "photo_xy": [x, y]}],
     "samples": [{"label", "kind", "photo_box", "game_box"}],
     "skyline_ignore": [[x0, y0, x1, y1]]}

Every pixel coordinate is in the photo's Commons thumb (``photo.size``, 1280 px wide
unless the original is smaller), which is also the size the game shot is cover-fitted
to before it is scored. Cameras are ``photo_camera.Camera`` dicts (level cm, Unreal
degrees, horizontal fov); landmark names are ``landmarks.catalog`` names. The sun's
azimuth is a compass bearing (clockwise from north), as ``setup_lighting.sun_rotation``
takes it.
"""

from __future__ import annotations

import json
from pathlib import Path

from .board_metrics import KINDS
from .photo_camera import Camera

PHOTO_KEYS = ("title", "page", "file_url", "author", "licence")
MIN_LANDMARKS = 6
MIN_SAMPLES = 4
ASPECT_TOLERANCE = 0.01  # aspect within 1 % of the thumb's shape


def load_board(path: Path) -> list[dict]:
    """The board's views, in file order."""
    return json.loads(Path(path).read_text())["views"]


def _box_problem(box, size) -> str | None:
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return "is not [x0, y0, x1, y1]"
    x0, y0, x1, y1 = box
    if not (x0 < x1 and y0 < y1):
        return "is empty"
    if size and not (0 <= x0 and 0 <= y0 and x1 <= size[0] and y1 <= size[1]):
        return "leaves the photo"
    return None


def _check_view(view: dict, landmark_names: set[str]) -> list[str]:
    name = view.get("name", "?")
    problems: list[str] = []

    def bad(msg: str) -> None:
        problems.append(f"{name}: {msg}")

    photo = view.get("photo") or {}
    for key in PHOTO_KEYS:
        if not photo.get(key):
            bad(f"photo has no {key}")
    size = photo.get("size")
    if not (isinstance(size, (list, tuple)) and len(size) == 2 and min(size) > 0):
        bad("photo has no size")
        size = None

    aspect = view.get("aspect")
    if not (isinstance(aspect, (list, tuple)) and len(aspect) == 2 and min(aspect) > 0):
        bad("has no aspect")
    elif size:
        want = size[0] / size[1]
        got = aspect[0] / aspect[1]
        if abs(got - want) > ASPECT_TOLERANCE * want:
            bad(f"aspect {aspect[0]}:{aspect[1]} does not match the photo's {size[0]}x{size[1]}")

    for key in ("camera_guess", "camera"):
        cam = view.get(key)
        if cam is None and key == "camera":
            continue
        try:
            Camera.from_dict(cam)
        except (KeyError, TypeError, ValueError):
            bad(f"{key} is not a camera")

    sun = view.get("sun") or {}
    for key in ("azimuth_deg", "elevation_deg"):
        if not isinstance(sun.get(key), (int, float)):
            bad(f"sun has no {key}")

    marks = view.get("landmarks") or []
    if view.get("fit_note"):
        if not marks:
            bad("has a fit_note but no landmarks to check its camera against")
    elif len(marks) < MIN_LANDMARKS:
        bad(f"has {len(marks)} landmarks, needs at least {MIN_LANDMARKS}")
    seen: set[str] = set()
    for mark in marks:
        mname = mark.get("name")
        if mname not in landmark_names:
            bad(f"unknown landmark {mname}")
        if mname in seen:
            bad(f"landmark {mname} twice")
        seen.add(mname)
        xy = mark.get("photo_xy")
        if not (isinstance(xy, (list, tuple)) and len(xy) == 2):
            bad(f"landmark {mname} has no photo_xy")
        elif size and not (0 <= xy[0] <= size[0] and 0 <= xy[1] <= size[1]):
            bad(f"landmark {mname} is outside the photo")

    samples = view.get("samples") or []
    if len(samples) < MIN_SAMPLES:
        bad(f"has {len(samples)} samples, needs at least {MIN_SAMPLES}")
    labels: set[str] = set()
    for sample in samples:
        label = sample.get("label", "?")
        if label in labels:
            bad(f"sample {label} twice")
        labels.add(label)
        if sample.get("kind") not in KINDS:
            bad(f"sample {label} has unknown kind {sample.get('kind')}")
        for key in ("photo_box", "game_box"):
            why = _box_problem(sample.get(key), size)
            if why:
                bad(f"sample {label} {key} {why}")

    for box in view.get("skyline_ignore") or []:
        why = _box_problem(box, size)
        if why:
            bad(f"skyline_ignore box {box} {why}")
    return problems


def validate_board(views: list[dict], landmark_names: set[str]) -> list[str]:
    """Problems with the board, each ``"<view>: <problem>"``; empty when it is valid.

    Each view needs a photo with title, page, file_url, author, licence and its
    thumb's size; an aspect within 1 % of that size; a sun with both angles; at
    least 6 landmarks, all names from ``landmark_names`` and inside the photo; at
    least 4 samples with kinds from ``board_metrics.KINDS``; and boxes inside the photo.
    A view whose landmarks cannot fit its camera (too few of the build's points are
    visible in the photo, or they disagree with it) says so in a ``fit_note`` and
    needs only one landmark, so its camera can still be checked against the photo.
    """
    problems: list[str] = []
    names = [v.get("name") for v in views]
    for n in sorted({n for n in names if names.count(n) > 1}):
        problems.append(f"{n}: view name used twice")
    for view in views:
        problems.extend(_check_view(view, landmark_names))
    return problems
