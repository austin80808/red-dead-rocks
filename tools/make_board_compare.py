"""Before | after | photo composites for the photo board, its GIF and its scores,
and the calibration check of the board's camera model.

For each view in docs/media/photo-board/views.json (--views) it reads
<shots>/<before><view>.png and <shots>/<after><view>.png (unreal/shoot_board.py),
fetches the view's Commons photo into --cache (out/refs, never committed) and
writes docs/media/photo-board/<pass>/ (--out):

  <view>.jpg   before | after | photo, each cropped to the view's aspect, captioned
               "Before", "After" and "<title> — <author>, <licence>" (--no-before,
               for a board with no before shots: photo | after)
  board.gif    each view's after panel next to its photo, 2.5 s a view
  scores.json  {view: board_metrics.score_view(after, photo)} and "worst": the
               --worst N (default 3) regions furthest from their photo

It prints the table (ΔE per region, and each view's skyline error with the share
of columns that had sky in both images, which the error rests on) and the worst
regions. Nothing is written unless every photo fetches (else "could not fetch
photo for <view>: <error>") and every shot is there and lit: a mean luminance
under 10/255 is refused ("<file> is black: Lumen or streaming not settled").
Both exit 1.

--calib checks the calibration shots instead (calib_<view>.png, from
shoot_board.py with REDROCKS_BOARD_CALIB=1): the magenta marker at each of a
view's landmarks must sit where photo_camera.project(camera, landmark) puts it,
within 0.5 % of the width on average ("projection convention mismatch"
otherwise, exit 1). Calibration shots are rendered without the tonemapper, so
they are not held to the luminance floor; the markers are emissive.
--write-landmarks writes <shots>/landmarks_cm.json, the landmark catalog in level
cm, which shoot_board.py reads to place the markers (the editor's Python has no
numpy): run it before a calibration shoot.

    PYTHONPATH=src .venv/bin/python tools/make_board_compare.py --write-landmarks --shots out/board
    PYTHONPATH=src .venv/bin/python tools/make_board_compare.py --shots out/board --pass p0 --calib
    PYTHONPATH=src .venv/bin/python tools/make_board_compare.py --shots out/board --pass p0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from redrocks import board_images
from redrocks.board_metrics import MIN_SHARED_SKY, score_view, worst_regions
from redrocks.photo_camera import Camera, project

REPO = Path(__file__).resolve().parent.parent
BOARD = REPO / "docs" / "media" / "photo-board"
VIEWS = BOARD / "views.json"
CACHE = REPO / "out" / "refs"
BUILD = REPO / "out"
SCAN_TOP = REPO / "assets" / "scan" / "scan_top.npz"
LANDMARKS_FILE = "landmarks_cm.json"
MIN_LUMINANCE = 10.0  # of 255: a darker shot is black or half-rendered
GIF_PANEL_W = 480
GIF_MS = 2500
CALIB_MAX = 0.005  # mean marker offset, as a fraction of the width


def load_views(path: Path) -> list[dict]:
    """The board's views (views.json's ``views``, the schema of redrocks.photo_board)."""
    return json.loads(Path(path).read_text())["views"]


def _rounded(score: dict) -> dict:
    regions = {label: dict(r, delta_e=round(r["delta_e"], 1)) for label, r in score["regions"].items()}
    err = score["skyline_error"]
    return {"regions": regions, "skyline_error": None if err is None else round(err, 4),
            "skyline_share": round(score["skyline_share"], 3)}


def _skyline_text(score: dict) -> str:
    share = "%.0f %% of the columns" % (100 * score["skyline_share"])
    if score["skyline_error"] is None:
        return "skyline n/a: sky in both images in only %s (under %.0f %%)" % (share, 100 * MIN_SHARED_SKY)
    return "skyline %.3f of the height over %s (sky in both)" % (score["skyline_error"], share)


def print_table(views: list[dict], scores: dict, worst: list) -> None:
    print("%-16s %-16s %-10s %-8s %-8s %6s" % ("view", "region", "kind", "photo", "game", "ΔE"))
    for v in views:
        s = scores[v["name"]]
        for label, r in s["regions"].items():
            print("%-16s %-16s %-10s %-8s %-8s %6.1f" % (v["name"], label, r["kind"], r["photo"], r["game"],
                                                      r["delta_e"]))
        print("%-16s %s" % (v["name"], _skyline_text(s)))
    print("worst regions:")
    for i, (view, label, de) in enumerate(worst, 1):
        print("  %d. %-16s %-16s ΔE %5.1f" % (i, view, label, de))


def write_gif(frames: list[Image.Image], path: Path, ms: int = GIF_MS) -> None:
    """One frame per view on a shared palette; frames padded to the largest."""
    w, h = max(f.width for f in frames), max(f.height for f in frames)
    canvas = []
    for f in frames:
        c = Image.new("RGB", (w, h), board_images.BACKGROUND)
        c.paste(f, (0, 0))
        canvas.append(c)
    sheet = Image.new("RGB", (w, h * len(canvas)))
    for k, f in enumerate(canvas):
        sheet.paste(f, (0, k * h))
    palette = sheet.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG) for f in canvas]
    quantized[0].save(path, save_all=True, append_images=quantized[1:], duration=ms, loop=0,
                      optimize=False, disposal=1)


def compare(views: list[dict], args) -> int:
    """Composites, GIF and scores; nothing is written unless every input is good."""
    photos = {}
    for v in views:
        try:
            with Image.open(board_images.fetch_photo(v["photo"], args.cache)) as img:
                photos[v["name"]] = img.convert("RGB")
        except Exception as exc:  # no network, an HTTP error, a corrupt cache file
            print("could not fetch photo for %s: %s" % (v["name"], exc), file=sys.stderr)
            return 1
    shots = {}
    for v in views:
        for kind, prefix in ([] if args.no_before else [("before", args.before)]) + [("after", args.after)]:
            path = args.shots / ("%s%s.png" % (prefix, v["name"]))
            if not path.exists():
                print("missing shot %s" % path, file=sys.stderr)
                return 1
            with Image.open(path) as img:
                shot = img.convert("RGB")
            if board_images.mean_luminance(shot) < MIN_LUMINANCE:
                print("%s is black: Lumen or streaming not settled" % path, file=sys.stderr)
                return 1
            shots[v["name"], kind] = shot

    out = args.out or BOARD / args.pass_name
    out.mkdir(parents=True, exist_ok=True)
    scores, frames = {}, []
    for v in views:
        name, photo, after = v["name"], photos[v["name"]], shots[v["name"], "after"]
        aspect = tuple(v["aspect"])
        if args.no_before:
            panels = [("photo", photo), ("After", after)]
        else:
            panels = [("Before", shots[name, "before"]), ("After", after), ("photo", photo)]

        def captions(items, width):
            return [board_images.photo_caption(v["photo"], width - 12) if c == "photo" else c for c, _ in items]

        board_images.compose_panels([img for _, img in panels], captions(panels, args.panel_width), aspect,
                                    args.panel_width).save(out / ("%s.jpg" % name), quality=88)
        gif = [p for p in panels if p[0] != "Before"]
        frames.append(board_images.compose_panels([img for _, img in gif], captions(gif, GIF_PANEL_W), aspect,
                                                  GIF_PANEL_W))
        scores[name] = _rounded(score_view(v, after, photo))
    worst = worst_regions(scores, args.worst)
    report = dict(scores)
    report["worst"] = [list(w) for w in worst]
    (out / "scores.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    write_gif(frames, out / "board.gif")
    print_table(views, scores, worst)
    print("wrote %d composites, board.gif and scores.json to %s" % (len(views), out))
    return 0


def _match(expected: np.ndarray, found: np.ndarray) -> list[tuple[int, int, float]]:
    """Greedy closest pairs (i, j, distance) between projected and found markers."""
    if not len(expected) or not len(found):
        return []
    d = np.linalg.norm(expected[:, None, :] - found[None, :, :], axis=2)
    pairs, used_e, used_f = [], set(), set()
    for flat in np.argsort(d, axis=None):
        i, j = divmod(int(flat), d.shape[1])
        if i not in used_e and j not in used_f:
            used_e.add(i)
            used_f.add(j)
            pairs.append((i, j, float(d[i, j])))
    return pairs


def calibrate_view(view: dict, path: Path, marks: dict) -> tuple[bool, str]:
    """(passed, detail) for one view's calibration shot."""
    if not view.get("camera"):
        return False, "no fitted camera"
    if not path.exists():
        return False, "no calibration shot %s" % path
    names = [lm["name"] for lm in view.get("landmarks", []) if lm["name"] in marks]
    if not names:
        return False, "none of the view's landmarks is in %s" % LANDMARKS_FILE
    with Image.open(path) as img:
        shot = img.convert("RGB")
    w, h = shot.size
    px, front = project(Camera.from_dict(view["camera"]), np.array([marks[n] for n in names]), (w, h))
    inside = front & (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    expected = px[inside]
    if not len(expected):
        return False, "none of the view's %d landmarks projects into the frame" % len(names)
    found = board_images.marker_pixels(shot)
    pairs = _match(expected, found)
    if len(pairs) < math.ceil(len(expected) / 2):
        return False, ("found %d of %d markers (too small to draw, hidden, or a dark shot: mean luminance %.0f/255)"
                       % (len(found), len(expected), board_images.mean_luminance(shot)))
    mean = float(np.mean([d for _, _, d in pairs]))
    detail = "%d of %d markers, mean offset %.1f px (%.2f %% of the width)" % (
        len(pairs), len(expected), mean, 100 * mean / w)
    if len(found) > len(pairs):
        detail += ", %d unmatched magenta blob(s)" % (len(found) - len(pairs))
    if mean / w > CALIB_MAX:
        return False, detail + ": over %.1f %%, projection convention mismatch" % (100 * CALIB_MAX)
    return True, detail


def check_calibration(views: list[dict], shots: Path) -> int:
    path = shots / LANDMARKS_FILE
    if not path.exists():
        print("no %s: run tools/make_board_compare.py --write-landmarks before the calibration shoot "
              "(shoot_board.py places its markers from it)" % path, file=sys.stderr)
        return 1
    marks = {name: tuple(xyz) for name, xyz in json.loads(path.read_text())["landmarks"].items()}
    failed = 0
    for v in views:
        passed, detail = calibrate_view(v, shots / ("calib_%s.png" % v["name"]), marks)
        print("%s%-16s %s" % ("ok    " if passed else "FAIL  ", v["name"], detail))
        failed += not passed
    print("calibration: %d of %d views within %.1f %% of the width" % (len(views) - failed, len(views),
                                                                       100 * CALIB_MAX))
    return 1 if failed else 0


def write_landmarks(path: Path, build: Path, scan_top: Path) -> int:
    from redrocks import landmarks  # numpy and the scan raster: only this mode needs them

    points = landmarks.to_level_cm(landmarks.catalog(build, scan_top))
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"about": "Photo-board landmarks in the Unreal level frame (cm): redrocks.landmarks.catalog(%s, %s) "
                    "through to_level_cm. unreal/shoot_board.py puts a calibration marker at each; "
                    "tools/make_board_compare.py --calib projects the same points." % (build, scan_top),
           "landmarks": {name: [round(c, 1) for c in xyz] for name, xyz in sorted(points.items())}}
    path.write_text(json.dumps(doc, indent=1) + "\n")
    print("wrote %s (%d landmarks)" % (path, len(points)))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shots", type=Path, required=True, help="the shots folder, e.g. out/board")
    ap.add_argument("--pass", dest="pass_name", help="the pass, e.g. p0 (writes docs/media/photo-board/<pass>/)")
    ap.add_argument("--before", default="before_", help="before shots' prefix (default before_)")
    ap.add_argument("--after", default="after_", help="after shots' prefix (default after_)")
    ap.add_argument("--no-before", action="store_true", help="no before shots: photo | after composites")
    ap.add_argument("--views", type=Path, default=VIEWS)
    ap.add_argument("--cache", type=Path, default=CACHE, help="Commons photo cache (default out/refs)")
    ap.add_argument("--out", type=Path, help="output folder (default docs/media/photo-board/<pass>)")
    ap.add_argument("--panel-width", type=int, default=640)
    ap.add_argument("--worst", type=int, default=3, metavar="N", help="how many worst regions to list")
    ap.add_argument("--only", help="comma list of views (default: the whole board)")
    ap.add_argument("--calib", action="store_true", help="check the calibration shots instead")
    ap.add_argument("--write-landmarks", action="store_true", help="write <shots>/landmarks_cm.json and stop")
    ap.add_argument("--build", type=Path, default=BUILD, help="the build for --write-landmarks (default out)")
    ap.add_argument("--scan-top", type=Path, default=SCAN_TOP)
    args = ap.parse_args(argv)
    if args.write_landmarks:
        return write_landmarks(args.shots / LANDMARKS_FILE, args.build, args.scan_top)
    views = load_views(args.views)
    if args.only:
        wanted = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = sorted(set(wanted) - {v["name"] for v in views})
        if unknown:
            print("not on the board: %s" % ", ".join(unknown), file=sys.stderr)
            return 1
        views = [v for v in views if v["name"] in wanted]
    if not views:
        print("no views in %s" % args.views, file=sys.stderr)
        return 1
    if args.calib:
        return check_calibration(views, args.shots)
    if not args.pass_name:
        ap.error("--pass is required to write the composites")
    return compare(views, args)


if __name__ == "__main__":
    sys.exit(main())
