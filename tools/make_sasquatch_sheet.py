"""Compose the sasquatch renders into docs/media/sasquatch/.

Reads the screenshots unreal/render_sasquatch.py writes and makes:
    sasquatch_turnaround.jpg   front, three-quarter and side, side by side, captioned
    sasquatch_face.jpg         the face close-up
    first_person.jpg           first person looking down, and walking
    sasquatch_walk.gif         third person walking, then running

    python tools/make_sasquatch_sheet.py \
        --shots "$HOME/Documents/Unreal Projects/RedRocks-sasquatch/Saved/Screenshots/MacEditor" \
        --manifest out/sasquatch/manifest.json --out docs/media/sasquatch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PANELS = [("front", "Front"), ("three_quarter", "Three-quarter"), ("side", "Side")]


def _font(size: int):
    for name in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
                 "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _find(shots: Path, name: str) -> Path:
    found = sorted(shots.glob("sasquatch_%s*.png" % name), key=lambda p: p.stat().st_mtime)
    if not found:
        raise FileNotFoundError("no sasquatch_%s*.png in %s" % (name, shots))
    return found[-1]


def turnaround(shots: Path, manifest: dict | None, out: Path, panel_w: int = 640) -> Path:
    images = [Image.open(_find(shots, key)).convert("RGB") for key, _ in PANELS]
    panel_h = int(panel_w * images[0].height / images[0].width)
    caption_h = 70
    sheet = Image.new("RGB", (panel_w * len(images), panel_h + caption_h), (18, 16, 15))
    draw = ImageDraw.Draw(sheet)
    title = _font(30)
    small = _font(20)
    for i, (img, (_, label)) in enumerate(zip(images, PANELS)):
        sheet.paste(img.resize((panel_w, panel_h), Image.LANCZOS), (i * panel_w, 0))
        draw.text((i * panel_w + 18, panel_h + 12), label, fill=(235, 225, 210), font=title)
    if manifest:
        text = "%.2f m tall  |  shoulders %.2f m  |  fingertips %.2f m (knee %.2f m)  |  %s strands" % (
            manifest["height_cm"] / 100, manifest["shoulder_width_cm"] / 100,
            manifest["hanging_fingertip_cm"] / 100, manifest["knee_height_cm"] / 100,
            f"{manifest['strands']['body'] + manifest['strands']['head']:,}")
        w = draw.textlength(text, font=small)
        draw.text((sheet.width - w - 18, panel_h + 22), text, fill=(190, 175, 155), font=small)
    path = out / "sasquatch_turnaround.jpg"
    sheet.save(path, quality=90)
    return path


def single(shots: Path, name: str, out: Path, filename: str, width: int = 1200) -> Path:
    img = Image.open(_find(shots, name)).convert("RGB")
    img = img.resize((width, int(width * img.height / img.width)), Image.LANCZOS)
    path = out / filename
    img.save(path, quality=90)
    return path


def pair(shots: Path, names: tuple[str, str], out: Path, filename: str, width: int = 800) -> Path:
    imgs = [Image.open(_find(shots, n)).convert("RGB") for n in names]
    h = int(width * imgs[0].height / imgs[0].width)
    sheet = Image.new("RGB", (width * 2, h))
    for i, img in enumerate(imgs):
        sheet.paste(img.resize((width, h), Image.LANCZOS), (i * width, 0))
    path = out / filename
    sheet.save(path, quality=88)
    return path


def walk_gif(shots: Path, out: Path, width: int = 640) -> Path:
    frames = []
    for phase in ("walk", "run"):
        for f in sorted(shots.glob("sasquatch_%s_*.png" % phase)):
            img = Image.open(f).convert("RGB")
            frames.append(img.resize((width, int(width * img.height / img.width)), Image.LANCZOS))
    if not frames:
        raise FileNotFoundError("no walk or run frames in %s" % shots)
    path = out / "sasquatch_walk.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=220, loop=0, optimize=True)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shots", required=True, help="the editor's Saved/Screenshots/MacEditor folder")
    ap.add_argument("--manifest", default="out/sasquatch/manifest.json")
    ap.add_argument("--out", default="docs/media/sasquatch")
    args = ap.parse_args(argv)
    shots, out = Path(args.shots), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text()) if Path(args.manifest).exists() else None
    print(turnaround(shots, manifest, out))
    print(single(shots, "face", out, "sasquatch_face.jpg", 900))
    print(pair(shots, ("first_person_down", "first_person_walk"), out, "first_person.jpg"))
    try:
        print(walk_gif(shots, out))
    except FileNotFoundError as exc:
        print("skipped the GIF: %s" % exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
