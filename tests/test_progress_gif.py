"""The README's progress GIF: one frame per stage, built from docs/media/progress/stages.json."""

import importlib.util
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def tool():
    spec = importlib.util.spec_from_file_location("make_progress_gif", ROOT / "tools" / "make_progress_gif.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def frames(tmp_path, sizes):
    stages = []
    for i, (w, h) in enumerate(sizes):
        p = tmp_path / f"f{i}.jpg"
        Image.new("RGB", (w, h), (40 * i, 80, 120)).save(p)
        stages.append({"frame": str(p), "title": f"Stage {i}", "pr": f"#{40 + i}"})
    return stages


def test_one_frame_per_stage_at_one_size_with_a_caption(tmp_path):
    t = tool()
    stages = frames(tmp_path, [(1200, 675), (800, 450), (1280, 720)])
    out = tmp_path / "progress.gif"
    info = t.build(stages, out, width=480)
    gif = Image.open(out)
    assert gif.n_frames == 3 and info["frames"] == 3
    assert gif.size == (480, 270 + t.CAPTION_H)
    # The caption bar is dark; the picture above it keeps its colour.
    gif.seek(2)
    rgb = gif.convert("RGB")
    assert sum(rgb.getpixel((5, 270 + t.CAPTION_H - 3))) < 120
    assert rgb.getpixel((240, 100))[0] > 60


def test_the_last_stage_is_held_longest(tmp_path):
    t = tool()
    out = tmp_path / "progress.gif"
    t.build(frames(tmp_path, [(160, 90)] * 3), out, width=160)
    gif = Image.open(out)
    durations = []
    for k in range(gif.n_frames):
        gif.seek(k)
        durations.append(gif.info["duration"])
    assert durations[-1] > durations[0] == durations[1]


def test_the_committed_stage_list_points_at_real_frames_in_order():
    stages = json.loads((ROOT / "docs" / "media" / "progress" / "stages.json").read_text())["stages"]
    assert len(stages) >= 5
    for s in stages:
        assert (ROOT / s["frame"]).is_file(), s["frame"]
        assert s["title"] and s["pr"].startswith("#")
