# Realistic Trees and Shrubs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Red Rocks hillside trees and shrubs match summer photos of Red Rocks Park in form,
scale, colour and density, proven by before | after | photo captures that Austin approves.

**Architecture:** The seeded Python generator (`src/redrocks/vegetation.py`) keeps writing three OBJ
parts per model, Pillow leaf atlases and `vegetation.json`. Atlas painting moves to its own module and
paints at real size. A new metrics module measures the spec's numbers: leaf size, crown fill, canopy
cover and ΔE. `unreal/setup_vegetation.py` gets new colours, a distance blend on the bark and per-part
cull distances. A new compare tool builds the side-by-sides from views framed to match the photos.
PR 2 adds grass after the seating PR merges.

**Tech Stack:** Python 3.11+, numpy, Pillow, requests (already dependencies), pytest; Unreal 5.8
Python (`unreal` module), Nanite HISM.

**Spec:** `docs/superpowers/specs/2026-11-04-realistic-vegetation-design.md`. Read it first.

## Global Constraints

- **No new dependencies.** numpy, Pillow and requests only (`pyproject.toml` unchanged; merge-gate
  holds dependency changes for Austin).
- **Nothing downloaded is committed.** The compare tool fetches Commons photos into `out/refs/`
  (gitignored). Only the composites are committed, with each photo's title, author and licence on
  them.
- **Models are deterministic.** The same name and index give the same OBJ bytes, as today.
- **Format unchanged.** OBJ in cm, X east, Y north, Z up, trunk base at the origin. Parts are `Trunk`,
  `Crown`, `Leaves`. Each `vegetation.json` instance row is `[x, y, z, yaw, pitch, roll, scale_xy,
  scale_z]`.
- **Colour targets (photo, sunlit summer):** juniper `#4B583E`–`#545E4D`, Gambel oak
  `#525D21`–`#656F3D`, ponderosa needles `#68744B`, ponderosa trunk at distance `#5A5236`, summer grass
  `#96914D`–`#B9A775`. Mahogany is greyer and lighter than oak, judged by eye.
- **Frame budget:** each matched view and the bowl-top, hillside and aerial views may be at most 15 %
  slower than the baseline measured in Task 2.
- **PR 1 touches only:** `src/redrocks/vegetation.py`, the new `src/redrocks/foliage_atlas.py` and
  `src/redrocks/vegetation_metrics.py`, `unreal/setup_vegetation.py`, `unreal/shoot_vegetation.py`,
  the new `tools/make_reference_compare.py`, `docs/media/vegetation-real/`, tests, the README's
  Vegetation section and `LICENSE`'s third-party list. Never `setup_materials.py`, `seating.py`,
  `stage.py`, `cli.py` or `setup_gameplay.py` (the seating PR owns them).
- **Git:** work only in the worktree `~/dev/redrocks-ue-trees`, on branch `feat/realistic-vegetation`.
  Never stash, reset or switch branches in `~/dev/redrocks-ue` (the seating session's checkout).
- **Unreal:** only in `~/Documents/Unreal Projects/RedRocks-trees`. Before any `UnrealEditor` or
  `UnrealEditor-Cmd` run, call `~/.local/bin/unreal-lock acquire "Red Rocks" "realistic vegetation"`.
  If someone else holds the lock, use `unreal-lock wait` instead. Run
  `~/.local/bin/unreal-lock release "Red Rocks"` as soon as the editor quits.
- **Commands used below:**
  - `UE=/Users/Shared/Epic Games/UE_5.8/Engine/Binaries/Mac/UnrealEditor-Cmd` (headless)
  - `UEG=/Users/Shared/Epic Games/UE_5.8/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor` (GUI)
  - `PROJ=$HOME/Documents/Unreal Projects/RedRocks-trees/RedRocks.uproject`
  - `WT=$HOME/dev/redrocks-ue-trees`
  - Every Unreal run gets `REDROCKS_OUT_DIR=$WT/out`.
- **Snow wiring:** `setup_vegetation.py` rebuilds its materials from scratch, which drops the snow
  lerp. After every `setup_vegetation.py` run, also run `setup_snow.py`.
- **Merge:** only `merge-gate <PR> --tested "<command, result>"`. The repo has no CI.

## Review Focus

These failure modes are implied by the spec but not covered by any single feature test. Each has a
test in the task named.

1. **A new model has no material, cull rule or bark group.** `OakPatch` and the fifth ponderosa must
   map to bark, foliage and cull settings, or `setup_vegetation.py` raises mid-import (Task 9:
   `test_every_model_maps_to_materials_and_cull`).
2. **Thicket patches overhang roads, paths, benches, the stage or the sled run.** A 3–6 m patch whose
   centre clears the mask can still spread over it (Task 8:
   `test_shrub_canopy_stays_off_masked_cells`).
3. **Snow stops whitening the new plants**, because the vegetation materials are rebuilt (Task 9:
   `test_snow_runs_after_vegetation_and_covers_veg_materials`).
4. **Instance count blows up and the import crawls.** On the real terrain, the total must stay at
   250,000 or fewer (Task 8 real-build step; a test bounds the synthetic-ground count).
5. **Leaf-card fade leaves floating stems or bare shrubs up close.** Stems must cull with their leaves
   and crowns must never cull (Task 9: `test_shrub_parts_cull_together_and_crowns_never`).

---

### Task 1: Matched views, compare tool and ΔE

**Files:**
- Create: `src/redrocks/vegetation_metrics.py` (colour part only)
- Create: `tools/make_reference_compare.py`
- Create: `docs/media/vegetation-real/views.json`
- Modify: `unreal/shoot_vegetation.py`
- Test: `tests/test_vegetation_metrics.py`, `tests/test_reference_compare.py`

Test files import `from redrocks import vegetation_metrics as vm` and `from redrocks import foliage_atlas as fa` (`fa` from Task 4 on); `m` is taken by loop variables in `tests/test_vegetation.py`.

**Interfaces:**
- Produces, in `redrocks.vegetation_metrics`:
  - `rgb_to_lab(rgb: np.ndarray) -> np.ndarray` (sRGB 0–255, shape (..., 3), D65)
  - `hex_to_lab(c: str) -> np.ndarray`
  - `delta_e(a: str | np.ndarray, b: str | np.ndarray) -> float` (CIE76 ΔE*ab; accepts hex or Lab)
  - `median_hex(img: np.ndarray, xy: tuple[int, int], r: int = 10) -> str`
- Produces, in `tools/make_reference_compare.py`:
  - `load_views(path: Path) -> list[dict]`
  - `fetch_photo(photo: dict, cache_dir: Path) -> Path`
  - `compose(before: Image, after: Image, photo: Image, captions: tuple[str, str, str], aspect: tuple[int, int], panel_w: int = 640) -> Image`
  - `main(argv: list[str] | None = None) -> int`, which takes `--shots DIR --out DIR --cache DIR`
- Produces: the schema of `views.json`. Each view is `{"name", "photo": {"id", "title", "page",
  "file_url", "author", "licence"}, "camera": {"x", "y", "z", "pitch", "yaw", "fov"}, "aspect": [w, h],
  "samples": [{"label", "photo_xy", "game_xy", "target"}]}`, all in cm and degrees.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_vegetation_metrics.py
def test_delta_e_of_identical_colours_is_zero():
    assert vm.delta_e("#505B46", "#505B46") == pytest.approx(0.0, abs=1e-6)

def test_delta_e_black_to_white_is_100():
    assert vm.delta_e("#000000", "#FFFFFF") == pytest.approx(100.0, abs=0.5)

def test_median_hex_ignores_outliers():
    img = np.full((50, 50, 3), (0x50, 0x5B, 0x46), np.uint8); img[25, 25] = 255
    assert vm.median_hex(img, (25, 25), r=5) == "#505B46"

# tests/test_reference_compare.py
def test_views_json_credits_every_photo_and_names_are_unique():
    views = rc.load_views(REPO / "docs/media/vegetation-real/views.json")
    assert {v["name"] for v in views} == {"trail", "ponderosa", "junipers", "hillsides"}
    for v in views:
        assert v["photo"]["page"].startswith("https://commons.wikimedia.org/wiki/File:")
        assert v["photo"]["licence"] and v["photo"]["author"] and v["photo"]["file_url"]
        assert set(v["camera"]) == {"x", "y", "z", "pitch", "yaw", "fov"}

def test_compose_is_three_panels_with_a_caption_strip():
    out = rc.compose(*(Image.new("RGB", (1600, 900), c) for c in "rgb"), ("a", "b", "c"), (16, 9), panel_w=320)
    assert out.width == 3 * 320 and out.height > 180  # 180 px panels plus the caption strip

def test_main_writes_composites_and_delta_e(tmp_path, monkeypatch):
    # synthetic before_/after_ shots + views file; fetch_photo monkeypatched to a local image
    ...
    assert rc.main(["--shots", str(shots), "--out", str(out), "--cache", str(tmp_path), "--views", str(vf)]) == 0
    assert (out / "trail.jpg").exists()
    de = json.loads((out / "delta_e.json").read_text())
    assert set(de["trail"]) == {"juniper"}
```

- [ ] **Step 2: Run the tests and check they fail**

Run: `cd $WT && .venv/bin/pytest tests/test_vegetation_metrics.py tests/test_reference_compare.py -v`
(make the worktree venv first: `python3 -m venv .venv && .venv/bin/pip install -e . pytest`)
Expected: FAIL (modules missing).

- [ ] **Step 3: Implement the metrics colour functions and the compare tool**

The tool reads `before_<name>.png` and `after_<name>.png` from `--shots`. Each panel is cover-fit and
centre-cropped to the view's aspect. Captions are "Before", "After" and
"<title> — <author>, <licence>". The tool writes `<out>/<name>.jpg` at quality 88, plus
`<out>/delta_e.json`, shaped `{view: {label: {"photo", "game", "delta_e"}}}`. It prints a table.
`fetch_photo` uses `requests` with User-Agent `redrocks-ue-reference/0.1` and caches the file at
`cache_dir/<id>.jpg`. `--views` defaults to the repo's `views.json`.

- [ ] **Step 4: Write `views.json` with the four photos**

Photos: ref09, ref02, ref15 and ref12, with the pages, authors and licences in the spec's table.
`file_url` is the Commons 1280 px thumb (from the API's `iiurlwidth=1280`). Camera values start at 0
and are filled in Task 2. `samples` start empty.

- [ ] **Step 5: Teach `shoot_vegetation.py` the matched views**

Add `VIEWS_PATH = os.environ.get("REDROCKS_VIEWS", <dir of this file>/../docs/media/vegetation-real/views.json)`.
Each view whose camera isn't all zeros is appended after `aerial` as
`(name, (x, y, z), {"pitch", "yaw", "roll": 0})`. Set the player camera's FOV to the view's `fov`, and
restore it afterwards. Screenshots keep the `<prefix>veg_<name>` naming. Add a plain-Python helper
`matched_views(views: list[dict]) -> list[tuple]` and test it in `tests/test_vegetation.py`, using the
same fake-`unreal` module the `sv` fixture uses.

- [ ] **Step 6: Run the tests and check they pass**

Run: `.venv/bin/pytest tests/test_vegetation_metrics.py tests/test_reference_compare.py tests/test_vegetation.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/redrocks/vegetation_metrics.py tools/make_reference_compare.py docs/media/vegetation-real/views.json unreal/shoot_vegetation.py tests/test_vegetation_metrics.py tests/test_reference_compare.py tests/test_vegetation.py
git commit -m "feat(vegetation): photo-matched views, before|after|photo compare tool, ΔE"
```

### Task 2: Baseline: project copy, matched cameras, before shots and frame rate

**Files:**
- Modify: `docs/media/vegetation-real/views.json` (cameras and samples)
- Create: `docs/media/vegetation-real/baseline.json` (frame rate per view)

**Interfaces:**
- Consumes: Task 1's `views.json` schema, `shoot_vegetation.py` and the compare tool.
- Produces: `out/shots/before_<view>.png` for all 7 views, and `baseline.json`, shaped
  `{view: {"fps", "worst_fps"}}`, which Task 10 compares against.

- [ ] **Step 1: Refresh the project copy**

The old `RedRocks-trees` has an earlier agent's uncommitted changes. Move it aside rather than
deleting it: `mv "$HOME/Documents/Unreal Projects/RedRocks-trees" "$HOME/Documents/Unreal Projects/RedRocks-trees-old-20261104"`.
Then clone the real project: `cp -cR "$HOME/Documents/Unreal Projects/RedRocks" "$HOME/Documents/Unreal Projects/RedRocks-trees"`.
Check that `git -C ".../RedRocks-trees" log --oneline -1` shows `f9acad9`.

- [ ] **Step 2: Clone the build output**

`cp -cR ~/dev/redrocks-ue/out $WT/out`. This is an APFS clone, so it's cheap and the source isn't
touched. Its `Benches.obj` is the seating branch's, which inside the bowl is masked anyway.

- [ ] **Step 3: Frame the four matched cameras**

Take the Unreal lock. Open the editor GUI on `$PROJ` and find each spot through the Unreal MCP server
(`http://127.0.0.1:8000/mcp`, skill `unreal-mcp`, `call_tool`), moving the viewport camera and taking
screenshots:
- **trail:** on a park path below a fin of rock, looking along the path with scrub on both sides and
  a rock fin skyline, like ref09.
- **ponderosa:** a lone, well-grown pine 25–40 m away against the sky, the whole tree in frame, like
  ref02 (portrait, aspect [3, 4]).
- **junipers:** two or three junipers 8–20 m away in front of red rock, like ref15.
- **hillsides:** from a high point looking across vegetated slopes to a ridge, like ref12.

Write each camera (x, y, z cm; pitch, yaw, fov) into `views.json`. Pick spots outside the scanned
bowl, where the generated plants grow.

- [ ] **Step 4: Shoot the baseline**

Run the shot script in the foreground:
`REDROCKS_OUT_DIR=$WT/out REDROCKS_SHOT_PREFIX=before_ "$UEG" "$PROJ" -nosplash -ini:EditorPerProjectUserSettings:[/Script/UnrealEd.EditorPerformanceSettings]:bThrottleCPUWhenNotForeground=False -ExecCmds="py $WT/unreal/shoot_vegetation.py"`.
Copy `Saved/Screenshots/MacEditor/before_veg_*.png` to `$WT/out/shots/before_<view>.png`. Read the
fps lines from `/tmp/redrocks_vegetation_shots.log`. Run it 3 times and record the median fps per
view in `baseline.json`. Release the lock.

- [ ] **Step 5: Pick the sample points**

For each matched view, add `samples` with the photo pixel `photo_xy` of each species visible in it:
juniper and oak (trail), needles and trunk (ponderosa), juniper (junipers), and shrub (hillsides).
Set `target` to the spec colour. `game_xy` is the same feature in the before shot. Run
`.venv/bin/python tools/make_reference_compare.py --shots out/shots --out out/compare-before --cache out/refs`
and check that every sample lands on its species in both images (open the composites).

- [ ] **Step 6: Commit**

```bash
git add docs/media/vegetation-real/views.json docs/media/vegetation-real/baseline.json
git commit -m "chore(vegetation): matched cameras and baseline frame rate"
```

### Task 3: Shape metrics

**Files:**
- Modify: `src/redrocks/vegetation_metrics.py`, `src/redrocks/vegetation.py` (`Model.tips`)
- Test: `tests/test_vegetation_metrics.py`

**Interfaces:**
- Consumes: `vegetation.Model`, `vegetation.Mesh`.
- Produces:
  - `Model.tips: list` (new last field, default empty): branch-tip points in metres
  - `side_fill(model: Model, atlas_alpha: np.ndarray | None = None, cell_m: float = 0.05, view: str = "y") -> float`:
    Trunk and Crown triangles count as solid. A leaf cell counts only where the atlas alpha at its UV
    is ≥ 0.4, or always when `atlas_alpha` is None. Returns filled cells ÷ cells inside the convex
    hull of the filled cells.
  - `card_widths(model: Model) -> np.ndarray`: the width of each leaf quad in metres (corner 0 to 1)
  - `tip_fraction(model: Model, radius_m: float = 1.0) -> float`: the share of leaf cards whose base
    is within `radius_m` of a `Model.tips` point
  - `trunk_diameter_at(model: Model, z_m: float = 1.4) -> float`: from Trunk vertices with z in
    `[z_m - 0.1, z_m + 0.1]`
  - `crown_base_fraction(model: Model) -> float`: the lowest Leaves z ÷ `height_m`
  - `footprint_radius_m(model: Model) -> float`: the 90th percentile XY distance of the Crown and
    Leaves vertices
  - `canopy_cover(instances: dict, models: dict, cells_xy_cm: np.ndarray, species: tuple[str, ...]) -> float`:
    the share of the given points that lie under a footprint circle of one of the named species
    (radius × `scale_xy`)

- [ ] **Step 1: Write the failing tests**

```python
def test_side_fill_of_a_solid_blob_is_near_one():
    model = _model_with(crown=blob at origin r=1)            # Crown only
    assert vm.side_fill(model) > 0.9

def test_side_fill_of_two_far_blobs_is_low():
    model = _model_with(crown=[blob at (-3,0,0) r=0.5, blob at (3,0,0) r=0.5])
    assert vm.side_fill(model) < 0.3

def test_card_widths_and_tip_fraction():
    model = _model_with(cards=[(base=(0,0,1), width=0.4)], tips=[(0, 0, 1.5)])
    assert vm.card_widths(model) == pytest.approx([0.4])
    assert vm.tip_fraction(model, 1.0) == 1.0 and vm.tip_fraction(model, 0.2) == 0.0

def test_trunk_diameter_of_a_cylinder():
    model = _model_with(trunk=tube r=0.3 from z=-0.5 to 5)
    assert vm.trunk_diameter_at(model, 1.4) == pytest.approx(0.6, rel=0.05)

def test_canopy_cover_of_one_shrub():
    # one Oak_0 instance at the origin, scale 1: points inside its footprint are covered
    by_name = {x.name: x for x in models}
    r = vm.footprint_radius_m(by_name["Oak_0"]) * 100
    pts = np.array([[0, 0], [r * 0.5, 0], [r * 2, 0], [0, r * 3]])
    assert vm.canopy_cover({"Oak_0": [[0, 0, 0, 0, 0, 0, 1, 1]]}, by_name, pts, ("Oak",)) == 0.5
```

`_model_with` is a small test helper that builds a `Model` from `add_blob`, `add_card` and `add_tube`.

- [ ] **Step 2: Run the tests and check they fail**

Run: `.venv/bin/pytest tests/test_vegetation_metrics.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement**

`side_fill` rasterises each triangle projected onto the plane facing `view`, on a `cell_m` grid, with
vectorised barycentric tests per triangle bounding box. Leaf alpha is sampled at each cell centre's
interpolated UV. The hull is `scipy`-free: use a monotone-chain convex hull of the filled cell
centres, then a point-in-polygon test on the grid. Record `Model.tips` in `_ponderosa`: the last
point of every branch path, and the top.

- [ ] **Step 4: Run the tests and check they pass, including the existing suite**

Run: `.venv/bin/pytest tests -q`. Expected: all pass.

- [ ] **Step 5: Commit**

`git commit -m "feat(vegetation): metrics for crown fill, leaf scale, trunk size and canopy cover"`

### Task 4: Real-size leaf atlases

**Files:**
- Create: `src/redrocks/foliage_atlas.py` (moves `ATLASES`, `_hex`, the painters and `paint_atlas` out of `vegetation.py`)
- Modify: `src/redrocks/vegetation.py` (`from .foliage_atlas import ATLASES, CARD_M, paint_atlas`; `ATLAS_SIZE` stays in `vegetation.py` so the CLI test's monkeypatch works)
- Test: `tests/test_vegetation.py`

**Interfaces:**
- Produces:
  - `CARD_M = {"Juniper": 0.32, "Pine": 0.48, "Oak": 0.38, "Mahogany": 0.28}` (card width in m)
  - `SIZE_M = {"Juniper": (0.10, 0.20), "Pine": (0.10, 0.18), "Oak": (0.05, 0.12), "Mahogany": (0.02, 0.035)}`:
    the spray, needle or leaf length in m
  - `JUNIPER_BRANCHLET_M = (0.001, 0.002)`
  - `PHOTO_COLOUR = {"Juniper": "#505B46", "Pine": "#68744B", "Oak": "#5C662F"}` (midpoints of the spec ranges)
  - `paint_atlas(atlas: str, size: int = 1024, supersample: int = 2) -> np.ndarray`: same contract as today
  - `painted_sizes(atlas: str, size: int = 256) -> dict[str, list[float]]`: what the painter drew, in m.
    Keys are `"leaf"` (or `"spray"`/`"needle"`) and, for juniper, `"branchlet"`
  - `mean_colour(img: np.ndarray) -> str`: alpha-weighted mean RGB as hex

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("atlas", veg.ATLASES)
def test_atlas_paints_at_real_size(atlas):
    sizes = fa.painted_sizes(atlas)
    lo, hi = fa.SIZE_M[atlas]
    main = sizes.get("leaf") or sizes.get("spray") or sizes.get("needle")
    assert main and lo <= np.median(main) <= hi and min(main) >= lo * 0.7 and max(main) <= hi * 1.3
    if atlas == "Juniper":
        b = sizes["branchlet"]; assert 0.001 <= np.median(b) <= 0.002

@pytest.mark.parametrize("atlas", ["Juniper", "Pine", "Oak"])
def test_atlas_colour_is_near_the_photo(atlas):
    assert vm.delta_e(fa.mean_colour(fa.paint_atlas(atlas, 256)), fa.PHOTO_COLOUR[atlas]) <= 20

def test_mahogany_is_greyer_and_lighter_than_oak():
    oak, mah = (vm.hex_to_lab(fa.mean_colour(fa.paint_atlas(a, 256))) for a in ("Oak", "Mahogany"))
    assert mah[0] > oak[0] and np.hypot(*mah[1:]) < np.hypot(*oak[1:])
```

The existing `test_atlases_have_cutout_alpha_and_no_black_fringe` stays as is.

- [ ] **Step 2: Run the tests and check they fail**

Run: `.venv/bin/pytest tests/test_vegetation.py -k "atlas or mahogany" -v`. Expected: FAIL.

- [ ] **Step 3: Implement**

Each painter gets `px_per_m = tile / CARD_M[atlas]`, draws every size in metres times `px_per_m`, and
appends what it drew to a recorder list. Leaf counts rise so the tile stays 20–60 % covered. Palettes
centre on `PHOTO_COLOUR`. Mahogany uses its own greyer palette.

- [ ] **Step 4: Run the whole suite**

Run: `.venv/bin/pytest tests -q`. Expected: all pass.

- [ ] **Step 5: Commit**

`git commit -m "feat(vegetation): leaf atlases painted at real size in photo colours"`

### Task 5: Dense junipers

**Files:**
- Modify: `src/redrocks/vegetation.py` (`_juniper`)
- Test: `tests/test_vegetation.py`

**Interfaces:**
- Consumes: `CARD_M`, `paint_atlas`, `side_fill` and `card_widths`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("species", ["JuniperRM", "JuniperOS"])
def test_junipers_are_dense_from_the_side(species):
    alpha = fa.paint_atlas("Juniper", 256)[..., 3] / 255.0
    for i in range(veg.SPECIES[species][1]):
        model = veg.build_model(species, i)
        assert vm.side_fill(model, alpha) >= 0.85, model.name

@pytest.mark.parametrize("species", list(veg.SPECIES))
def test_leaf_cards_use_their_atlas_card_width(species):
    for i in range(veg.SPECIES[species][1]):
        model = veg.build_model(species, i)
        w = vm.card_widths(model)
        assert np.all(np.abs(w / fa.CARD_M[model.atlas] - 1) <= 0.2), model.name
```

(`test_leaf_cards_use_their_atlas_card_width` stays red for pines and shrubs until Tasks 6–7. Mark
those cases `xfail(strict=True)` here, and remove the marks in those tasks.)

- [ ] **Step 2: Run the tests and check they fail**

Expected: the juniper cases FAIL.

- [ ] **Step 3: Implement**

Raise the crown-core share. Cores fill the envelope at `rho` 0–0.85; the holes rule drops to about
10 % of volume. Cards are only on clumps whose `rho` is ≥ 0.6, sized `CARD_M["Juniper"] ± 20 %`, with
more of them per clump. Heights stay 5.0 m (RM) and 3.6 m (OS), so the existing height test still
holds. Set `SCALE["JuniperOS"] = (0.56, 1.1)` so one-seed instances are 2–4 m (spec); RM's
(0.62, 1.2) already gives 3.1–6 m.

- [ ] **Step 4: Run the whole suite.** Expected: all pass, except the strict xfails.

- [ ] **Step 5: Commit**

`git commit -m "feat(vegetation): junipers read as dense grey-green masses"`

### Task 6: Ponderosa form

**Files:**
- Modify: `src/redrocks/vegetation.py` (`_ponderosa`, `SPECIES["Ponderosa"]` count 5, `SCALE["Ponderosa"] = (0.7, 1.3)`)
- Test: `tests/test_vegetation.py`

**Interfaces:**
- Produces: `Ponderosa_0..4`. Index 0 is young and conical, 1–3 are mature, 4 is old and flat-topped.
  Every one fills `Model.tips`.

- [ ] **Step 1: Write the failing tests**

```python
PINES = range(5)

def test_ponderosa_has_five_variants():
    assert veg.SPECIES["Ponderosa"][1] == 5

@pytest.mark.parametrize("i", PINES)
def test_ponderosa_crown_base_and_trunk(i):
    p = veg.build_model("Ponderosa", i)
    assert 0.40 <= vm.crown_base_fraction(p) <= 0.60
    lo, hi = veg.SCALE["Ponderosa"]
    d = vm.trunk_diameter_at(p, 1.4)
    assert d * lo >= 0.35 and d * hi <= 0.80
    assert 12.0 <= lo * p.height_m and hi * p.height_m <= 25.0

@pytest.mark.parametrize("i", PINES)
def test_ponderosa_is_open_with_needles_at_the_tips(i):
    p = veg.build_model("Ponderosa", i)
    alpha = fa.paint_atlas("Pine", 256)[..., 3] / 255.0
    assert 0.45 <= vm.side_fill(p, alpha) <= 0.75
    assert vm.tip_fraction(p, 1.0) >= 0.70

def test_old_ponderosa_is_flat_topped_and_young_is_conical():
    young, old = veg.build_model("Ponderosa", 0), veg.build_model("Ponderosa", 4)
    def width_at(p, f):   # 90th-percentile XY radius of leaf vertices within ±5 % of height fraction f of the crown
        ...
    assert width_at(young, 0.9) < 0.35 * width_at(young, 0.3)
    assert width_at(old, 0.9) > 0.6 * width_at(old, 0.5)
```

Update `test_ponderosa_trunk_is_clear_low_and_scaled_instances_reach_10_to_25_m` to the new 40 %
clear bound and 12–25 m. Drop the Ponderosa xfail in `test_leaf_cards_use_their_atlas_card_width`.

- [ ] **Step 2: Run the tests and check they fail.**

- [ ] **Step 3: Implement**

Branches: radius `0.10 * (length/3.6)^0.8 + 0.03` (thicker). Each path dips (`droop`), then upturns
(`curve_up`) over the last third, with a crook from `_wobbly_path`'s wobble at 0.25. Needle tufts:
round clumps of 6–10 cards, `CARD_M["Pine"]` wide, centred within 0.6 m of each branch tip and each
sub-tip (one or two forks per branch), and none along the inner half of the branch. The crown base is
`height * uniform(0.42, 0.58)`. Base radius is `0.36 ± 10 %`, so the 1.4 m diameter × scale lands in
range. Young: branch length tapers linearly to the top. Old: lengths stay ≥ 70 % to the top, with 2–3
flat upper limbs.

- [ ] **Step 4: Run the whole suite.** Expected: all pass, except the shrub xfails.

- [ ] **Step 5: Commit**

`git commit -m "feat(vegetation): ponderosa with tufts at the tips, open crowns, five forms"`

### Task 7: Gambel oak, oak patches and mountain mahogany

**Files:**
- Modify: `src/redrocks/vegetation.py` (`_shrub`, new `_oak_patch`, `SPECIES["OakPatch"] = 3`, `SCALE`)
- Test: `tests/test_vegetation.py`

**Interfaces:**
- Produces: `OakPatch_0..2` (species `gambel_oak`, atlas `Oak`, height about 2.6 m). Scales:
  `SCALE["Oak"] = (0.6, 1.5)` and `SCALE["OakPatch"] = (0.8, 1.3)` with the Oak model at 2.6 m, and
  `SCALE["Mahogany"] = (0.55, 1.4)` with a 2.1 m model.

- [ ] **Step 1: Write the failing tests**

```python
def test_oak_patches_are_3_to_6_m_across_with_a_continuous_canopy():
    alpha = fa.paint_atlas("Oak", 256)[..., 3] / 255.0
    for i in range(3):
        p = veg.build_model("OakPatch", i)
        lo, hi = veg.SCALE["OakPatch"]
        across = 2 * vm.footprint_radius_m(p)
        assert across * lo >= 3.0 and across * hi <= 6.0
        assert vm.side_fill(p, alpha) >= 0.7
        assert len(p.parts["Trunk"].f) > 0 and veg.closed_and_outward(p.parts["Trunk"])
```

Update `test_model_heights_match_the_species` for Oak (2.0–3.0), Mahogany (1.6–2.6) and OakPatch
(2.0–3.2), and check the scaled instance range: oak bushes 1.5–4 m, mahogany 1–3 m. Drop the
remaining xfails. `test_every_species_has_variants` covers OakPatch automatically.

- [ ] **Step 2: Run the tests and check they fail.**

- [ ] **Step 3: Implement**

`_oak_patch` grows 12–25 stems over an ellipse 1.6–2.6 m in radius (so 3.2–5.2 m across before
scale), with clumps filling a flattened dome across the whole ellipse. Mahogany gets more, thinner
twigs (5–8 stems, 3 forks each) and small cards at `CARD_M["Mahogany"]`.

- [ ] **Step 4: Run the whole suite.** Expected: all pass.

- [ ] **Step 5: Commit**

`git commit -m "feat(vegetation): taller scrub oak, oak patches, twiggy mahogany"`

### Task 8: Thickets, juniper groups and ponderosa groves

**Files:**
- Modify: `src/redrocks/vegetation.py` (`place` and new module-level helpers)
- Test: `tests/test_vegetation.py`

**Interfaces:**
- Produces:
  - `thicket_field(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray` (0..1)
  - `THICKET_LEVEL: float`: inside a thicket where the field is ≥ this. The implementer sets it so
    thickets cover roughly 35–50 % of eligible lower slopes on the real terrain.
  - `THICKET_SPACING_CM = 300.0`
  - `place(...)` keeps its signature and return shape.

- [ ] **Step 1: Write the failing tests**

```python
def _low_gentle_ground():
    return veg.Ground(_slope_grid(n=81, size=320000.0, fn=lambda x, y: 0.08 * x + 20000.0))

def test_thickets_are_continuous_and_openings_are_open(models):
    by_name = {mm.name: mm for mm in models}
    g = _low_gentle_ground()
    out = veg.place(g, set(), by_name)
    pts = grid of points every 200 cm over the ground, outside SCAN_CLEAR_CM
    inside = veg.thicket_field(pts[:, 0], pts[:, 1], veg.SEED) >= veg.THICKET_LEVEL
    shrubs = ("Oak", "OakPatch", "Mahogany")
    assert vm.canopy_cover(out, by_name, pts[inside], shrubs) >= 0.60
    assert vm.canopy_cover(out, by_name, pts[~inside], shrubs) <= 0.15

def test_shrub_canopy_stays_off_masked_cells(models):
    # a road across the middle; no shrub footprint circle reaches into a blocked cell
    ...
    for name, rows in out.items():
        r = vm.footprint_radius_m(by_name[name]) * 100
        for x, y, *_, sxy, _ in rows:
            for dx, dy in ((r*sxy, 0), (-r*sxy, 0), (0, r*sxy), (0, -r*sxy)):
                assert cell_of(x + dx, y + dy) not in blocked   # cell_of = floor(xy / BLOCK_CELL_CM)

def test_ponderosa_grows_in_groves(models):
    out = veg.place(veg.Ground(_slope_grid(n=81, size=320000.0, fn=lambda x, y: 90000.0 - 0.25 * y)), set(), by_name)
    pines = np.array([r[:2] for n, rs in out.items() if n.startswith("Ponderosa") for r in rs])
    sizes = single_linkage_group_sizes(pines, 1500.0)   # test helper: union-find over pairs closer than 15 m
    assert 3 <= np.median(sizes) <= 8

def test_instance_count_is_bounded(models):
    out = veg.place(_low_gentle_ground(), set(), {mm.name: mm for mm in models})
    assert sum(map(len, out.values())) <= 0.25 * (3200 ** 2) / 9   # at most one plant per 36 m²
```

The existing placement tests (masks, determinism, cliffs, north-facing pines, pines higher than
shrubs) must still pass.

- [ ] **Step 2: Run the tests and check they fail.**

- [ ] **Step 3: Implement**

- **Thicket field:** keep today's two-octave noise as `thicket_field`, plus a draw term (positive
  height Laplacian at 40 m) and the existing low-elevation shrub weight.
- **Inside thickets:** a jittered `THICKET_SPACING_CM` grid places shrubs, about 60 % `OakPatch`,
  25 % `Oak` and 15 % `Mahogany`. Mahogany rises with `rocky`.
- **Outside thickets:** keep the 6.5 m grid, but shrubs there need `keep_roll < 0.12`.
- **Masks for shrubs:** dilate the blocked set for shrubs by `ceil(max footprint × max scale ÷ 800 cm)`
  cells.
- **Junipers:** groups of 1–3 within 4 m; their weight gains `(1 + 0.6 * clip(-north, 0, 1))`.
- **Pines:** each chosen pine seeds 2–7 more within 12 m, each kept only if its own ground passes the
  pine slope mask.

- [ ] **Step 4: Run the whole suite.** Expected: all pass.

- [ ] **Step 5: Build on the real terrain and check counts**

Run: `.venv/bin/redrocks vegetation --out out`. It prints counts; read `out/vegetation/vegetation.json`
`counts`. Expected: a total of 250,000 or fewer, with OakPatch present. If it's over, raise
`THICKET_SPACING_CM` in 50 cm steps and rerun.

- [ ] **Step 6: Commit**

`git commit -m "feat(vegetation): continuous scrub thickets, juniper groups, ponderosa groves"`

### Task 9: Unreal side: colours, bark distance blend and cull distances

**Files:**
- Modify: `unreal/setup_vegetation.py`
- Test: `tests/test_vegetation.py`

**Interfaces:**
- Produces:
  - `SHRUBS = ("Oak", "OakPatch", "Mahogany")`
  - `cull_distances(model_name: str, part: str) -> tuple[int, int]`, as `(start_cm, end_cm)` with
    `(0, 0)` meaning never. Shrub `Crown` is `(0, 0)`. Shrub `Leaves` and `Trunk` are
    `(32000, 40000)`. Trees are `(0, 0)` for every part.
  - Each `BARK[group]` becomes `(near_light, near_dark, far, plate_cm, stretch, contrast)`. Ponderosa
    `far = "#5A5236"`.
  - New material params on `M_Veg_Bark`: `BarkFar`, `FarStartCm` (600) and `FarEndCm` (1500). They
    lerp near to far by `MaterialExpressionPixelDepth`.
  - `FOLIAGE` tints sit within ±10 % of white, since the atlas now carries the colour. Crown cores
    are the photo colour × 0.8 (A) and × 0.95 (B).

- [ ] **Step 1: Write the failing tests**

```python
def test_every_model_maps_to_materials_and_cull(sv, models):
    for mm in models:
        assert sv.bark_group(mm.name) in sv.BARK and mm.atlas in sv.FOLIAGE
        for part in veg.PARTS:
            s, e = sv.cull_distances(mm.name, part)
            assert (s, e) == (0, 0) or 0 < s < e

def test_shrub_parts_cull_together_and_crowns_never(sv):
    for name in ("Oak_0", "OakPatch_1", "Mahogany_2"):
        assert sv.cull_distances(name, "Crown") == (0, 0)
        assert sv.cull_distances(name, "Leaves") == sv.cull_distances(name, "Trunk") == (32000, 40000)
    assert sv.cull_distances("Ponderosa_4", "Leaves") == (0, 0)

def test_ponderosa_bark_is_dark_grey_brown_at_distance(sv):
    assert sv.BARK["Ponderosa"][2] == "#5A5236"

def test_snow_runs_after_vegetation_and_covers_veg_materials():
    steps = [s for s, _ in rebuild_level.STEPS]
    assert steps.index("setup_snow.py") > steps.index("setup_vegetation.py")
    assert 'GAME_DIR = "/Game/RedRocks"' in (UNREAL / "setup_snow.py").read_text()   # walks every material under it
    assert '"/Game/RedRocks/Vegetation"' in (UNREAL / "setup_vegetation.py").read_text()
```

(Load `rebuild_level.py` with the same fake-`unreal` loader as `sv`.)

- [ ] **Step 2: Run the tests and check they fail.**

- [ ] **Step 3: Implement**

In `_place`, set `instance_start_cull_distance` and `instance_end_cull_distance` from
`cull_distances` for every component, which replaces `SHRUB_CULL_CM`. Shrub leaves keep
`cast_shadow=False`. Extend `build_bark` with the far lerp. `_materials` passes the far colour and the
new scalars. Update the docstring's steps 2 and 4.

- [ ] **Step 4: Run the whole suite.** Expected: all pass.

- [ ] **Step 5: Commit**

`git commit -m "feat(vegetation): photo colours, bark that greys with distance, crowns never cull"`

### Task 10: Build, shoot, tune and get Austin's look approval

**Files:**
- Create: `docs/media/vegetation-real/{trail,ponderosa,junipers,hillsides}.jpg`, `delta_e.json` and `frame_rate.json`
- Modify: `README.md` (the Vegetation section only), `LICENSE` (third-party list),
  `docs/media/vegetation/` (refresh `after_*.jpg`)

**Interfaces:**
- Consumes: everything above, plus Task 2's `baseline.json` and `out/shots/before_*.png`.

- [ ] **Step 1: Import into the project copy**

Take the lock. Run `REDROCKS_OUT_DIR=$WT/out "$UE" "$PROJ" -run=pythonscript -script=$WT/unreal/setup_vegetation.py -unattended -nosplash -stdout`,
then the same with `setup_snow.py`. Expected: `/tmp/redrocks_vegetation.log` and
`/tmp/redrocks_snow.log` have no `FAIL`, `ERROR` or `ABORT`, and the vegetation log lists OakPatch
and five Ponderosa models.

- [ ] **Step 2: Shoot the after views** the same way as Task 2 Step 4, with `REDROCKS_SHOT_PREFIX=after_`
and 3 runs. Copy the shots to `out/shots/after_<view>.png`. Release the lock.

- [ ] **Step 3: Compare**

Run: `.venv/bin/python tools/make_reference_compare.py --shots out/shots --out docs/media/vegetation-real --cache out/refs`.
Expected: every sample has ΔE ≤ 15, and every view's median fps is ≥ 0.85 × baseline. Write the fps
table to `frame_rate.json`.

- [ ] **Step 4: Tune until both hold**

- **Colour off:** adjust `PHOTO_COLOUR`-centred palettes or the Unreal tints.
- **Too slow:** cut in the spec's order: cards per clump, then shrub leaf cull distance, then
  thicket density.
- Repeat Steps 1–3. Each tuning round is its own commit.

- [ ] **Step 5: Check play still works**

Run `playtest_explore.py` (GUI, with the lock). Expected: the climb, wingsuit, cart, sled and tubing
runs finish as in the README. The sled run and roads are clear of shrubs.

- [ ] **Step 6: Update the README and LICENSE**

In the README's Vegetation section:
- the new species table, with sizes, OakPatch and five ponderosa forms;
- placement (thickets, groups, groves) and the cull rules;
- the four composites, each credited with its photo's title link, author and licence;
- a line saying the composites are CC BY-SA because they contain CC BY-SA photos.

Refresh `docs/media/vegetation/after_*.jpg` from the after shots. In `LICENSE`, add "Reference
photos in docs/media/vegetation-real/: Wikimedia Commons, each credited on the image and in the
README (CC BY 2.0, CC BY-SA 3.0, CC BY-SA 4.0)."

- [ ] **Step 7: Commit, then show Austin**

`git commit -m "docs(vegetation): side by side with Red Rocks Park photos; ΔE and frame rate"`.
Send Austin the four composites, the ΔE table and the fps table. Wait for his approval of the look.
If he asks for changes, go back to Step 4.

### Task 11: PR 1 and merge

- [ ] **Step 1: Full test run.** `.venv/bin/pytest -q`. Expected: all pass. Note the count.
- [ ] **Step 2: Rebase if the seating PR has merged.** `git fetch origin && git rebase origin/main`,
  then rerun the tests. If `tools/make_progress_gif.py` is now on main, append
  `{"frame": "docs/media/vegetation-real/after_bowl_top.jpg", "title": "Realistic trees and shrubs, matched to Red Rocks Park photos", "pr": "#<PR 1>"}`
  to `docs/media/progress/stages.json`, copy the bowl-top after shot to that path, run
  `.venv/bin/python tools/make_progress_gif.py` and commit. If it isn't on main yet, the frame moves to
  PR 2.
- [ ] **Step 3: Push and open the PR.** `git push -u origin feat/realistic-vegetation` and
  `gh pr create --repo austin80808/red-dead-rocks`. The body has the composites table, the ΔE and fps
  tables, a test summary and the known issues. Then `get_status` and `bind_pr` with the ccd_pr tools.
- [ ] **Step 4: Merge.** `merge-gate austin80808/red-dead-rocks#<N> --tested ".venv/bin/pytest -q: <n> passed; rebuild steps OK in RedRocks-trees; playtest_explore OK"`.
  Expected: exit 0 (merged) or 3 (held for Austin: tell him in one line with the link).
- [ ] **Step 5: Hand-over note.** Leave a short note in the shared notes folder: what merged, and
  that Austin's own project needs `rebuild_level.py` (or `setup_vegetation.py` plus `setup_snow.py`)
  to pick it up.

### Task 12: PR 2: grass (start only after the seating PR has merged)

**Files:**
- Modify: `unreal/setup_materials.py` (ground grass colours and amount), `src/redrocks/vegetation.py`
  (grass clumps), `src/redrocks/foliage_atlas.py` (`Grass` atlas), `unreal/setup_vegetation.py` (grass
  components)
- Test: `tests/test_vegetation.py`, `tests/test_look_scripts.py`

**Interfaces:**
- Produces:
  - `Grass_0..3` models (Leaves only, no Trunk or Crown: allow an empty part list for this species)
  - `CARD_M["Grass"] = 0.5`, `SIZE_M["Grass"] = (0.2, 0.6)` (blade length)
  - `out/vegetation/grass.f32`: little-endian float32 rows `x, y, z, yaw, scale_xy, scale_z`. Kept out
    of `vegetation.json` so it stays small; Unreal reads it with the stdlib `array` module.
  - `place_grass(ground, blocked, seed) -> np.ndarray` (N, 6): one clump per 9 m², gentle ground only
    (`nz ≥ 0.9`), outside the masks and outside thickets.
  - `cull_distances("Grass_*", "Leaves") == (12000, 15000)`.

- [ ] **Step 1: Rebase onto main and confirm `setup_materials.py` matches main.**
- [ ] **Step 2: Failing tests.**
  - `GrassColor` and `DryGrassColor` in `setup_materials.py` are within ΔE 15 of `#96914D` and
    `#B9A775`, and `GrassAmount` is ≥ 0.8.
  - Grass atlas blades are 0.2–0.6 m.
  - `place_grass` density is 1/9 m² ± 15 % on flat open ground, and 0 on cliffs, masks and thickets.
  - `grass.f32` round-trips.
- [ ] **Step 3: Implement** to the interfaces above. `setup_vegetation.py` makes one HISM per grass
  model, with no collision and no shadow.
- [ ] **Step 4: Measure.** Re-shoot all views with grass. Expected: fps ≥ 0.85 × Task 2's baseline,
  and the grass sample in the trail view within ΔE 15. If it's too slow, lower the density in steps of
  1 per 12 m², then 1 per 16 m².
- [ ] **Step 5: Austin's look approval, the progress GIF frame if it wasn't done in PR 1, then the PR
  and `merge-gate`**, as in Tasks 10–11. The material change is normal code, so expect 🟡.
