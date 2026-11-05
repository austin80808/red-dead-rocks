# Photo Board (P0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the photo board (six real Commons photos, each with a fitted game camera, shot in Unreal and scored for colour and skyline against the photo) and commit today's level as the baseline. No visual change to the level.

**Architecture:** Pure-Python modules in `src/redrocks/` do the maths offline and are unit-tested: a pinhole camera model and least-squares fit (`photo_camera.py`), skyline and region scores (`board_metrics.py`), named world landmarks from the build (`landmarks.py`), the board file (`photo_board.py`). Two tools drive them (`tools/fit_photo_camera.py`, `tools/make_board_compare.py`), and one Unreal script shoots the views in the editor world (`unreal/shoot_board.py`). The vegetation work's compare code is moved into `src/redrocks/board_images.py` and shared, not copied.

**Tech Stack:** Python 3.10+, numpy, Pillow, requests (all already in `pyproject.toml`), Unreal Engine 5.8 editor Python.

**Spec:** `docs/superpowers/specs/2026-11-05-look-like-red-rocks-design.md` (sections "Success", "The photo board", "P0 The board", "Ground rules", "Tests").

P1-P4 each get their own plan after this one lands: their order inside each pass comes from P0's baseline scores ("the three worst regions on the board").

## Global Constraints

- Colour target: each sampled region within CIE76 ΔE 10 of the photo, none over 20.
- Shape target: skyline within 2 % of image height of the photo's, on average along its length.
- A camera fit with a mean landmark error over 1.5 % of the image width is refused.
- Photos download into `out/refs/` and are never committed. Composites are committed with the photo's title, author and licence in the caption; LICENSE says board composites are CC BY-SA.
- No new dependency: numpy, Pillow and requests only (scipy is installed but not a project dependency; don't use it).
- Nothing paid or from Fab in git; both repos are public.
- Unreal work happens in the project copy `~/Documents/Unreal Projects/RedRocks-look`, never in Austin's `~/Documents/Unreal Projects/RedRocks`, while holding `~/.local/bin/unreal-lock acquire "Red Rocks look" "<why>"` (release it after).
- Level frame: x = east·100, y = −north·100, z = (elevation − 1742.098 m)·100, in cm (`unreal/level_frame.py`, after #55).
- `pytest`, the play-tests (`unreal/playtest_explore.py`) and `tools/check_seating_clear.py` stay green. Merge only with `merge-gate <PR> --tested "<command, result>"`.

## Review Focus

1. **Commons unreachable or a photo URL gone:** `make_board_compare.py` stops before writing anything and names the view ("could not fetch photo for postcard: …"), exit 1. Test in Task 7.
2. **A shot that isn't the photo's aspect** (someone shoots 16:9): scores resize the game shot to the photo's aspect by cover-crop before measuring, so a 1600x900 shot of a 4:3 photo scores the same as a 1600x1200 one of the same scene. Test in Task 2.
3. **No sky in some columns** (a cliff touching the top edge, or a white overcast sky): those columns are left out of the skyline error, never counted as row 0; a view with fewer than 20 % of columns shared reports `skyline_error: null`, not 0. Test in Task 2.
4. **A black or half-rendered frame** (Lumen or streaming not settled): the compare tool refuses a game shot whose mean luminance is under 10/255 and names it. Test in Task 7.
5. **A landmark name the catalog doesn't know, or a landmark behind the camera:** board validation names the view and the landmark; the fit refuses points behind the camera instead of producing a mirrored solution. Tests in Tasks 1 and 4.

---

### Task 1: Camera model and fit

**Files:**
- Create: `src/redrocks/photo_camera.py`
- Test: `tests/test_photo_camera.py`

**Interfaces:**
- Produces:
  - `@dataclass Camera: x: float, y: float, z: float, yaw: float, pitch: float, roll: float = 0.0, fov: float = 70.0` (level cm; Unreal degrees; `fov` horizontal). `Camera.to_dict() -> dict`, `Camera.from_dict(d: dict) -> Camera` (keys `x y z yaw pitch roll fov`; `roll` optional).
  - `project(cam: Camera, points_cm: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]`: pixel `(N, 2)` (x right, y down, origin top-left) and a bool `(N,)` "in front of the camera".
  - `fit_camera(points_cm: np.ndarray, pixels: np.ndarray, size: tuple[int, int], guess: Camera, iterations: int = 60) -> tuple[Camera, float]`: fitted camera and mean reprojection error as a fraction of image width.
  - `class FitRefused(ValueError)`.

- [ ] **Step 0: Branch**

```bash
cd ~/dev/redrocks-ue
git fetch origin
git switch -c look/p0-photo-board origin/feat/realistic-vegetation   # rebase onto main once that PR merges
git cherry-pick bbabfda                                              # the spec
.venv/bin/python -m pytest -q 2>&1 | tail -1                         # expect all passed
```

- [ ] **Step 1: Write the failing tests**

```python
def test_a_point_straight_ahead_lands_in_the_centre():
    cam = Camera(x=0, y=0, z=0, yaw=0, pitch=0, fov=90)
    px, front = project(cam, np.array([[1000.0, 0, 0]]), (1600, 1200))
    assert front.all() and px[0] == pytest.approx([800, 600], abs=1e-6)

def test_level_plus_y_is_screen_right_and_plus_z_is_up():
    cam = Camera(x=0, y=0, z=0, yaw=0, pitch=0, fov=90)
    px, _ = project(cam, np.array([[1000.0, 500, 0], [1000.0, 0, 500]]), (1600, 1200))
    assert px[0][0] > 800 and px[1][1] < 600      # Unreal is left-handed: +Y right of +X

def test_fov_is_horizontal():
    cam = Camera(x=0, y=0, z=0, yaw=0, pitch=0, fov=90)
    px, _ = project(cam, np.array([[1000.0, 1000, 0]]), (1600, 1200))   # 45° right = image edge
    assert px[0][0] == pytest.approx(1600, abs=1e-6)

def test_fit_recovers_a_synthetic_camera():
    true = Camera(x=12000, y=-3000, z=2500, yaw=-150, pitch=-8, roll=1.5, fov=63)
    pts = np.array(<8 points spread 50-400 m in front of `true`, z 0-9000 cm>)
    px, _ = project(true, pts, (1280, 960))
    got, err = fit_camera(pts, px, (1280, 960), Camera(x=11000, y=-2000, z=2000, yaw=-140, pitch=0, fov=70))
    assert abs(got.x - true.x) < 50 and abs(got.y - true.y) < 50 and abs(got.z - true.z) < 50   # 0.5 m
    for a in ("yaw", "pitch", "roll", "fov"):
        assert abs(getattr(got, a) - getattr(true, a)) < 0.5
    assert err < 0.001

def test_fit_refuses_noisy_landmarks():
    # same setup, pixels jittered by ±3 % of the width
    with pytest.raises(FitRefused, match="1.5 %"):
        fit_camera(pts, noisy, (1280, 960), guess)

def test_fit_refuses_fewer_than_four_points_and_points_behind_the_camera():
    with pytest.raises(FitRefused, match="at least 4"):
        fit_camera(pts[:3], px[:3], (1280, 960), guess)
    # a landmark 100 m behind `true` at its pixel: refused, not fitted mirrored
    with pytest.raises(FitRefused, match="behind"):
        fit_camera(pts_with_one_behind, px, (1280, 960), guess)
```

- [ ] **Step 2: Run them; expect `ModuleNotFoundError: redrocks.photo_camera`**

Run: `.venv/bin/python -m pytest tests/test_photo_camera.py -v`

- [ ] **Step 3: Implement `Camera`, `project`, `fit_camera`, `FitRefused`**

Camera axes from the Unreal rotator: forward `f = (cos p cos y, cos p sin y, sin p)`, right `r = (−sin y, cos y, 0)`, up `u = r × f` reversed so +Z is up for pitch 0 (check with the second test), then roll rotates `r, u` about `f`. Focal length `F = (W/2) / tan(fov/2)`; pixel `= (W/2 + F·(d·r)/(d·f), H/2 − F·(d·u)/(d·f))` for `d = p − c`; in front means `d·f > 0`. `fit_camera`: Levenberg-Marquardt over the 7 parameters with a numeric Jacobian (central differences: 1 cm, 0.01°), starting at `guess`; any point behind the fitted camera raises `FitRefused("... behind the camera")`; mean error over 1.5 % of `W` raises `FitRefused("... over 1.5 % ...")`.

- [ ] **Step 4: Run the tests; expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/redrocks/photo_camera.py tests/test_photo_camera.py
git commit -m "feat(board): pinhole camera in the level frame and a least-squares fit from landmarks"
```

---

### Task 2: Skyline and colour scores

**Files:**
- Create: `src/redrocks/board_metrics.py`
- Test: `tests/test_board_metrics.py`

**Interfaces:**
- Consumes: `redrocks.vegetation_metrics.delta_e(a, b) -> float`, `median_hex(img, xy, r=10) -> str` (from `feat/realistic-vegetation`).
- Produces:
  - `fit_to(img: Image.Image, size: tuple[int, int]) -> Image.Image`: cover-crop and resize to `size`.
  - `sky_mask(rgb: np.ndarray) -> np.ndarray`: bool HxW, sky-like pixels (blue-dominant `b > r + 10 and b > g`, or bright and grey `min ≥ 170 and max − min ≤ 30`) connected to the top row.
  - `skyline(mask: np.ndarray) -> np.ndarray`: int per column, the lowest sky row, −1 where the column has no sky.
  - `skyline_error(game: Image.Image, photo: Image.Image, ignore: list[list[int]]) -> float | None`: mean |Δrow| / H over columns where both have sky and outside `ignore` boxes (photo pixels); `None` when fewer than 20 % of columns qualify. Measured at the photo's size after `fit_to(game, photo.size)`.
  - `region_hex(rgb: np.ndarray, box: list[int], kind: str) -> str`: median colour of the box; `kind` in `sky rock_lit rock_shade tread riser plants plains`; `plants` keeps green pixels (the vegetation tool's "green" rule), `sky` keeps `sky_mask` pixels, the rest take the whole box.
  - `score_view(view: dict, game: Image.Image, photo: Image.Image) -> dict`: `{"regions": {label: {"kind", "photo", "game", "delta_e"}}, "skyline_error": float | None}`.
  - `worst_regions(scores: dict[str, dict], n: int = 3) -> list[tuple[str, str, float]]`: `(view, label, ΔE)` highest first.

- [ ] **Step 1: Write the failing tests**

```python
def test_skyline_finds_a_drawn_ridge():          # blue above row 200 + 50·sin(x/100), sandstone below
    assert np.abs(skyline(sky_mask(img)) - truth).max() <= 1

def test_columns_without_sky_are_left_out_not_zero():
    # rock fills columns 0-399 to the top edge; ridge elsewhere identical in both images
    assert skyline(sky_mask(a))[:400].tolist() == [-1] * 400
    assert skyline_error(game, photo, []) == pytest.approx(0.0, abs=1e-9)

def test_too_little_shared_sky_is_none():          # sky in 10 % of columns
    assert skyline_error(game, photo, []) is None

def test_skyline_error_is_a_fraction_of_height_and_ignores_boxes():
    # game ridge 24 px lower than photo's in a 1200 px tall image -> 0.02
    assert skyline_error(game, photo, []) == pytest.approx(0.02, abs=1e-3)
    assert skyline_error(game_with_flag, photo, [[600, 0, 700, 400]]) == pytest.approx(0.02, abs=1e-3)

def test_a_wrong_aspect_shot_scores_like_the_right_one():
    # the same scene rendered 1600x900 and 1600x1200 against a 4:3 photo
    assert skyline_error(wide, photo, []) == pytest.approx(skyline_error(tall, photo, []), abs=0.005)

def test_region_kinds_and_worst_regions():
    assert region_hex(rgb, [0, 0, 10, 10], "sky") == "#5A8CDC"   # blue patch with a white cloud corner
    assert worst_regions({"a": {"regions": {"x": {"delta_e": 3.0}, "y": {"delta_e": 22.0}}},
                          "b": {"regions": {"z": {"delta_e": 11.0}}}}) == [("a", "y", 22.0), ("b", "z", 11.0), ("a", "x", 3.0)]
```

- [ ] **Step 2: Run them; expect `ModuleNotFoundError: redrocks.board_metrics`**

Run: `.venv/bin/python -m pytest tests/test_board_metrics.py -v`

- [ ] **Step 3: Implement the functions above.** `sky_mask` uses a flood fill from the top row over sky-like pixels (iterative with numpy, row by row: a pixel is sky if it is sky-like and its upper neighbour or a left/right sky neighbour on the same row is sky).

- [ ] **Step 4: Run the tests; expect PASS**

- [ ] **Step 5: Commit** `git commit -m "feat(board): skyline and region colour scores against a photo"`

---

### Task 3: Landmarks from the build

**Files:**
- Create: `src/redrocks/landmarks.py`
- Test: `tests/test_landmarks.py`

**Interfaces:**
- Consumes: `out/manifest.json` (`monoliths`, `stage.aisles`, `benches.tread_levels_m`, `booth`, `stage.walkway_*`), `assets/scan/scan_top.npz` via `redrocks.scan_blend.ScanHeights.load`, the monolith OBJs in `out/`.
- Produces:
  - `catalog(out_dir: Path, scan_top: Path) -> dict[str, tuple[float, float, float]]`: name → terrain-frame (east m, north m, elevation m).
  - `to_level_cm(points: dict[str, tuple]) -> dict[str, tuple[float, float, float]]`: the Global Constraints rule.
  - Names: `creation_rock_summit`, `ship_rock_summit`, `stage_rock_summit` (highest `scan_top` cell inside the monolith OBJ's footprint hull, falling back to the manifest's `top_elevation_m` at its centroid where the scan has no cells), `aisle_start_row{N}` / `aisle_end_row{N}` for N in 8, 20, 35, 50, 70 (the `stage.aisles` step whose z is nearest `tread_levels_m[N-1]`), `booth_roof` (top centre of `Booth.obj`), `walkway_centre` (`stage.walkway_east_m/north_m` at `walkway_elevation_m`).

- [ ] **Step 1: Write the failing tests** against the real build (skip with a clear reason when `out/manifest.json` is missing): every name above is present; `creation_rock_summit` elevation is within 15 m of the manifest's 2023.847; `aisle_end_row70` elevation is within 0.6 m of `tread_levels_m[69]`; `to_level_cm` matches `unreal/level_frame.to_level_cm` on three points (load that module with `importlib`, as `tests/test_level_frame.py` does).

- [ ] **Step 2: Run them; expect `ModuleNotFoundError: redrocks.landmarks`**

- [ ] **Step 3: Implement `catalog` and `to_level_cm`.**

- [ ] **Step 4: Run the tests; expect PASS**

- [ ] **Step 5: Commit** `git commit -m "feat(board): named landmarks from the build for camera fits"`

---

### Task 4: The six board views

**Files:**
- Create: `docs/media/photo-board/views.json`, `src/redrocks/photo_board.py`
- Modify: `LICENSE` (board composites are CC BY-SA, after the vegetation note)
- Test: `tests/test_photo_board.py`

**Interfaces:**
- Consumes: `landmarks.catalog` names (Task 3), `Camera.from_dict` (Task 1).
- Produces:
  - `views.json`: `{"about": str, "views": [view]}`; a view is `{"name", "photo": {"id", "title", "page", "file_url", "author", "licence"}, "aspect": [w, h], "camera_guess": {...}, "camera": {...} | null, "fit_error": float | null, "sun": {"azimuth_deg", "elevation_deg"}, "landmarks": [{"name", "photo_xy": [x, y]}], "samples": [{"label", "kind", "photo_box", "game_box"}], "skyline_ignore": [[x0, y0, x1, y1]]}`. Pixel coordinates are in the 1280 px Commons thumb.
  - `load_board(path: Path) -> list[dict]` and `validate_board(views: list[dict], landmark_names: set[str]) -> list[str]` (problems, empty when valid).

- [ ] **Step 1: Write the failing tests**

```python
def test_the_committed_board_is_valid():
    views = load_board(BOARD)
    assert [v["name"] for v in views] == ["postcard", "up_the_rows", "across_the_rows", "stage_front", "monolith_plaza", "aerial"]
    assert validate_board(views, set(catalog_names)) == []

def test_validation_names_the_view_and_the_problem():
    bad = deepcopy(views); bad[0]["landmarks"][0]["name"] = "eiffel_tower"; del bad[1]["photo"]["licence"]
    problems = validate_board(bad, set(catalog_names))
    assert "postcard: unknown landmark eiffel_tower" in problems
    assert "up_the_rows: photo has no licence" in problems
```

Validation rules: photo has `title page file_url author licence`; at least 6 landmarks, all known; `sun` has both angles; at least 4 samples with known kinds; `aspect` matches the thumb's shape within 1 %.

- [ ] **Step 2: Run them; expect FAIL** (no `photo_board` module, no `views.json`).

- [ ] **Step 3: Write `photo_board.py`, then fill `views.json`** from the spec's table (pages, authors, licences as listed there). Download each 1280 px thumb into `out/refs/` (`board_images.fetch_photo`, Task 7, or by hand with the same URL for now), open it and record: 6-10 `landmarks` per view by their pixel position (the summits, row ends at rows 8/20/35/50/70, booth roof and walkway centre that are visible in that photo); a `camera_guess` (spot and heading read off the map: e.g. `postcard` from `walkway_centre`, 1.7 m up, facing the stage); the `sun` from the photo's shadows; 4-7 `samples` covering the kinds that view shows; `skyline_ignore` boxes over people, flags and lens flare. `camera` and `fit_error` stay null until Task 5.

- [ ] **Step 4: Run the tests; expect PASS**

- [ ] **Step 5: Commit** `git commit -m "feat(board): six Commons photo views with landmarks, sun and colour samples"`

---

### Task 5: Fit the cameras

**Files:**
- Create: `tools/fit_photo_camera.py`
- Test: `tests/test_fit_photo_camera.py`
- Modify: `docs/media/photo-board/views.json` (cameras written by the tool)

**Interfaces:**
- Consumes: `load_board`, `landmarks.catalog` + `to_level_cm`, `fit_camera`, `FitRefused`.
- Produces: CLI `tools/fit_photo_camera.py [--views PATH] [--out out] [--only NAME]`: writes `camera` and `fit_error` (rounded to 4 places) per view; a refused view keeps its old `camera` and prints `REFUSED <view>: <reason>`; exit 1 if any view was refused.

- [ ] **Step 1: Write the failing tests** on a temp board with one synthetic view (landmark pixels made by projecting a known camera) and one with a deliberately wrong landmark: the first gets a camera within 0.5 m / 0.5°; the second prints `REFUSED`, keeps `camera: null`; exit code 1.
- [ ] **Step 2: Run; expect FAIL** (no tool).
- [ ] **Step 3: Implement the tool.**
- [ ] **Step 4: Run the tests; expect PASS.** Then run it on the real board: `.venv/bin/python tools/fit_photo_camera.py`. Expected: six views fitted, every `fit_error` ≤ 0.015, exit 0. A refused view means its landmarks are wrong: fix them in `views.json` (Task 4's data) and re-run.
- [ ] **Step 5: Commit** `git commit -m "feat(board): fit each view's camera from its landmarks"`

---

### Task 6: Shoot the board in Unreal

**Files:**
- Create: `unreal/shoot_board.py`
- Test: `tests/test_shoot_board.py` (offline maths, loaded with a stub `unreal` module like `tests/test_explore_scripts.py`)

**Interfaces:**
- Consumes: `docs/media/photo-board/views.json` (`camera`, `aspect`, `sun`, `landmarks`), `landmarks` catalog from `out/` for calibration markers, `setup_lighting.sun_rotation(azimuth_deg, elevation_deg) -> (pitch, yaw)`.
- Produces: `out/board/<prefix><view>.png` (1600 px wide at the view's aspect) and, with `REDROCKS_BOARD_CALIB=1`, `out/board/calib_<view>.png` with a 60 cm unlit magenta (#FF00FF) sphere at each of the view's landmarks. Env: `REDROCKS_OUT_DIR`, `REDROCKS_BOARD_PREFIX` (default `before_`), `REDROCKS_BOARD_ONLY` (comma list). Log: `$REDROCKS_LOG_DIR/redrocks_board.log` with one `ok`/`FAIL` line per view.
  - Offline-tested helpers: `shot_size(aspect: list[int], width: int = 1600) -> tuple[int, int]`; `camera_settings(view: dict) -> dict` (`location`, `rotation` as pitch/yaw/roll, `fov`, `aspect_ratio`).

Runs in the editor world (not Play In Editor), so no HUD draws: it loads `/Game/RedRocks/Lvl_RedRocks`, runs `DisableAllScreenMessages`, spawns one `CameraActor` with `constrain_aspect_ratio` on, and per view sets the camera, sets the `Sun` actor's rotation from the view's sun, waits 8 s of editor ticks (Lumen and streaming), then `unreal.AutomationLibrary.take_high_res_screenshot(w, h, path, camera)`. It quits the editor when done.

- [ ] **Step 1: Write the failing tests:** `shot_size([4, 3]) == (1600, 1200)`; `camera_settings` returns the view's camera and `aspect_ratio == 4/3`; the module imports with the stub and does nothing at import.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests; expect PASS.**
- [ ] **Step 5: Commit** `git commit -m "feat(board): shoot the board views in the editor world"`

---

### Task 7: Composites, scores and the calibration check

**Files:**
- Create: `src/redrocks/board_images.py`, `tools/make_board_compare.py`
- Modify: `tools/make_reference_compare.py` (import `fetch_photo`, `compose`, `_cover` from `redrocks.board_images` instead of defining them; its tests must stay green)
- Test: `tests/test_board_compare.py`

**Interfaces:**
- Consumes: `board_metrics.score_view`, `worst_regions`, `photo_camera.project`, `Camera.from_dict`, `load_board`.
- Produces:
  - `board_images.fetch_photo(photo: dict, cache_dir: Path) -> Path`, `compose(before, after, photo, captions, aspect, panel_w=640) -> Image`, `cover(img, w, h) -> Image`, moved unchanged (and `_cover` renamed `cover`).
  - `board_images.marker_pixels(img: Image.Image) -> np.ndarray`: centroids of magenta blobs (`r > 200, g < 60, b > 200`), `(N, 2)`.
  - CLI `tools/make_board_compare.py --shots out/board --pass p0 [--before before_ --after after_] [--views PATH] [--cache out/refs] [--calib]`:
    writes `docs/media/photo-board/<pass>/<view>.jpg` (before | after | photo, captioned "Before", "After", "<title> — <author>, <licence>"), `docs/media/photo-board/<pass>/board.gif` (the after panels next to the photos, 2.5 s each), `docs/media/photo-board/<pass>/scores.json` (`{view: score_view(...)}` plus `"worst": worst_regions(...)`), and prints a table and the three worst regions. With `--calib`, compares each `calib_<view>.png`'s marker pixels with `project(camera, landmarks)` and fails if the mean distance is over 0.5 % of the width ("projection convention mismatch").
  - Before writing anything it fetches every photo (exit 1, "could not fetch photo for <view>: <error>") and checks every shot's mean luminance ≥ 10/255 (exit 1, "<file> is black: Lumen or streaming not settled").

- [ ] **Step 1: Write the failing tests:** composites and scores for two synthetic views (tmp dirs, photos served from a pre-filled cache); a fetch error exits 1 naming the view and writes no files; a black shot exits 1 naming the file; `marker_pixels` finds three drawn magenta dots within 1 px; `--calib` fails on markers shifted 2 % and passes at 0.2 %; `tests/test_reference_compare.py` still passes.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement** (move the shared functions first, then the new tool).
- [ ] **Step 4: Run the whole suite; expect all passed.**
- [ ] **Step 5: Commit** `git commit -m "feat(board): before|after|photo composites, board GIF, scores and a projection check"`

---

### Task 8: Baseline in Unreal and the PR

**Files:**
- Create: `docs/media/photo-board/p0/` (composites, `board.gif`, `scores.json`)
- Modify: `README.md` (a "Photo board" section under the look work: what the board is, how to shoot and score it, today's baseline table and the three worst regions), `.gitignore` (`out/board/`, `out/refs/` if not already ignored)

- [ ] **Step 1: Make the project copy** (if `~/Documents/Unreal Projects/RedRocks-look` doesn't exist): `git clone https://github.com/austin80808/red-dead-rocks-unreal.git "~/Documents/Unreal Projects/RedRocks-look"` then `git -C ... lfs pull`. Expected: `Content/RedRocks/Lvl_RedRocks.umap` present, branch `main` at or after the #14 merge.
- [ ] **Step 2: Take the lock:** `~/.local/bin/unreal-lock acquire "Red Rocks look" "P0 photo board baseline (RedRocks-look)"`. Expected: `acquired by Red Rocks look`.
- [ ] **Step 3: Calibration shots:** run `shoot_board.py` with `REDROCKS_BOARD_CALIB=1` (editor GUI, `-ExecCmds="py …/unreal/shoot_board.py"`, the `-ini:…bThrottleCPUWhenNotForeground=False` override from `shoot_vegetation.py`), then `.venv/bin/python tools/make_board_compare.py --shots out/board --pass p0 --calib`. Expected: every view passes the 0.5 % projection check. A failure means `photo_camera.project`'s axes disagree with Unreal's: fix Task 1's model (with a test that reproduces it) before going on.
- [ ] **Step 4: Baseline shots:** run `shoot_board.py` with `REDROCKS_BOARD_PREFIX=before_`, copy the same files to `after_` (P0 changes nothing), then `make_board_compare.py --shots out/board --pass p0`. Expected: six composites, `board.gif`, `scores.json`, a printed table and three worst regions. Look at every composite: a camera that's visibly off goes back to Task 5.
- [ ] **Step 5: Release the lock:** `~/.local/bin/unreal-lock release "Red Rocks look"`.
- [ ] **Step 6: README and commit** `git commit -m "docs(board): today's level on the photo board, the P0 baseline"`
- [ ] **Step 7: PR and merge.** Push `look/p0-photo-board`, open the PR (target `main`; if `feat/realistic-vegetation` hasn't merged, say in the PR that it includes that branch and wait for it), show Austin the composites and `board.gif`, then `merge-gate <PR> --tested ".venv/bin/python -m pytest -q: <N> passed; board calibration and baseline shot in RedRocks-look"`.
