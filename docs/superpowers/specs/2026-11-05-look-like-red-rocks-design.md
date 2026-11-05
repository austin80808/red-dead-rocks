# Make Red Rocks look like Red Rocks: design

Date: 2026-11-05. Status: draft for Austin's review.

## What Austin asked for

"Put all the other activities on the backburner. Make redrocks look great."
Asked what "great" means, he chose **like the real photos**: true shapes, true
colours, real daylight. Accuracy wins over drama. For the method he chose the
**photo board loop**: a fixed set of real photos with a matched game camera at
each, worked in passes with the biggest visible gap first, every pass ending in
a side-by-side against the photos.

On the backburner: controller and phone play (its session is archived), the snow
tracks trial (DeformableSnowSystem) and anything Fab. The realistic-vegetation
work (`feat/realistic-vegetation`) keeps going, because the trees are part of the
look. Its four photo views join the board.

## Success

Red Rocks is done looking right when, for every board view:

1. **Side by side, it reads as the same place.** Austin looks at the game shot
   next to the photo and says yes. This is the real gate; the numbers below only
   catch drift and show progress.
2. **Colour:** each sampled region (sky, sunlit sandstone, shaded sandstone,
   treads, risers, plants, the plains) is within CIE76 ΔE 10 of the photo, and
   none is over 20. The vegetation work uses ΔE ≤ 15; this is stricter because
   the rocks and seats fill most of every frame.
3. **Shape:** the skyline (where rock, roof or hill meets sky) is within 2 % of
   the image height of the photo's, on average along its length, for views
   whose geometry is in scope of a finished pass.

A pass is done when its views meet those targets (or Austin accepts what is
left) and he has seen its side-by-sides.

## Where it stands today

- The level is a good blockout: the 3DEP terrain, the ml0w drone scan over the
  bowl and rocks, terraced seating with aisle stairs and the sound booth
  (#53, #54), the scan's collision cut out of the rows. The level is no longer
  mirrored (#55).
- `unreal/setup_lighting.py` lights it at golden hour: the sun at bearing 75°,
  24° up, a SkyAtmosphere, height fog at density 0.05 and sparse volumetric
  clouds. `unreal/configure_level.py` turns on Lumen and histogram auto exposure
  clamped to a daylight EV100 range, and grades the image. The editor warns that
  Lumen and the sky capture will clip at that exposure range.
- Visible gaps in today's shots (`docs/media/level-frame/`): grey haze, a dark
  smear of scan behind the top rows, the booth as a plain white box, a stage
  blockout instead of the real roof and towers, debug text in the shots.
- Creation Rock and Ship Rock outside the scan are still OSM extrusions.
- `feat/realistic-vegetation` built the first photo comparison: photo-matched
  cameras in `docs/media/vegetation-real/views.json`, `tools/make_reference_compare.py`
  (before, after and photo panels, captioned with the photo's author and licence)
  and `redrocks.vegetation_metrics.delta_e`. The board generalises these instead
  of starting over.

## The photo board

Six Wikimedia Commons photos, daylight, all freely licensed:

| View | Photo | Author, licence | What it judges |
|---|---|---|---|
| `postcard` | [Red Rocks Amphitheater.JPG](https://commons.wikimedia.org/wiki/File:Red_Rocks_Amphitheater.JPG) | Scapler, CC BY-SA 3.0 | from the top row: the rows, stage roof, Stage Rock, the plains to Denver |
| `up_the_rows` | [Red Rocks Amphitheatre, Colorado 28.jpg](https://commons.wikimedia.org/wiki/File:Red_Rocks_Amphitheatre,_Colorado_28.jpg) | Sarbjit Bahga, CC BY-SA 4.0 | from the lower rows: the monolith over the seats, an aisle, sky |
| `across_the_rows` | [Red Rock Amphitheatre Seatings.jpg](https://commons.wikimedia.org/wiki/File:Red_Rock_Amphitheatre_Seatings.jpg) | Mitul0520, CC BY-SA 3.0 | mid-bowl: treads, risers and planks in the sun, the rock wall beside them |
| `stage_front` | [Red Rock Amphitheatre.jpg](https://commons.wikimedia.org/wiki/File:Red_Rock_Amphitheatre.jpg) | Mitul0520, CC BY-SA 3.0 | from the front rows: the stage roof, its two stone towers, the rock behind |
| `monolith_plaza` | [Red Rocks Amphitheatre, Colorado 13.jpg](https://commons.wikimedia.org/wiki/File:Red_Rocks_Amphitheatre,_Colorado_13.jpg) | Sarbjit Bahga, CC BY-SA 4.0 | the top plaza: a monolith's full face, its strata and colour |
| `aerial` | [Aerial view of Red Rocks Amphitheatre, January 2013.jpg](https://commons.wikimedia.org/wiki/File:Aerial_view_of_Red_Rocks_Amphitheatre,_January_2013.jpg) | Stoermerjp, CC BY-SA 3.0 | the whole bowl, both monoliths and the hills around it |

The four vegetation views (`trail`, `ponderosa`, `junipers`, `hillsides`) join
the board when `feat/realistic-vegetation` merges.

**Data.** `docs/media/photo-board/views.json`, one entry per view in the
vegetation schema: the photo (Commons page, file URL, author, licence), the
camera (x, y, z in level cm, pitch, yaw, fov), the sun for that view
(azimuth and elevation in degrees) and the samples (regions in the photo paired
with the same feature in the game shot, each with a kind: `sky`, `rock_lit`,
`rock_shade`, `tread`, `riser`, `plants`, `plains`). Photos download into
`out/refs/` and are never committed. Composites are committed with the photo's
title, author and licence in the caption, and the LICENSE file says the board
composites are CC BY-SA because of the photos in them.

**Cameras.** `tools/fit_photo_camera.py` fits each view's camera from 6-10
landmarks clicked once in the photo and taken from the pipeline's own data:
the stage roof corners, the towers' tops, the monolith summits, row ends at
known row numbers, the booth. It solves the pinhole camera (position, yaw,
pitch, roll and fov) by least squares in numpy, with no new dependency, and
writes the camera and its reprojection error into `views.json`. A fit with a
mean landmark error over 1.5 % of the image width is refused. This replaces
fitting cameras by hand.

**Sun per view.** Each view sets the sun the photo was taken under, judged from
its shadows (the photo's EXIF date is only a starting guess; several cameras
logged the wrong time zone). The game's own default light becomes one clear
summer late morning, chosen to match most of the board.

**Shooting.** `unreal/shoot_board.py` runs in a project copy: for each view it
places the camera, sets that view's sun, hides debug text and the HUD, waits for
Lumen and streaming to settle and writes `out/board/<pass>_<view>.png`.
`tools/make_board_compare.py` (the vegetation compare tool, generalised) then
writes the before | after | photo composites, a GIF per pass, and
`docs/media/photo-board/<pass>/scores.json` with ΔE per region and the skyline
error per view. It also prints the three worst regions on the board, which
picks the next thing to fix.

**Skyline error.** Sky is segmented in both images (bright, low-saturation or
blue pixels connected to the top edge); the skyline is the lowest sky pixel per
column. The error is the mean absolute difference between the two skylines over
the columns both have, as a fraction of image height. People, flags and lens
flare in a photo are masked by a per-view `skyline_ignore` list of boxes.

## The passes

Each pass is one branch and one PR in red-dead-rocks (and in
red-dead-rocks-unreal when content changes), merged through `merge-gate` after
Austin has seen its side-by-sides.

**P0 The board.** Generalise the vegetation compare tool, fit the six cameras,
write `shoot_board.py`, shoot the board as it is today and score it. No visual
change. Done when every view shoots cleanly, every camera fit passes its 1.5 %
check, and today's scores are committed as the baseline.

**P1 Light and air.** The sun per view and the new default daylight; the sky
colour and cumulus that most of the photos have; less haze (the photos show
the plains to Denver crisp at 25 km); an exposure range that no longer clips;
a grade that matches the photos' sandstone and sky. Debug text goes from all
shots. Expected to move ΔE most of any pass, because every pixel is lit by it.

**P2 Rocks.** Find and fix the dark smear behind the top rows. Calibrate the
scan's colour against the photos: the scan's texture has the drone day's light
and shadows baked in, so this pass evens out the baked shade and matches the
sandstone's colour in sun and in shade. Replace the OSM extrusions of Creation
and Ship Rock outside the scan with shapes from USGS 3DEP lidar (README Next
steps, item 2). Done when `up_the_rows`, `monolith_plaza` and `aerial` meet the
colour and skyline targets.

**P3 Stage.** The real stage: its roof (shape, thickness, colour, the trusses
under it), the two stone towers either side and the deck, measured from
`stage_front` and `postcard`, the scan and published dimensions. Done when
`stage_front` and `postcard` meet the targets.

**P4 Details.** The sound booth's real look in place of the white box,
handrails, the top plaza's buildings, then the parking lots (stall lines,
curbs, lights) last. The board decides the order: the worst regions go first.

Every pass re-shoots the whole board, so a change that helps one view and hurts
another shows up in its scores.

## Ground rules

- Nothing paid or from Fab goes in git; both repos are public.
- Building and shooting happen in a project copy (`RedRocks-look`), never in
  Austin's own project, under the shared lock
  (`unreal-lock acquire "Red Rocks look" "<why>"`). Austin's project is
  rebuilt only when he says go, because `rebuild_level.py` wipes the level.
- The play-tests (`playtest_explore.py`: seating, bowl, climb and the rest) and
  `tools/check_seating_clear.py` still pass after every pass.
- Each pass's PR carries its board GIF and scores.

## Out of scope

Night and concert looks, crowds, interiors (the visitor centre, the Hall of
Fame), weather other than a clear day, and art direction away from the photos.

## Risks

- **Camera fits.** A wrong camera makes every score wrong. The 1.5 % landmark
  check refuses a bad fit, and the side-by-side shows one at a glance.
- **Photo variety.** The photos are from different years, seasons and cameras,
  with their own white balance. Scores are a guide; Austin's eye is the gate.
  The board leaves out photos with snow, night or heavy crowds.
- **Baked light in the scan.** Its texture carries one afternoon's shadows. If
  evening them out isn't enough, P2 may need a de-lit texture, which is a
  bigger piece and gets its own design.
- **Lidar access.** 3DEP lidar for Jefferson County is public, but the point
  clouds are large; P2 downloads only the tiles around the two monoliths, and
  asks Austin before the download.
- **Editor time.** Shooting the board takes the shared lock for about 15-30
  minutes per pass; other sessions wait behind it.

## Tests

- Camera fit: synthetic cameras project known points; the fit recovers
  position within 0.5 m and angles within 0.5°, and refuses noisy landmarks.
- Skyline: synthetic images with known skylines; masked boxes are ignored.
- Scores: `delta_e` against published CIELAB pairs (as the vegetation tests do).
- `views.json`: every view has a photo with author and licence, a fitted
  camera, a sun and at least four samples.
- The existing suite, the play-tests and the seating clearance check stay green.
