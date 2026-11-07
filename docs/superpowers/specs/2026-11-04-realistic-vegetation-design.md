# Realistic trees and shrubs: design

Date: 2026-11-04. Status: design approved in chat by Austin; this spec is waiting on his review.

## Goal

Make the trees and shrubs on the Red Rocks hillsides look like the real ones in summer: the right
form, scale, colour and density for ponderosa pine, the two junipers, Gambel oak and mountain
mahogany, and the right way of sitting on the slopes. Austin judges it from captures set side by side
with public photos of Red Rocks Park.

Done when Austin approves the look from the side-by-side shots, the change is merged through
`merge-gate`, and the README's progress GIF has a new frame.

## Decisions already made

- **Season: summer.** Scrub oak is in leaf and the grass is green-gold. The N snow mode keeps
  working on top of it.
- **Approach: improve the existing generator.** `src/redrocks/vegetation.py` keeps growing every
  model from a seed and painting every texture with Pillow. Nothing is downloaded. The
  `vegetation.json` format and the three parts per model (Trunk, Crown, Leaves) stay as they are, so
  `unreal/setup_vegetation.py` keeps working with them.
- **Rejected: downloaded trees** (Fab or Megascans, or Unreal's experimental vegetation editor).
  The public repo can't carry their assets, merge-gate holds outside downloads, and the editor
  doesn't fit the scripted rebuild.
- **Rejected: retuning colour and placement only.** It leaves the giant leaves, which are the main
  thing that makes the plants look fake.
- **Two PRs.** PR 1 changes the plants and touches only the vegetation files. PR 2 adds grass and
  changes `unreal/setup_materials.py`, so it waits until the terraced-seating PR (which edits that
  file) has merged.

## What is wrong today

Measured against the reference photos (list at the end):

1. **The leaves are drawn far too big.** Each leaf card shows leaves the size of a hand, so the oak
   and mahogany look like giant houseplants and the junipers look like broadleaf trees.
2. **The colours are too bright and saturated.** Ponderosa trunks are bright orange (`#8D3719`
   sampled in game). In the photos they are dark grey-brown from any distance (`#5A5236`).
3. **The hillsides are too bare.** About 99,500 plants stand singly on bare ground. The real lower
   slopes are continuous oak and mahogany thickets. Shrubs are culled past 1.2 km, so the aerial
   view is nearly empty.
4. **There is no grass between the plants.** The ground shows as orange dirt, so the plants look
   stuck onto the slope.

## Colour targets

These are median pixels sampled from the photos (sunlit, summer). The atlas albedo is tuned until
the same regions sampled in a capture land within ΔE 15 (CIELAB) of the photo; the PR reports each
species' ΔE.

| What | Photo colour | Source |
|---|---|---|
| Rocky Mountain juniper foliage | `#4B583E`–`#545E4D` (dark grey-green) | ref09 |
| Gambel oak foliage | `#525D21`–`#656F3D` (olive) | ref09 |
| Ponderosa needles | `#68744B` (yellow-green) | ref02 |
| Ponderosa trunk and branches at distance | `#5A5236` (dark grey-brown) | ref02 |
| Summer grass | `#96914D`–`#B9A775` (green-gold) | ref09 |
| Mountain mahogany foliage | greyer and lighter than oak; no clean sample, judged by eye | ref04, ref13 |

## PR 1: the plants

### Leaf scale

Each atlas tile is painted at a real size: the painter gets the card's width in metres and draws
leaves, sprays and needles in metres too.

| Atlas | Card width | Leaf, spray or needle size |
|---|---|---|
| Juniper | 0.25–0.40 m | scale-leaf branchlets 1–2 mm thick, in sprays 10–20 cm long |
| Pine | 0.35–0.60 m | needles 10–18 cm, in tufts at the twig end |
| Oak | 0.30–0.45 m | lobed leaves 5–12 cm |
| Mahogany | 0.20–0.35 m | narrow leaves 2–3.5 cm, plus a few feathery seed plumes |

There are more, smaller cards. The opaque crown core carries each plant's mass, and the masked leaf
cards sit only on the outer shell. That keeps masked overdraw, the costly part, down.

### Forms

| Species | Target |
|---|---|
| **Ponderosa** (5 variants) | 12–25 m once scaled. Straight trunk, 0.35–0.8 m across at 1.4 m. Crown on the top 40–60 % of the height, open and irregular. Branches thicker and crooked, dipping and then turning up at the tips. Needles in dense round tufts at the branch tips, with sky showing through between them. Variants: 1 young conical, 3 mature, 1 old flat-topped. Dead stubs stay on the clear lower trunk. |
| **Rocky Mountain juniper** (3) | 3–6 m. Dense oval-to-conical mass that is almost solid from the side, with a shaggy edge. One or two trunks. |
| **One-seed juniper** (3) | 2–4 m. Several stems from the base and a broad, rounded crown down to the ground. Same foliage treatment as Rocky Mountain juniper. |
| **Gambel oak** (3 bushes, 3 patches) | Bushes 1.5–4 m once scaled. New `OakPatch` models are clonal patches 3–6 m across, many stems, with a continuous canopy. |
| **Mountain mahogany** (3) | 1–3 m. Twiggy, open, grey-green, with small leaves. |

### Bark

- **Ponderosa**: dark grey-brown from a distance, with cinnamon plates and black fissures only within
  about 10 m. The material blends from the far colour to the near one by camera distance.
- **Juniper**: shreddy grey-brown, as now but less saturated.
- **Shrubs**: grey stems.

### Where they grow

- **Shrub thickets** cover the lower slopes and draws. Inside a thicket, at least 60 % of the
  ground is under shrub canopy. Between thickets are grassy openings with no more than 15 % cover.
- **Junipers** stand singly or in small groups through and beside the thickets, more of them on rocky
  south-facing ground.
- **Ponderosa** forms open woodland and groves of 3–8 trees on the upper slopes and north-facing
  ground.
- Masks stay as they are: no plants on cliffs, the scanned bowl, roads, paths, parking, benches, the
  stage or the sled run.

### Distance

- **Shrub crowns** (opaque) are not culled, so the hillsides read as covered from the air.
- **Shrub leaf cards** fade out from 320 m and are gone by 400 m.
- **Trees** keep every part at every distance, as now.

### Frame rate

1. Before changing anything, measure the current frame rate with `unreal/shoot_vegetation.py`. Run it
   with the editor in the foreground and `bThrottleCPUWhenNotForeground=False`, and read `stat unit`
   at the bowl-top, hillside and aerial views.
2. PR 1 may be at most 15 % slower than that at each view. The PR lists before and after numbers.
3. If it's over budget, cut in this order: cards per clump, shrub leaf-card cull distance, thicket
   density.

### Tests (`tests/test_vegetation.py`)

The tests check that:

- each atlas paints its leaves within the real-size ranges above, for its card width;
- every model's leaf cards use its atlas's card width (±20 %);
- the alpha-weighted mean colour of each atlas (sRGB) is within ΔE 20 of its photo colour. This guards
  against a return to bright greens; the in-game ΔE 15 check is the real test;
- juniper crowns are dense: rasterised from the side at 5 cm, crown cores and leaf cards fill ≥ 85 % of their convex hull;
- ponderosa crowns are open: the same fill is 45–75 %, and ≥ 70 % of needle cards sit within 1 m of a
  branch tip;
- the ponderosa crown base sits at 40–60 % of the height, and the trunk is 0.35–0.8 m across at 1.4 m
  over the instance scale range;
- thicket cover is ≥ 60 % inside thickets and ≤ 15 % in the openings, on a synthetic ground;
- shrub crowns have no end cull distance, and shrub leaves start fading at 320 m and are gone by 400 m;
- the existing tests still pass: watertight outward trunks, UVs and normals on every face, models
  identical byte for byte every build, and the masks.

## PR 2: grass (after the seating PR merges)

- **Terrain colour.** The terrain material's grass on gentle ground moves to the summer targets
  (`#96914D`–`#B9A775`), and `GrassAmount` rises so the ground between thickets reads as grass,
  not bare soil. Bare red soil stays on paths, steep ground and rock.
- **Grass clumps.** New generated bunchgrass clump models (masked cards, painted with Pillow) go on
  gentle ground outside the masks and outside the thickets. They're culled at 150 m. Density starts
  at one clump per 9 m² and is tuned against the same 15 % frame budget, measured from the PR 1
  numbers.
- Grass gets its own matched views and tests on the same pattern as PR 1.

## Proof: side by side with the photos

`unreal/shoot_vegetation.py` gains four views framed to match the photos:

| View | Photo |
|---|---|
| A trail through the scrub toward the rocks | ref09, the trail view, Aug 2021 |
| A lone ponderosa against the sky | ref02, the ponderosa photo |
| Junipers up close in front of rock | ref15, the trail views, and ref11, the park view |
| The hillsides from above | ref12, RedRocks.jpg |

- **The comparison tool.** `tools/make_reference_compare.py` builds each comparison as before | after
  | photo, with the photo's title, author and licence printed under it. The photos are fetched from
  Commons when the tool runs and are not committed.
- **Licensing.** The composites are committed to `docs/media/vegetation-real/`. Because they include
  CC BY-SA photos, the README credits each photo and marks the composites CC BY-SA.
- **The progress GIF.** It gets the "after" bowl-view frame appended to
  `docs/media/progress/stages.json`, then `tools/make_progress_gif.py` runs. That tool arrives with
  the seating PR, so the frame waits for that merge too.

## Where the work happens

- **Code:** worktree `~/dev/redrocks-ue-trees`, branch `feat/realistic-vegetation`, cut from `main`.
  It gets rebased onto main once the seating PR merges.
- **Unreal:** project copy `~/Documents/Unreal Projects/RedRocks-trees`, refreshed from Austin's
  `RedRocks` project first. It's never Austin's real project.
- **The Unreal lock:** every editor or build step takes `unreal-lock` as "Red Rocks" and releases it
  after. The seating session and the controllers spike share it.

## Out of scope

- New species (yucca, rabbitbrush, skunkbush, chokecherry). They could come later.
- Wind animation.
- Autumn or winter colours.
- Changes to the scanned bowl's own plants.

## Reference photos (Wikimedia Commons)

| Id | Photo | Author | Licence |
|---|---|---|---|
| ref02 | [Conifer (Red Rocks Park, west of Denver, Colorado, USA)](https://commons.wikimedia.org/wiki/File:Conifer_(Red_Rocks_Park,_west_of_Denver,_Colorado,_USA).jpg) | James St. John | CC BY 2.0 |
| ref04 | [Foothills in Red Rocks Park, Morrison, Colorado](https://commons.wikimedia.org/wiki/File:Foothills_in_Red_Rocks_Park,_Morrison,_Colorado.jpg) | Lovemedead | CC BY-SA 4.0 |
| ref09 | [Red Rocks Park in Denver](https://commons.wikimedia.org/wiki/File:Red_Rocks_Park_in_Denver.jpg) | Bigmacthealmanac | CC BY-SA 4.0 |
| ref11 | [Red Rocks Park, Morrison, Colorado](https://commons.wikimedia.org/wiki/File:Red_Rocks_Park,_Morrison,_Colorado.jpg) | Lovemedead | CC BY-SA 4.0 |
| ref12 | [RedRocks.jpg](https://commons.wikimedia.org/wiki/File:RedRocks.jpg) | Hogs555 | CC BY-SA 3.0 |
| ref13 | [Rocky ridge in Red Rocks Park, Morrison, Colorado](https://commons.wikimedia.org/wiki/File:Rocky_ridge_in_Red_Rocks_Park,_Morrison,_Colorado.jpg) | Lovemedead | CC BY-SA 4.0 |
| ref15 | [Trail views in Red Rocks Park, Morrison, Colorado](https://commons.wikimedia.org/wiki/File:Trail_views_in_Red_Rocks_Park,_Morrison,_Colorado.jpg) | Lovemedead | CC BY-SA 4.0 |
