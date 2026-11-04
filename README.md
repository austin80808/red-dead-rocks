# red-dead-rocks

Builds a real-world **Red Rocks Amphitheatre** for Unreal Engine from public
geodata: a 1:1 terrain from the USGS DEM, the two flanking monoliths extruded
from their surveyed OSM footprints, and the OSM feature set for the park, all
georeferenced to a single origin.

This is the data half of the "GTA in your own city" idea. The Python package
writes engine-agnostic OBJ and heightmap files; the `unreal/` scripts import
them into a UE 5.8 project and build the level.

The local checkout lives at `~/dev/redrocks-ue`.

## Status

Working: DEM fetch, heightmap and terrain-mesh export, OSM ingest,
georeferencing, seating-bowl generation, monolith meshes, tests.

**In Unreal:** the project is at `~/Documents/Unreal Projects/RedRocks`
(UE 5.8, Apple Silicon). Its level `/Game/RedRocks/Lvl_RedRocks` holds the
terrain mesh, Creation Rock and Ship Rock at their real positions, a sun, sky
atmosphere, sky light, height fog, a PlayerStart and a framing camera. It is the
editor startup map and the game default map.

The Unreal project is versioned separately in
[red-dead-rocks-unreal](https://github.com/austin80808/red-dead-rocks-unreal)
(private, assets in Git LFS). After a rebuild that you want to keep, commit
there:

```bash
cd "$HOME/Documents/Unreal Projects/RedRocks"
git add -A && git commit -m "Rebuild: <what changed>" && git push
```

To roll the level back, `git checkout <commit> -- Content/` with the editor
closed.

The sandstone material is applied to the terrain and both rocks. Written but not
yet run in the level: the bench import and the ml0w scan import (see the script
table). Not done: the stage, tuning the material (the ground reads pale and the
fine bands shimmer at distance).

## What it produces

```
out/
  redrocks_heightmap.png   2017x2017 16-bit heightmap, Unreal landscape format
  RedRocks_Terrain.obj     377x377 grid terrain mesh, 282,752 triangles, cm
  Creation_Rock.obj        north monolith, 91.4 m, watertight
  Ship_Rock.obj            south monolith, 61.0 m, watertight
  Benches.obj              68 bench rows, 816 triangles, terrain frame
  manifest.json            Z scale, bbox, origin, venue, monoliths, attribution
```

Clone with Git LFS installed: the scan, rock models and textures in `assets/`
are LFS files, and without it you get 133-byte pointer files instead.

```bash
git lfs install   # once per machine
git clone https://github.com/austin80808/red-dead-rocks.git
```

Run:

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/redrocks build --out out
```

`redrocks build` writes everything above. `--terrain-grid` sets the terrain
mesh resolution (default 377; `0` skips it).

Benches are one solid box per generated row, 0.9 m deep and 0.45 m tall, in
the terrain mesh's frame so they need no transform in Unreal. Rows are straight
chords, because the slicer cuts the theatre footprint with straight lines, and
each bench is flat at its row centre's DEM elevation.

Current output for a 2.4 km box centred on the amphitheatre:

| | |
|---|---|
| Heightmap | 2017 × 2017, 16-bit |
| Ground resolution | 1.19 m per pixel |
| Elevation range | 1753.9 m – 2400.0 m (646 m relief) |
| **Z scale** | **126.2013** |
| Terrain mesh | 377 × 377 vertices, 282,752 triangles |
| OSM features | 15,114 nodes, 1,397 ways |

### Terrain mesh

The terrain goes into Unreal as a static mesh, not a Landscape. Landscape
heightmap import can't be driven from Python in vanilla UE, while an OBJ imports
through `AssetImportTask` like any other asset, including from a headless
commandlet. The cost is losing Landscape features (paint layers, grass,
Landscape LOD); the mesh can be swapped for a Landscape later.

Axes are X east, Y north, Z up, in centimetres, centred on the origin, with the
lowest point at Z = 0. Faces wind counter-clockwise seen from above, so normals
point up.

### Monoliths

Both flanking rocks are extruded from their **surveyed OSM footprints**, so the
plan-view silhouette is real rather than invented. Heights come from the
official figures, not the DEM — the DEM reads 20–40 ft high because it samples
the ridge behind each rock.

| | footprint | height | top elevation |
|---|---|---|---|
| Creation Rock (north) | 163.6 × 94.8 m, 8,132 m² | 91.4 m (300 ft) | 2023.8 m |
| Ship Rock (south) | 236.1 × 177.2 m, 18,647 m² | 61.0 m (200 ft) | 1993.4 m |

Meshes are watertight, outward-facing, and their volume matches
footprint × height to six digits. Units are centimetres, matching Unreal. Side
walls carry UVs with perimeter distance along U and height along V, so strata
banding in a material runs horizontally.

The 30° westward strata dip is **not** applied as a geometric shear. The
monoliths are vertical columns; the dip is a property of the bedding and belongs
in the material as diagonal banding. Shearing the mesh would fake a lean that
does not exist.

## Importing into Unreal

The scripts in `unreal/` run headless against the project:

```bash
UE="/Users/Shared/Epic Games/UE_5.8/Engine/Binaries/Mac/UnrealEditor-Cmd"
PROJ="$HOME/Documents/Unreal Projects/RedRocks/RedRocks.uproject"
"$UE" "$PROJ" -run=pythonscript -script=unreal/import_redrocks.py -unattended -nosplash -stdout
```

Run them in this order; each writes a log to `/tmp/redrocks_*.log`:

| Script | Does |
|---|---|
| `import_redrocks.py` | imports the terrain, rocks and benches OBJs to `/Game/RedRocks` and places them in `Lvl_RedRocks`. **Clears every actor in the level first.** |
| `setup_lighting.py` | adds sun, sky atmosphere, sky light, height fog, PlayerStart |
| `configure_level.py` | fixed-exposure post-process volume and a framing camera |
| `fix_camera.py` | lifts the camera above the 645 m ridge, removes test actors |
| `setup_materials.py` | `M_Sandstone` with 30° strata bands; rock and ground instances; bench material |
| `import_scan.py` | imports the ml0w scan with Nanite and credits it; run after `import_redrocks.py` |
| `render_preview.py` | renders the level to `Saved/redrocks_preview.png` |
| `diagnose.py` | lists actors, mesh triangle counts and the startup map |

Close the editor before running them, or the commandlet and the editor will
both hold `Lvl_RedRocks`. Inside a running editor, run one from the console
instead: `py "/Users/austin/dev/redrocks-ue/unreal/diagnose.py"`.

The `probe_*.py` scripts are the experiments that found the import problems
below; they are kept for reference.

### Landscape instead of a mesh

To use the heightmap as a Landscape:

1. New level, enable the **Landscape** plugin.
2. Landscape mode → **Import from File** → `out/redrocks_heightmap.png`.
3. Set **Scale X/Y** to `118.99` (cm per pixel, from the manifest) and
   **Z Scale** to `126.2013`.
4. The terrain is now 1:1 with reality. Place the player at the origin from
   `manifest.json` and the amphitheatre is where it should be.

The Z scale formula is `relief / 512 * 100`, because Unreal's landscape maps the
full 0–65535 heightmap range onto 100 units per 512 units of scale.

### Unreal gotchas

Each of these cost real time:

- **`unreal.Rotator` positional order is (roll, pitch, yaw).** Always pass
  keywords: `unreal.Rotator(pitch=-38, yaw=145, roll=0)`. Positional
  `(-38, 145, 0)` gives pitch 145°, which puts the sun below the horizon:
  the level renders as night, with black terrain and black sky.
- **OBJs need a UV channel.** Without `vt` lines, the OBJ importer fails an
  ensure and keeps only a fraction of the triangles (a 282,752-triangle terrain
  came back with 3,716). Every exported OBJ now carries UVs.
- **`spawn_actor_from_object` does nothing in a headless commandlet.** Spawn a
  `StaticMeshActor` with `spawn_actor_from_class` and set its mesh instead.
- **A minimized editor looks frozen.** macOS throttles it to a few frames per
  second and it ignores input until it's brought forward and finishes compiling
  shaders.
- **The project still contains the FirstPerson template level**
  (`/Game/FirstPerson/Lvl_FirstPerson`: a grey platform with yellow pads). It
  isn't Red Rocks; open `Lvl_RedRocks`.

## Assets

| Folder | What | Licence |
|---|---|---|
| `assets/rocks` | 5 Poly Haven rock and cliff FBX models, for surface detail and scatter | CC0 |
| `assets/scan` | ml0w's drone photogrammetry of Red Rocks, 2,995,743 triangles, Y-up USD | **CC BY 4.0, attribution required** |

Each folder has an `ATTRIBUTION.md`. The scan's credit to ml0w has to appear in
any build that ships.

## Data sources

- **Elevation** — USGS 3DEP bare-earth DEM, public domain, via the
  `3DEPElevation` ImageServer. 1 m source resolution in this area.
- **Features** — OpenStreetMap via the main OSM API (not Overpass, which was
  rate-limited to the point of being unusable). ODbL 1.0, attribution required.

Both are fetched live; nothing is vendored.

## What the data actually gives you

**The bowl is real.** The DEM resolves 32.2 m of relief across the amphitheatre
footprint against an official stage-to-top-row rise of 30.48 m — a 6% error. The
surrounding hogback, the park road up from Morrison, and the rock formations all
come through correctly. The hillshade is instantly recognisable as Red Rocks.

**The seating rows are not in the DEM.** One row rises 0.44 m; a DEM pixel spans
1.19 m. Individual rows are 2.7× below the sampling floor, so terrain data can
never give you benches. They are generated instead, by slicing the OSM
`amenity=theatre` footprint into 70 cross-sections and distributing the official
9,525 seats across them in proportion to row length.

That is the honest boundary of free geodata: it gives you the *setting* exactly,
and the *venue* only as a footprint to build from.

### Verified against official figures

| | generated | official | error |
|---|---|---|---|
| Seats | 9,525 | 9,525 | 0 |
| Rise, stage → top row | 30.23 m | 30.48 m | −0.25 m |
| Rows | 68 | 70 | −2 |
| Seat pitch | 0.422 m (16.6 in) | 2.5 mi / 9,525 seats | — |

The 2 missing rows are where the footprint is too narrow at the stage end to
produce a cross-section; they are dropped rather than faked.

Seat pitch is the strongest independent check: 2.5 miles of bench across 9,525
seats gives 16.6 inches per seat, which matches Red Rocks' famously
shoulder-to-shoulder benches. That figure was never an input — it falls out of
the official capacity and bench length.

## Storage

Disk was the original blocker; UE 5.8 is now installed under
`/Users/Shared/Epic Games/UE_5.8`, with about 46 GB free on the system disk
(2026-11-03).

Two files in the repo are over GitHub's 50 MB recommendation:
`assets/scan/usdz_extract/scene.usdc` (70 MB) and
`assets/rocks/sand_rocks_small_01.fbx` (57 MB). Both are under the 100 MB hard
limit; move them to Git LFS before adding anything larger.

### MEGA as storage

MEGA is a viable target for the *assets*, not the *working tree*.

- 3.3 TB quota, 2.2 TB free on the account already configured here.
- `mega-put` / `mega-get` / `mega-sync` are installed and authenticated.
- **No macFUSE**, so `mega-mount` is not available — there is no mounted volume
  to run the editor from. MEGA has to be sync-and-copy, not a live working disk.

That is the right shape anyway: closing a project on a network mount is slow
enough that Unreal's own docs advise against it. Keep the project and DerivedData
Cache on local disk, sync the source and built packages to MEGA.

## Next steps

1. Run `import_redrocks.py` (benches) and `import_scan.py` in the level, then
   tune the scan's placement constants by eye and decide whether it replaces the
   extruded monoliths visually, with the monoliths kept as collision.
2. Tune `MI_Sandstone_Ground`: less exposure, thicker bands.
3. Curve the benches: have the slicer produce arcs instead of straight chords.
4. Stage shell, roof, and retaining walls. None of these exist in any free
   dataset.
5. First-person traversal pass, then multiplayer.

For a *fantasy replication* rather than a survey-accurate twin, the monoliths,
tunnel, and stage shell are the pieces that carry the feel. Those are modelling
work, and no amount of geodata substitutes for them.

## Tests

```bash
.venv/bin/python -m pytest tests -q   # 121 passed
```

Covers the projection (round-trip, axis orientation, distance accuracy), DEM
sampling and interpolation, the Unreal Z scale formula, heightmap encoding, OSM
parsing and feature classification, the seating bowl, monolith geometry, the OBJ
exports (terrain grid, benches, winding, and a valid UV on every face), and the
CLI build. No network in tests.

The `unreal/` scripts are only checked offline: they must parse, and none may
pass `unreal.Rotator` angles positionally. Running them needs the editor.
