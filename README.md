# redrocks-ue

Builds a real-world **Red Rocks Amphitheatre** terrain package for Unreal Engine
from public geodata: a 1:1 landscape heightmap plus the OSM feature set for the
park, georeferenced to a single origin.

This is the data half of the "GTA in your own city" idea. It is engine-agnostic
in principle but the heightmap is written to Unreal's landscape spec.

## Status

Working: DEM fetch, heightmap export, OSM ingest, georeferencing, tests.

**Not done: the Unreal project itself.** See [Blockers](#blockers) — there is no
Unreal Engine on this machine and not enough disk to install one.

## What it produces

```
out/
  redrocks_heightmap.png   2017x2017 16-bit heightmap, Unreal landscape format
  manifest.json            Z scale, bbox, origin, feature counts, attribution
```

Run:

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/redrocks build --out out
```

Current output for a 2.4 km box centred on the amphitheatre:

| | |
|---|---|
| Heightmap | 2017 × 2017, 16-bit |
| Ground resolution | 1.19 m per pixel |
| Elevation range | 1753.9 m – 2400.0 m (646 m relief) |
| **Z scale** | **126.2013** |
| OSM features | 15,114 nodes, 1,397 ways |

## Importing into Unreal

1. New level, enable the **Landscape** plugin.
2. Landscape mode → **Import from File** → `out/redrocks_heightmap.png`.
3. Set **Scale X/Y** to `118.99` (cm per pixel, from the manifest) and
   **Z Scale** to `126.2013`.
4. The terrain is now 1:1 with reality. Place the player at the origin from
   `manifest.json` and the amphitheatre is where it should be.

The Z scale formula is `relief / 512 * 100`, because Unreal's landscape maps the
full 0–65535 heightmap range onto 100 units per 512 units of scale.

## Data sources

- **Elevation** — USGS 3DEP bare-earth DEM, public domain, via the
  `3DEPElevation` ImageServer. 1 m source resolution in this area.
- **Features** — OpenStreetMap via the main OSM API (not Overpass, which was
  rate-limited to the point of being unusable). ODbL 1.0, attribution required.

Both are fetched live; nothing is vendored.

## What the data actually gives you

Worth being precise, because it decides how much has to be hand-authored.

**The bowl is real.** The DEM resolves ~32 m of elevation change across the
amphitheatre footprint, and the surrounding hogback, the park road up from
Morrison, and the natural rock formations all come through correctly. The
hillshade is instantly recognisable as Red Rocks.

**The seating rows are not.** Real Red Rocks rises ~61 m from stage to top row
across ~70 rows of ~0.9 m each. At 1.19 m per pixel the DEM cannot resolve
individual rows — it sees the bowl as a smooth slope. The 32 m it does capture
is the underlying geology, not the seating.

So the seating, stage, roof, and retaining walls have to be **modelled by hand**
or generated from the OSM footprint. That is the honest boundary of what
free geodata buys you, and it is the same boundary that stops Google 3D tiles
from being a game world.

## Blockers

Two things stop this from being a playable Unreal level today.

**1. No Unreal Engine installed.** Nothing under `/Applications` or
`/Users/Shared/Epic Games`. Installing UE 5.x is ~60 GB with the engine plus
starter content, and this machine has **22 GB free** on a 460 GB disk that is
95% full. That is not a "run the installer and wait" problem; it needs disk
freed first, or an external drive.

**2. macOS 27 / arm64.** UE 5.4+ supports Apple Silicon natively, so this is
workable, but the editor is heavy and the machine is already tight on space.

Neither is a reason the approach fails — the data pipeline above is the part
that was actually uncertain, and it works. But I am not going to pretend I
built and opened a level I could not build and open.

## Next steps

1. **Free disk or attach a drive**, then install UE 5.5.
2. Import the heightmap, confirm the Z scale lands the bowl at the right depth.
3. Generate the amphitheatre geometry from the OSM `amenity=theatre` footprint
   (28 nodes, already parsed) — seating rake, stage, roof.
4. Add a drivable proxy layer: flatten roads from the 319 OSM road ways so
   vehicles do not bounce on raw terrain.
5. Only then worry about multiplayer.

## Tests

```bash
.venv/bin/python -m pytest tests -q   # 34 passed
```

Covers the projection (round-trip, axis orientation, distance accuracy), DEM
sampling and interpolation, the Unreal Z scale formula, heightmap encoding, and
OSM parsing and feature classification. No network in tests.
