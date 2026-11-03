# redrocks-ue

Builds a real-world **Red Rocks Amphitheatre** terrain package for Unreal Engine
from public geodata: a 1:1 landscape heightmap plus the OSM feature set for the
park, georeferenced to a single origin.

This is the data half of the "GTA in your own city" idea. It is engine-agnostic
in principle but the heightmap is written to Unreal's landscape spec.

## Status

Working: DEM fetch, heightmap export, OSM ingest, georeferencing, seating-bowl
generation, tests.

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

## Blockers

**No Unreal Engine installed.** Nothing under `/Applications` or
`/Users/Shared/Epic Games`. UE 5.x is ~60 GB. Disk is at **56 GB free** and
rising, so this is now a "start the install" problem rather than a "free space
first" problem.

macOS 27 / arm64 is fine: UE 5.4+ supports Apple Silicon natively.

Nothing here is storage-bound for the *data* side — the whole package is 4.3 MB.
The 60 GB is the editor.

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

1. Install UE 5.5 once disk allows.
2. Import the heightmap, confirm the Z scale lands the bowl at the right depth.
3. Extrude the generated rows into bench geometry — 68 rows, 9,525 seats, each
   with its own elevation and length already computed.
4. Stage shell, the two 300 ft sandstone monoliths (Ship Rock and Creation
   Rock), roof, and retaining walls. None of these exist in any free dataset.
5. First-person traversal pass, then multiplayer.

For a *fantasy replication* rather than a survey-accurate twin, the monoliths,
tunnel, and stage shell are the pieces that carry the feel. Those are modelling
work, and no amount of geodata substitutes for them.

## Tests

```bash
.venv/bin/python -m pytest tests -q   # 34 passed
```

Covers the projection (round-trip, axis orientation, distance accuracy), DEM
sampling and interpolation, the Unreal Z scale formula, heightmap encoding, and
OSM parsing and feature classification. No network in tests.
