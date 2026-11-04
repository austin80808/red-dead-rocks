# Red Rocks, Colorado — photogrammetry scan

**Source:** https://sketchfab.com/3d-models/red-rocks-colorado-9dec580ed1254c13be687a93bae55ae1
**Author:** ml0w (https://sketchfab.com/ml0w)
**Licence:** CC BY 4.0 — attribution required, commercial use permitted
**Original data:** DroneMapper sample dataset (drone photogrammetry, Agisoft Metashape)

## Files

| File | Notes |
|---|---|
| `usdz_extract/scene.usdc` | 73.7 MB USD crate, 2,995,743 triangles |
| `usdz_extract/0/material0_baseColor.jpg` | 1.1 MB albedo |
| `usdz_extract/0/material0_normal.jpg` | 2.5 MB normal map |
| `usdz_extract/0/material0_metallicRoughness_*.jpg` | ORM channels |

## Geometry

- 31 mesh prims sharing one frame, `upAxis = "Y"`, `metersPerUnit = 0.01`, with
  a 2.02 scale on `/scene/Meshes`. World extent **1050 × 300 × 1093 units**
  (x, up, z), read with `unreal/probe_scan.py`. An earlier note here said
  78.9 × 163.3 × 52.1 m; nothing in the file has that size.
- The units are Sketchfab-normalised, not real-world. It covers the Visitor
  Center side of the park (roads, parking, buildings, outcrops), not just the
  bowl.
- Real scale and placement come from fitting its heights to the 3DEP terrain
  (`tools/fit_scan.py`): about 0.31 m per unit, so roughly 340 m across before
  the import's own scaling; it lands about 400 m north of the bowl. The fit's
  detrended correlation is ~0.70, so treat it as good, not surveyed.

## Why this matters

This is real drone photogrammetry of the amphitheatre — the actual rock surfaces,
not an extrusion. It supersedes the parametric monoliths for visual fidelity,
though the monoliths remain useful as clean collision proxies.

## Attribution requirement

CC BY 4.0 obliges crediting ml0w wherever this appears. Keep this file with the
asset and surface the credit in any build that ships.
