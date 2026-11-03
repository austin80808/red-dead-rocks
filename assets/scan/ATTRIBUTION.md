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

- Extent: **78.9 × 163.3 × 52.1 m** (raw units are metres)
- `upAxis = "Y"` — **must be rotated to Z-up for Unreal**
- `metersPerUnit = 0.01` in the USD header, but the raw extents only make sense
  as metres. Treat the declared unit as unreliable and scale by 1.0, then verify
  against the 120 m bowl width.

## Why this matters

This is real drone photogrammetry of the amphitheatre — the actual rock surfaces,
not an extrusion. It supersedes the parametric monoliths for visual fidelity,
though the monoliths remain useful as clean collision proxies.

## Attribution requirement

CC BY 4.0 obliges crediting ml0w wherever this appears. Keep this file with the
asset and surface the credit in any build that ships.
