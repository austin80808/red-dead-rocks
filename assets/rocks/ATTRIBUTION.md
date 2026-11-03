# Rock assets

All from [Poly Haven](https://polyhaven.com), **CC0** (public domain) — no
attribution legally required, credited here anyway.

| File | Source | Notes |
|---|---|---|
| `rock_moss_set_01.fbx` | polyhaven.com/a/rock_moss_set_01 | mossy boulder cluster |
| `rock_moss_set_02.fbx` | polyhaven.com/a/rock_moss_set_02 | mossy boulder cluster |
| `namaqualand_cliff_02.fbx` | polyhaven.com/a/namaqualand_cliff_02 | cliff face, best match for sandstone |
| `namaqualand_boulder_02.fbx` | polyhaven.com/a/namaqualand_boulder_02 | large boulder |
| `sand_rocks_small_01.fbx` | polyhaven.com/a/sand_rocks_small_01 | scattered sandstone debris |

Downloaded at 2K texture resolution, FBX 7400 binary.

## Why these and not a Red Rocks scan

Two real photogrammetry scans of Red Rocks exist and both are gated:

- **Sketchfab** `Red Rocks Amphitheatre -RAWscan` (Spogna, CC BY-NC, 1.13M faces)
  and `Red Rocks, Colorado` (ml0w, CC BY, 3.0M faces) — both downloadable only
  with a Sketchfab account. The API returns 401 without auth.
- **DroneMapper** publishes the source dataset behind the ml0w model, but the
  S3 bucket is **Requester Pays**: anonymous GET returns 403 and needs AWS
  credentials to bill the download.

So these are generic CC0 rocks used as surface detail and scatter, not a
replacement for the surveyed monolith footprints.
