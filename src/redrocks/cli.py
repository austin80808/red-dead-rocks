"""Command line entry point: build the Red Rocks terrain package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dem import BBox, fetch_dem, square_bbox
from .geo import LocalProjection
from .landscape import export_landscape
from .osm import fetch_osm

# Centre of the Red Rocks Amphitheatre seating bowl.
RED_ROCKS_LAT = 39.66550
RED_ROCKS_LON = -105.20550

# The park plus the surrounding hogback, so the venue sits in its real bowl.
DEFAULT_SIDE_M = 2400.0


def build(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bbox = square_bbox(RED_ROCKS_LAT, RED_ROCKS_LON, args.side_m)
    print(f"bbox  W{bbox.west:.5f} S{bbox.south:.5f} E{bbox.east:.5f} N{bbox.north:.5f}")

    print("fetching 3DEP DEM ...")
    dem = fetch_dem(bbox, size=args.dem_size)
    print(f"  {dem.shape[1]}x{dem.shape[0]}  {dem.min_elevation:.1f}..{dem.max_elevation:.1f} m")

    print("exporting landscape heightmap ...")
    landscape = export_landscape(dem, out_dir / "redrocks_heightmap.png", size=args.landscape_size)
    print(landscape.summary())

    print("fetching OSM ...")
    osm = fetch_osm(bbox)
    counts: dict[str, int] = {}
    for way in osm.ways:
        counts[way.kind()] = counts.get(way.kind(), 0) + 1
    print(f"  {len(osm.nodes)} nodes, {len(osm.ways)} ways")
    for kind, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {kind:10s} {count}")

    projection = LocalProjection(RED_ROCKS_LAT, RED_ROCKS_LON, dem.sample(RED_ROCKS_LAT, RED_ROCKS_LON))
    east, north, up = projection.to_local(RED_ROCKS_LAT, RED_ROCKS_LON)
    print(f"origin local coords  E{east:.2f} N{north:.2f} U{up:.2f}")

    manifest = {
        "venue": "Red Rocks Amphitheatre",
        "origin": {"lat": RED_ROCKS_LAT, "lon": RED_ROCKS_LON},
        "bbox": {
            "west": bbox.west,
            "south": bbox.south,
            "east": bbox.east,
            "north": bbox.north,
        },
        "landscape": {
            "heightmap": landscape.png_path.name,
            "size": landscape.size,
            "z_scale": round(landscape.z_scale, 6),
            "min_elevation_m": round(landscape.min_elevation, 3),
            "max_elevation_m": round(landscape.max_elevation, 3),
            "metres_per_pixel": round(landscape.metres_per_pixel, 4),
            "scale_xy_cm": round(landscape.metres_per_pixel * 100.0, 4),
        },
        "osm": {"nodes": len(osm.nodes), "ways": len(osm.ways), "by_kind": counts},
        "attribution": [
            "Elevation: USGS 3DEP, public domain",
            "Map data (c) OpenStreetMap contributors, ODbL 1.0",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {out_dir}/manifest.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="redrocks", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build_cmd = sub.add_parser("build", help="fetch terrain and write the Unreal package")
    build_cmd.add_argument("--out", default="out", help="output directory")
    build_cmd.add_argument("--side-m", type=float, default=DEFAULT_SIDE_M, help="bbox side in metres")
    build_cmd.add_argument("--dem-size", type=int, default=2017, help="DEM raster size to request")
    build_cmd.add_argument("--landscape-size", type=int, default=2017, help="landscape vertices per side")
    build_cmd.set_defaults(func=build)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
