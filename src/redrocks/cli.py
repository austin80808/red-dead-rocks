"""Command line entry point: build the Red Rocks terrain package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dem import BBox, fetch_dem, square_bbox
from .geo import LocalProjection
from .benches import export_benches_obj, row_polylines, terrain_base_elevation
from .landscape import export_landscape
from .monoliths import build_monoliths, monolith_to_obj
from .osm import fetch_osm
from .spawn import player_spawn
from .stage import DIMENSIONS, export_stage_obj
from .park import export_park
from .terrain_mesh import TerrainSurface, export_terrain_obj, summary as terrain_summary
from .venue import build_venue, summary as venue_summary

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

    terrain = None
    if args.terrain_grid:
        print("exporting terrain mesh ...")
        terrain = export_terrain_obj(dem, out_dir / "RedRocks_Terrain.obj", grid=args.terrain_grid)
        print(terrain_summary(terrain))
    else:
        print("skipping terrain mesh (--terrain-grid 0)")

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

    print("\ngenerating seating bowl ...")
    venue_way = next((w for w in osm.ways if w.tags.get("amenity") == "theatre"), None)
    venue = None
    if venue_way is None:
        print("  no amenity=theatre way found; skipping venue")
    else:
        venue = build_venue(
            osm.way_coords(venue_way), dem, projection, RED_ROCKS_LAT, RED_ROCKS_LON
        )
        print(venue_summary(venue))

    benches = None
    stage = None
    spawn = None
    if venue is not None:
        print("\nbuilding benches ...")
        # Benches share the terrain mesh's Z datum, so use the same grid it was
        # resampled at; a different grid shifts the minimum elevation.
        grid = args.terrain_grid or 377
        base = terrain_base_elevation(dem, grid=grid)
        lines = row_polylines(venue, osm.way_coords(venue_way), projection)
        # Sit each bench on the terrain mesh as exported, not the finer DEM.
        benches = export_benches_obj(
            venue.rows, lines, out_dir / "Benches.obj", base, ground=TerrainSurface(dem, grid)
        )
        print(f"  {benches.benches} benches, {benches.triangles} triangles -> {benches.path.name}")

        print("\nblocking out the stage ...")
        stage = export_stage_obj(venue.rows, lines, out_dir / "Stage.obj", base)
        print(f"  {len(stage.parts)} parts, {stage.triangles} triangles -> {stage.path.name}")

        spawn = player_spawn(
            stage.front_edge_east, stage.front_edge_north, stage.axis,
            stage.deck_top_elevation_m, TerrainSurface(dem, grid),
        )
        print(
            f"  player spawn E{spawn.east_m:.1f} N{spawn.north_m:.1f} on {spawn.ground_m:.2f} m "
            f"(deck {spawn.deck_top_m:.2f}, terrain {spawn.terrain_m:.2f}), yaw {spawn.yaw_deg:.1f}"
        )

    print("\nextruding monoliths ...")
    monoliths = build_monoliths(osm, dem, projection, RED_ROCKS_LAT, RED_ROCKS_LON)
    for rock in monoliths:
        print(rock.summary())
        obj_path = monolith_to_obj(rock, out_dir / f"{rock.name.split(' (')[0].replace(' ', '_')}.obj")
        print(f"  -> {obj_path.name}")

    print("\nbuilding roads, paths and parking ...")
    park = export_park(osm, dem, out_dir, grid=args.terrain_grid or 377)
    for mesh in park.values():
        print(f"  {mesh.features} {mesh.name.lower()}, {mesh.triangles:,} triangles -> {mesh.path.name}")

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
        "monoliths": [
            {
                "name": r.name,
                "height_m": round(r.height_m, 3),
                "base_elevation_m": round(r.base_elevation_m, 3),
                "top_elevation_m": round(r.top_elevation_m, 3),
                "footprint_m2": round(r.footprint.area, 2),
                "centroid_east": round(r.footprint.centroid.x, 2),
                "centroid_north": round(r.footprint.centroid.y, 2),
                "strata_dip_degrees": r.strata_dip_degrees,
                "obj": f"{r.name.split(' (')[0].replace(' ', '_')}.obj",
            }
            for r in monoliths
        ],
        "attribution": [
            "Elevation: USGS 3DEP, public domain",
            "Map data (c) OpenStreetMap contributors, ODbL 1.0",
        ],
    }
    if terrain is not None:
        manifest["terrain_mesh"] = {
            "obj": terrain.path.name,
            "grid": terrain.grid,
            "triangles": terrain.triangles,
            "width_m": round(terrain.width_m, 3),
            "height_m": round(terrain.height_m, 3),
            "min_elevation_m": round(terrain.min_elevation_m, 3),
            "max_elevation_m": round(terrain.max_elevation_m, 3),
        }
    if venue is not None:
        manifest["venue"] = {
            "rows": len(venue.rows),
            "seats": venue.total_seats,
            "rise_m": round(venue.rise_m, 3),
            "rake_degrees": round(venue.rake_degrees, 2),
            "row_1_elevation_m": round(venue.rows[0].elevation_m, 3),
        }
    if benches is not None:
        manifest["benches"] = {
            "obj": benches.path.name,
            "benches": benches.benches,
            "triangles": benches.triangles,
            "base_elevation_m": round(benches.base_elevation_m, 3),
        }
    if spawn is not None:
        manifest["spawn"] = {
            "east_m": round(spawn.east_m, 3),
            "north_m": round(spawn.north_m, 3),
            "ground_elevation_m": round(spawn.ground_m, 3),
            "deck_top_elevation_m": round(spawn.deck_top_m, 3),
            "terrain_elevation_m": round(spawn.terrain_m, 3),
            "yaw_deg": round(spawn.yaw_deg, 3),
        }
    if stage is not None:
        manifest["stage"] = {
            "obj": stage.path.name,
            "parts": {name: {"first_triangle": a, "end_triangle": b} for name, (a, b) in stage.parts.items()},
            "triangles": stage.triangles,
            "base_elevation_m": round(stage.base_elevation_m, 3),
            "deck_top_elevation_m": round(stage.deck_top_elevation_m, 3),
            "walkway_elevation_m": round(stage.walkway_elevation_m, 3),
            "front_edge_east_m": round(stage.front_edge_east, 3),
            "front_edge_north_m": round(stage.front_edge_north, 3),
            "axis_east_north": [round(stage.axis[0], 6), round(stage.axis[1], 6)],
            "dimensions": [
                {"name": d.name, "value_m": round(d.value_m, 4), "kind": d.kind, "source": d.source}
                for d in DIMENSIONS
            ],
        }
    if park:
        manifest["park"] = {
            key: {
                "obj": m.path.name,
                "features": m.features,
                "vertices": m.vertices,
                "triangles": m.triangles,
            }
            for key, m in park.items()
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
    build_cmd.add_argument(
        "--terrain-grid",
        type=int,
        default=377,
        help="terrain mesh vertices per side (0 skips RedRocks_Terrain.obj)",
    )
    build_cmd.set_defaults(func=build)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
