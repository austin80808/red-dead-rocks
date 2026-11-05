"""Command line entry point: build the Red Rocks terrain package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dem import BBox, fetch_dem, square_bbox
from .explore import Route, carve_creek, creek_route, export_water_obj, plan as explore_plan
from .geo import LocalProjection
from .benches import row_polylines, terrain_base_elevation
from .landscape import export_landscape
from .monoliths import build_monoliths, monolith_to_obj
from .osm import fetch_osm
from .spawn import player_spawn
from .stage import DIMENSIONS, export_stage_obj
from .park import TerrainSurface as ParkSurface, export_park
from .seating import (
    SCAN_CLEARANCE_M, export_seating, seating_fade, seating_layout, seating_obstacles, trim_row_ends,
)
from .scan_blend import (
    FADE_M, FALLOFF_M, MARGIN_M, ScanGround, ScanHeights, blend_summary, conform_dem,
)
from .terrain_mesh import TerrainSurface, export_terrain_obj, summary as terrain_summary
from .venue import build_venue, summary as venue_summary

# Centre of the Red Rocks Amphitheatre seating bowl.
RED_ROCKS_LAT = 39.66550
RED_ROCKS_LON = -105.20550

# The park, the hogback around it and downtown Morrison (1.7 km east-southeast
# of the bowl), so the shopping-cart road runs all the way into town.
DEFAULT_SIDE_M = 3600.0
# 6.4 m terrain cells and a 1.19 m DEM across that box, as at 2.4 km.
DEFAULT_TERRAIN_GRID = 563
DEFAULT_DEM_SIZE = 3025

# The ml0w scan's floor and ground rasters (tools/scan_heights.py), relative to the checkout.
DEFAULT_SCAN_HEIGHTS = Path("assets") / "scan" / "scan_heights.npz"
DEFAULT_SCAN_GROUND = Path("assets") / "scan" / "scan_ground.npz"
# The scan's highest point per cell and its fade mask (tools/scan_heights.py):
# the seating is cut out of a copy of the mask, out/scan_fade.png.
DEFAULT_SCAN_TOP = Path("assets") / "scan" / "scan_top.npz"
DEFAULT_SCAN_FADE = Path("assets") / "scan" / "scan_fade.png"


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

    # Benches, stage and park share the terrain mesh's Z datum, so use the same
    # grid it was resampled at; a different grid shifts the minimum elevation.
    grid = args.terrain_grid or DEFAULT_TERRAIN_GRID
    base = terrain_base_elevation(dem, grid=grid)
    lines = row_polylines(venue, osm.way_coords(venue_way), projection) if venue is not None else None

    # Issue #17: pull the terrain under the ml0w scan where the scan covers it.
    # The monoliths and landscape heightmap stay on the DEM.
    ground_dem, scan_blend = _scan_blended_dem(args, dem, base, grid)
    # The benches, stage and spawn stand on the scan's ground where it has
    # one; the walkable terrain mesh (as exported) everywhere else.
    walkable = TerrainSurface(ground_dem, grid)
    # Cut Bear Creek's bed into the ground so its water shows the whole way.
    frame = ParkSurface(ground_dem, grid)
    creek = creek_route(osm, frame.to_xy, walkable, frame.half_x, level="floor")
    if len(creek.points) > 1:
        carved = carve_creek(ground_dem, creek, frame.half_x, frame.half_y, floor_m=base)
        if abs(terrain_base_elevation(carved, grid=grid) - base) < 1e-3:
            ground_dem = carved
            walkable = TerrainSurface(ground_dem, grid)
            print(f"carved {creek.length_m:.0f} m of Bear Creek into the terrain")
        else:
            print("skipping the creek carve: it would move the terrain datum")
    seat_ground, ground_info = _seat_ground(args, walkable)

    terrain = None
    if args.terrain_grid:
        print("\nexporting terrain mesh ...")
        terrain = export_terrain_obj(ground_dem, out_dir / "RedRocks_Terrain.obj", grid=args.terrain_grid)
        print(terrain_summary(terrain))
    else:
        print("skipping terrain mesh (--terrain-grid 0)")

    seating = None
    layout = None
    scan_cut = None
    stage = None
    spawn = None
    for stale in ("scan_fade.png", "scan_fade.json"):
        (out_dir / stale).unlink(missing_ok=True)
    if venue is not None:
        print("\nbuilding the terraced seating ...")
        # On the scan, treads clear its surface (its chunks keep collision).
        on_scan = ground_info != "terrain mesh"
        # Rows end at their aisles, clear of the stage towers and the rock walls.
        lines, trim_info = _trim_rows(args, osm, venue_way, projection, lines, seat_ground if on_scan else None)
        if trim_info["rows"]:
            print(f"  {trim_info['rows']} rows pulled in at their ends ({trim_info['metres']:.0f} m in all) "
                  f"so their aisles clear {trim_info['buildings']} buildings and the scan's rock")
        layout = seating_layout(
            venue.rows, lines, ground=seat_ground, floor=walkable,
            clearance_m=SCAN_CLEARANCE_M if on_scan else 0.0,
        )
        seating = export_seating(layout, out_dir, base, ground=seat_ground, floor=walkable)
        i = layout.info
        print(f"  {seating.terraces} rows + {seating.front_steps} front steps, {seating.brackets} brackets; "
              f"rises {i['rise_min_m']:.2f}..{i['rise_max_m']:.2f} m, {i['total_rise_m']:.1f} m in all"
              + (f"; rows {i['rows_ignored']} left out of the fit" if i["rows_ignored"] else ""))
        for name, path in seating.paths.items():
            print(f"  {seating.triangles[name]:,} triangles -> {path.name}")
        if on_scan:
            scan_cut = _cut_scan_fade(args, layout, out_dir)

        print("\nblocking out the stage ...")
        booth_way, booth_plan = _sound_booth(osm, venue_way, projection)
        stage = export_stage_obj(venue.rows, lines, out_dir / "Stage.obj", base, ground=seat_ground,
                                 levels=layout.levels, booth=booth_plan)
        print(f"  {len(stage.parts)} parts, {stage.triangles} triangles -> {stage.path.name}")
        print(f"  {stage.handrail_triangles} triangles -> {stage.handrail_path.name}")
        if stage.booth_path is not None:
            print(f"  sound booth on OSM way {booth_way.id}, deck {stage.booth_deck_elevation_m:.2f} m "
                  f"-> {stage.booth_path.name}")

        spawn = player_spawn(
            stage.front_edge_east, stage.front_edge_north, stage.axis,
            stage.deck_top_elevation_m, seat_ground,
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
    park = export_park(osm, ground_dem, out_dir, grid=grid)
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
    if scan_blend is not None:
        manifest["scan_blend"] = scan_blend
    if venue is not None:
        manifest["venue"] = {
            "rows": len(venue.rows),
            "seats": venue.total_seats,
            "rise_m": round(venue.rise_m, 3),
            "rake_degrees": round(venue.rake_degrees, 2),
            "row_1_elevation_m": round(venue.rows[0].elevation_m, 3),
        }
    if seating is not None:
        manifest["benches"] = {
            "obj": seating.paths["Benches"].name,
            "planks_obj": seating.paths["Bench_Planks"].name,
            "brackets_obj": seating.paths["Bench_Brackets"].name if "Bench_Brackets" in seating.paths else None,
            "rows": seating.terraces,
            "front_steps": seating.front_steps,
            "brackets": seating.brackets,
            "triangles": seating.triangles["Benches"],
            "planks_triangles": seating.triangles["Bench_Planks"],
            "brackets_triangles": seating.triangles.get("Bench_Brackets", 0),
            "base_elevation_m": round(seating.base_elevation_m, 3),
            "ground": ground_info,
            "clearance_m": layout.info["clearance_m"],
            "rows_ignored": layout.info["rows_ignored"],
            "rise_m": {k: layout.info[k] for k in ("rise_min_m", "rise_max_m", "rise_mean_m", "total_rise_m")},
            "tread_levels_m": [round(v, 3) for v in layout.levels],
            "rows_trimmed": trim_info["rows"],
            "trimmed_m": round(trim_info["metres"], 1),
        }
    if scan_cut is not None:
        manifest["scan_cut"] = scan_cut
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
        manifest["handrail"] = {"obj": stage.handrail_path.name, "triangles": stage.handrail_triangles}
        if stage.stairs_path is not None:
            manifest["stairs"] = {"obj": stage.stairs_path.name, "triangles": stage.stairs_triangles,
                                  "parts": len(stage.stairs_parts)}
        if stage.booth_path is not None:
            manifest["booth"] = {
                "obj": stage.booth_path.name,
                "osm_way": booth_way.id,
                "deck_elevation_m": round(stage.booth_deck_elevation_m, 3),
                "east_m": round(float(np.mean(booth_plan[:, 0])), 3),
                "north_m": round(float(np.mean(booth_plan[:, 1])), 3),
            }
        manifest["stage"] = {
            "obj": stage.path.name,
            "parts": {name: {"first_triangle": a, "end_triangle": b} for name, (a, b) in stage.parts.items()},
            "triangles": stage.triangles,
            "base_elevation_m": round(stage.base_elevation_m, 3),
            "deck_top_elevation_m": round(stage.deck_top_elevation_m, 3),
            "walkway_elevation_m": round(stage.walkway_elevation_m, 3),
            "walkway_east_m": round(stage.walkway_centre[0], 3),
            "walkway_north_m": round(stage.walkway_centre[1], 3),
            "aisles": {side: [list(p) for p in steps] for side, steps in stage.aisles.items()},
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
    print("\nplanning carts, sleds and teleport spots ...")
    park_frame = ParkSurface(ground_dem, grid)
    explore = explore_plan(
        osm, park_frame.to_xy, walkable, park_frame.half_x,
        manifest.get("spawn"), manifest.get("stage"), manifest["monoliths"],
    )
    manifest["explore"] = explore
    if explore.get("tubes"):
        t = explore["tubes"]
        water = Route([tuple(p) for p in t["route"]])
        tris = export_water_obj(water, walkable.base_elevation_m, out_dir / "Water.obj", t["width_m"])
        manifest.setdefault("park", {})["water"] = {"obj": "Water.obj", "features": 1, "triangles": tris}
        print(f"  {t['name']}: {t['route_length_m']:.0f} m of creek, dropping {t['route_drop_m']:.0f} m "
              f"-> Water.obj ({tris} triangles)")
    if explore["carts"]:
        c = explore["carts"]
        print(f"  carts: {c['route_length_m']:.0f} m of road to {c['destination']}, dropping {c['route_drop_m']:.0f} m")
    if explore["sleds"]:
        s = explore["sleds"]
        print(f"  sled hill: {s['run_m']:.0f} m run, {s['drop_m']:.0f} m drop")
    print(f"  {len(explore['spots'])} spots: " + ", ".join(sp["title"] for sp in explore["spots"]))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {out_dir}/manifest.json")
    return 0


def _trim_rows(args, osm, venue_way, projection, lines, seat_ground):
    """(row lines with their ends pulled in clear of the aisles' obstacles, summary)."""
    from shapely.geometry import LineString, Polygon

    bowl = Polygon([projection.to_local(*osm.nodes[n])[:2] for n in venue_way.node_ids])
    near = bowl.buffer(15.0)
    buildings = []
    for way in osm.ways:
        # Roofs in the bowl (the sound booth, the stage roof) are not in the aisles' way.
        if "building" not in way.tags or way.tags.get("building") == "roof":
            continue
        try:
            poly = Polygon([projection.to_local(*osm.nodes[n])[:2] for n in way.node_ids])
        except (KeyError, ValueError):
            continue
        if poly.is_valid and poly.intersects(near):
            buildings.append(poly)
    top = ground = None
    top_path = Path(getattr(args, "scan_top", "") or "")
    if seat_ground is not None and top_path.name and top_path.is_file():
        top = ScanHeights.load(top_path)
        ground = getattr(seat_ground, "ground", None)
    obstacles = seating_obstacles(top, ground, buildings, within=near)
    trimmed = trim_row_ends(lines, obstacles)
    cut = [LineString(a).length - LineString(b).length for a, b in zip(lines, trimmed)]
    return trimmed, {"rows": sum(1 for c in cut if c > 1e-6), "metres": float(sum(cut)), "buildings": len(buildings)}


def _sound_booth(osm, venue_way, projection):
    """(OSM way, 4-corner footprint in local metres) of the sound booth's roof, or (None, None).

    The booth is the ``building=roof`` way inside the amphitheatre's footprint
    (way 229359709 at the real venue).
    """
    from shapely.geometry import Polygon

    def plan(way):
        try:
            return Polygon([projection.to_local(*osm.nodes[n])[:2] for n in way.node_ids])
        except (KeyError, ValueError):
            return None

    bowl = plan(venue_way)
    if bowl is None or not bowl.is_valid:
        return None, None
    for way in osm.ways:
        if way.tags.get("building") != "roof":
            continue
        roof = plan(way)
        if roof is not None and roof.is_valid and 4.0 < roof.area < 200.0 and bowl.contains(roof.centroid):
            rect = np.array(roof.minimum_rotated_rectangle.exterior.coords)[:4]
            return way, rect
    return None, None


def _cut_scan_fade(args, layout, out_dir: Path) -> dict | None:
    """out/scan_fade.png and .json: the scan's fade mask with the seating cut out."""
    from PIL import Image

    png = Path(getattr(args, "scan_fade", "") or "")
    if not png.name or not png.is_file() or not png.with_suffix(".json").is_file():
        print(f"no scan fade mask at {png}; the scan is not cut out of the seating")
        return None
    place = json.loads(png.with_suffix(".json").read_text())
    alpha = np.asarray(Image.open(png).convert("L"), dtype=float) / 255.0
    top_path = Path(getattr(args, "scan_top", "") or "")
    top = ScanHeights.load(top_path) if top_path.name and top_path.is_file() else None
    cut, meta = seating_fade(alpha, place, layout, top=top)
    img = np.round(cut * 255.0).astype(np.uint8)
    Image.fromarray(np.dstack([img, img, img]), "RGB").save(out_dir / "scan_fade.png")
    meta["about"] = (f"{place.get('about', 'ml0w scan opacity')}; seating cut out by redrocks build "
                     f"from {png.name}" + (f" and {top_path.name}" if top is not None else ""))
    (out_dir / "scan_fade.json").write_text(json.dumps(meta, indent=2) + "\n")
    info = meta["seating_cut"]
    print(f"  scan cut out over {info['cut_m2']:,.0f} m^2 of seating, {info['kept_tall_m2']:,.0f} m^2 of "
          f"tall scan kept -> scan_fade.png ({meta['texels'][0]}x{meta['texels'][1]})")
    return info


def _seat_ground(args, walkable):
    """(ground the benches, stage and spawn stand on, what it is for the manifest)."""
    path = getattr(args, "scan_ground", None)
    if not path or not Path(path).is_file():
        print(f"no scan ground at {path}; benches and stage stand on the terrain mesh")
        return walkable, "terrain mesh"
    print(f"benches, stage and spawn stand on the ml0w scan's ground ({path})")
    return ScanGround(ScanHeights.load(Path(path)), fallback=walkable), f"ml0w scan ({Path(path).name})"


def _scan_blended_dem(args, dem, base: float, grid: int):
    """(dem for the terrain and park, manifest block or None)."""
    path = getattr(args, "scan_heights", None)
    if not path or not Path(path).is_file():
        print(f"\nno scan heights at {path}; terrain stays on the DEM (tools/scan_heights.py writes them)")
        return dem, None
    print(f"\nblending the terrain into the ml0w scan ({path}) ...")
    scan = ScanHeights.load(Path(path))
    blended = conform_dem(dem, scan, grid=grid)
    # The Z datum of every OBJ is the 377-grid minimum; the blend must not move it.
    if abs(terrain_base_elevation(blended, grid=grid) - base) > 1e-3:
        raise RuntimeError("scan blend moved the terrain minimum; the OBJs would disagree on Z")
    info = {
        "scan_heights": Path(path).name,
        "margin_m": MARGIN_M,
        "falloff_m": FALLOFF_M,
        "fade_m": FADE_M,
        **blend_summary(dem, blended),
    }
    print(f"  changed {info['changed_area_m2']:,.0f} m^2; " + ", ".join(
        f"{k} {v}" for k, v in info.items() if k.startswith(("raised", "lowered", "delta"))))
    return blended, info


def vegetation(args: argparse.Namespace) -> int:
    """Grow the tree and shrub models and place them on a built terrain."""
    from . import vegetation as veg

    out_dir = Path(args.out)
    terrain = out_dir / "RedRocks_Terrain.obj"
    if not terrain.is_file():
        print(f"no {terrain}; run `redrocks build --out {out_dir}` first")
        return 1
    info = veg.export(out_dir / "vegetation", terrain_obj=terrain, avoid_dir=out_dir, seed=args.seed)
    tris = sum(p["triangles"] for m in info["models"] for p in m["parts"].values())
    print(f"{len(info['models'])} models, {tris:,} triangles, {len(info['atlases'])} leaf atlases")
    for species, count in info["counts"].items():
        print(f"  {species:10s} {count:6d}")
    print(f"wrote {out_dir / 'vegetation'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="redrocks", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build_cmd = sub.add_parser("build", help="fetch terrain and write the Unreal package")
    build_cmd.add_argument("--out", default="out", help="output directory")
    build_cmd.add_argument("--side-m", type=float, default=DEFAULT_SIDE_M, help="bbox side in metres")
    build_cmd.add_argument("--dem-size", type=int, default=DEFAULT_DEM_SIZE, help="DEM raster size to request")
    build_cmd.add_argument("--landscape-size", type=int, default=2017, help="landscape vertices per side")
    build_cmd.add_argument(
        "--terrain-grid",
        type=int,
        default=DEFAULT_TERRAIN_GRID,
        help="terrain mesh vertices per side (0 skips RedRocks_Terrain.obj)",
    )
    build_cmd.add_argument(
        "--scan-heights",
        default=str(DEFAULT_SCAN_HEIGHTS),
        help="ml0w scan height raster from tools/scan_heights.py; the terrain is "
        "conformed to it (issue #17). Pass '' to keep the bare DEM.",
    )
    build_cmd.add_argument(
        "--scan-ground",
        default=str(DEFAULT_SCAN_GROUND),
        help="ml0w scan ground raster from tools/scan_heights.py; the benches, stage "
        "and player spawn stand on it. Pass '' to put them on the terrain mesh.",
    )
    build_cmd.add_argument(
        "--scan-top",
        default=str(DEFAULT_SCAN_TOP),
        help="ml0w scan top raster from tools/scan_heights.py; where the scan stands "
        "tall in the seating it stays visible. Pass '' to cut the seating out regardless.",
    )
    build_cmd.add_argument(
        "--scan-fade",
        default=str(DEFAULT_SCAN_FADE),
        help="ml0w scan fade mask (with its .json) from tools/scan_heights.py; out/ gets a "
        "copy with the seating cut out. Pass '' for no cut.",
    )
    build_cmd.set_defaults(func=build)

    veg_cmd = sub.add_parser(
        "vegetation", help="grow junipers, ponderosa and scrub and place them on --out's terrain"
    )
    veg_cmd.add_argument("--out", default="out", help="directory holding RedRocks_Terrain.obj")
    veg_cmd.add_argument("--seed", type=int, default=1941, help="placement seed")
    veg_cmd.set_defaults(func=vegetation)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
