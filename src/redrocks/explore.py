"""Where the explorable things go, worked out from the map instead of by hand.

* **Shopping carts** wait at the highest drivable road node near the
  amphitheatre, pointed down the road. The cart route is the shortest road path
  from there to downtown Morrison (Bear Creek Avenue); its length and drop go in
  the manifest so the HUD can count down to town.
* **Sleds** wait at the top of the longest steady slope near the bowl that has
  no road, path or building on it: a fall line of 120-300 m whose grade stays
  between sledding and cliff.
* **Teleport spots** (number keys in game): the stage, the top of the bowl, the
  foot and summit of Creation Rock, the cart lot, the sled hill, the high ridge
  for long wingsuit flights, and Morrison.

Everything is in the terrain frame: metres east and north of the bbox centre,
elevations absolute (the Unreal scripts subtract the terrain minimum). The
Unreal side traces down onto whatever is really there (the ml0w scan's rocks
stand where the OSM footprints only approximate them), so a spot only has to be
in the right place, not at the exact height.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from .osm import OsmData, Way

# Downtown Morrison: Bear Creek Avenue between Stone Street and Market Street.
MORRISON_LAT = 39.65405
MORRISON_LON = -105.19115

# Road classes a cart can roll on.
CART_ROADS = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
    "residential", "service", "living_street",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
}
# The cart lot is the highest node on a road of these classes (not a lot aisle).
CART_START_ROADS = {"secondary", "tertiary", "unclassified", "residential"}
CART_START_RADIUS_M = 700.0

SLED_SEARCH_RADIUS_M = 1300.0
SLED_KEEP_OUT_M = 18.0
SLED_STEP_M = 6.0
SLED_MIN_GRADE = 0.10
SLED_MAX_GRADE = 0.5
SLED_IDEAL_GRADE = 0.28
SLED_MIN_RUN_M = 120.0
SLED_MAX_RUN_M = 320.0
SLED_AVOID_RADIUS_M = 260.0  # keep the hill out of the bowl itself


@dataclass
class Spot:
    key: str
    title: str
    tip: str
    east_m: float
    north_m: float
    elevation_m: float
    yaw_deg: float
    # How the Unreal script finds the floor: "ground" traces down from above,
    # "summit" traces down from high above (lands on the rock top).
    place: str = "ground"

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "tip": self.tip,
            "east_m": round(self.east_m, 2),
            "north_m": round(self.north_m, 2),
            "elevation_m": round(self.elevation_m, 2),
            "yaw_deg": round(self.yaw_deg, 1),
            "place": self.place,
        }


@dataclass
class Route:
    points: list[tuple[float, float, float]] = field(default_factory=list)  # (east, north, elevation)

    @property
    def length_m(self) -> float:
        return sum(
            math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(self.points, self.points[1:])
        )

    @property
    def drop_m(self) -> float:
        return self.points[0][2] - self.points[-1][2] if self.points else 0.0


def yaw_deg(dx: float, dy: float) -> float:
    """Unreal yaw (degrees) for a direction in east/north metres (X east, Y north)."""
    return math.degrees(math.atan2(dy, dx))


def road_graph(osm: OsmData, to_xy, classes=CART_ROADS):
    """Nodes (id -> (east, north)) and adjacency (id -> [(id, metres, way)])."""
    nodes: dict[int, tuple[float, float]] = {}
    adj: dict[int, list[tuple[int, float, Way]]] = {}
    for way in osm.ways:
        if way.tags.get("highway") not in classes or "tunnel" in way.tags:
            continue
        ids = [n for n in way.node_ids if n in osm.nodes]
        for n in ids:
            if n not in nodes:
                lat, lon = osm.nodes[n]
                nodes[n] = to_xy(lat, lon)
        # Carts ignore one-way signs: the graph is undirected.
        for a, b in zip(ids, ids[1:]):
            d = math.dist(nodes[a], nodes[b])
            adj.setdefault(a, []).append((b, d, way))
            adj.setdefault(b, []).append((a, d, way))
    return nodes, adj


def shortest_path(adj, start: int, goal: int) -> list[int]:
    """Dijkstra by length; [] when the goal is unreachable."""
    dist = {start: 0.0}
    prev: dict[int, int] = {}
    heap = [(0.0, start)]
    while heap:
        d, n = heapq.heappop(heap)
        if n == goal:
            break
        if d > dist.get(n, math.inf):
            continue
        for m, w, _ in adj.get(n, ()):
            nd = d + w
            if nd < dist.get(m, math.inf):
                dist[m] = nd
                prev[m] = n
                heapq.heappush(heap, (nd, m))
    if goal not in dist:
        return []
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1]


def nearest_node(nodes: dict[int, tuple[float, float]], x: float, y: float, among=None) -> int:
    keys = among if among is not None else nodes.keys()
    return min(keys, key=lambda n: (nodes[n][0] - x) ** 2 + (nodes[n][1] - y) ** 2)


def cart_route(osm: OsmData, to_xy, elevation, goal_xy: tuple[float, float]) -> Route:
    """From the highest road node near the bowl down to downtown Morrison."""
    nodes, adj = road_graph(osm, to_xy)
    if not nodes:
        return Route()
    starts = set()
    for way in osm.ways:
        if way.tags.get("highway") in CART_START_ROADS:
            for n in way.node_ids:
                if n in nodes and math.hypot(*nodes[n]) < CART_START_RADIUS_M and adj.get(n):
                    starts.add(n)
    if not starts:
        return Route()
    goal = nearest_node(nodes, *goal_xy)
    # Highest start that actually reaches town.
    for start in sorted(starts, key=lambda n: -elevation(*nodes[n])):
        path = shortest_path(adj, start, goal)
        if path:
            break
    else:
        return Route()
    return Route([(nodes[n][0], nodes[n][1], elevation(*nodes[n])) for n in path])


def densify(route: Route, step_m: float = 20.0) -> Route:
    """The route resampled every ``step_m`` metres (elevations interpolated)."""
    if len(route.points) < 2:
        return route
    out = [route.points[0]]
    carry = 0.0
    for a, b in zip(route.points, route.points[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        t = step_m - carry
        while t < seg:
            f = t / seg
            out.append(tuple(a[i] + (b[i] - a[i]) * f for i in range(3)))
            t += step_m
        carry = seg - (t - step_m)
    if out[-1] != route.points[-1]:
        out.append(route.points[-1])
    return Route(out)


def _keep_out_mask(osm: OsmData, to_xy, xs: np.ndarray, ys: np.ndarray, radius: float) -> np.ndarray:
    """True where a grid point is within ``radius`` of a road, path or building."""
    mask = np.zeros(xs.shape, dtype=bool)
    step = xs[0, 1] - xs[0, 0] if xs.shape[1] > 1 else 1.0
    for way in osm.ways:
        t = way.tags
        if not ("highway" in t or "building" in t or t.get("amenity") == "parking"):
            continue
        pts = [to_xy(*osm.nodes[n]) for n in way.node_ids if n in osm.nodes]
        for (ax, ay), (bx, by) in zip(pts, pts[1:] or pts):
            seg = max(math.hypot(bx - ax, by - ay), 1e-6)
            for k in range(int(seg / (step / 2)) + 1):
                f = min(k * (step / 2) / seg, 1.0)
                px, py = ax + (bx - ax) * f, ay + (by - ay) * f
                mask |= (xs - px) ** 2 + (ys - py) ** 2 < radius * radius
    return mask


def fall_line(elevation, x: float, y: float, step: float = SLED_STEP_M, max_len: float = SLED_MAX_RUN_M):
    """Walk steepest descent from (x, y); returns [(x, y, z), ...]."""
    pts = [(x, y, elevation(x, y))]
    for _ in range(int(max_len / step)):
        cx, cy, cz = pts[-1]
        best = None
        for k in range(16):
            a = 2 * math.pi * k / 16
            nx, ny = cx + step * math.cos(a), cy + step * math.sin(a)
            nz = elevation(nx, ny)
            if best is None or nz < best[2]:
                best = (nx, ny, nz)
        if best[2] >= cz - 0.05:
            break
        pts.append(best)
    return pts


def sled_hill(osm: OsmData, to_xy, elevation, half_extent: float):
    """(start (x, y, z), yaw, run m, drop m, fall line [(x, y, z)]) of the best sled run, or None."""
    r = min(SLED_SEARCH_RADIUS_M, half_extent - 50.0)
    grid = np.arange(-r, r + 1, 40.0)
    xs, ys = np.meshgrid(grid, grid)
    near = (xs**2 + ys**2 <= r * r) & (xs**2 + ys**2 >= SLED_AVOID_RADIUS_M**2)
    blocked = _keep_out_mask(osm, to_xy, xs, ys, SLED_KEEP_OUT_M + 20.0)
    best = None
    for x, y in zip(xs[near & ~blocked], ys[near & ~blocked]):
        line = fall_line(elevation, float(x), float(y))
        if len(line) < 3:
            continue
        run = 0.0
        ok = True
        for a, b in zip(line, line[1:]):
            d = math.hypot(b[0] - a[0], b[1] - a[1])
            grade = (a[2] - b[2]) / d
            if grade > SLED_MAX_GRADE:
                ok = False
                break
            if grade < SLED_MIN_GRADE:
                break
            run += d
        if not ok or run < SLED_MIN_RUN_M:
            continue
        end = _point_at(line, run)
        drop = line[0][2] - end[2]
        grade = drop / run
        # A long run at a fun grade beats a short steep one; nearer the bowl breaks ties.
        score = run * max(0.0, 1.0 - 2.5 * abs(grade - SLED_IDEAL_GRADE)) - 0.03 * math.hypot(x, y)
        if best is None or score > best[0]:
            dx, dy = line[1][0] - line[0][0], line[1][1] - line[0][1]
            trimmed = [p for p in line if math.hypot(p[0] - line[0][0], p[1] - line[0][1]) <= run + 1e-6]
            best = (score, line[0], yaw_deg(dx, dy), run, drop, trimmed + [end])
    if best is None:
        return None
    return best[1], best[2], best[3], best[4], best[5]


def _point_at(line, dist: float):
    acc = 0.0
    for a, b in zip(line, line[1:]):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        if acc + d >= dist:
            f = (dist - acc) / d
            return tuple(a[i] + (b[i] - a[i]) * f for i in range(3))
        acc += d
    return line[-1]


def high_ridge(elevation, half_extent: float, margin: float = 150.0):
    """The highest terrain point inside the world (away from its edge), and the yaw toward the bowl."""
    r = half_extent - margin
    grid = np.arange(-r, r + 1, 25.0)
    best = None
    for x in grid:
        for y in grid:
            z = elevation(float(x), float(y))
            if best is None or z > best[2]:
                best = (float(x), float(y), z)
    return best, yaw_deg(-best[0], -best[1])


def plan(osm: OsmData, to_xy, elevation, half_extent: float, spawn: dict | None,
         stage: dict | None, monoliths: list[dict],
         town: tuple[float, float] = (MORRISON_LAT, MORRISON_LON)) -> dict:
    """The manifest's ``explore`` block; ``town`` (lat, lon) is where the cart road ends."""
    spots: list[Spot] = []
    if spawn:
        spots.append(Spot("stage", "The stage", "Walk up the bowl, or turn around and climb the rocks.",
                          spawn["east_m"], spawn["north_m"], spawn["ground_elevation_m"], spawn["yaw_deg"]))
    if stage:
        ax, ay = stage["axis_east_north"]
        n = math.hypot(ax, ay) or 1.0
        ax, ay = ax / n, ay / n
        # The walkway is about 70 m up the bowl from the stage front, on its axis.
        tx = stage["front_edge_east_m"] + 72.0 * ax
        ty = stage["front_edge_north_m"] + 72.0 * ay
        spots.append(Spot("bowl_top", "Top of the bowl", "The whole amphitheatre below you.",
                          tx, ty, stage["walkway_elevation_m"], yaw_deg(-ax, -ay)))

    creation = next((m for m in monoliths if m["name"].startswith("Creation")), None)
    if creation:
        cx, cy = creation["centroid_east"], creation["centroid_north"]
        # The foot of the rock on the bowl side: from the bowl centre toward the rock.
        d = math.hypot(cx, cy) or 1.0
        ux, uy = cx / d, cy / d
        spots.append(Spot("climb", "Creation Rock: the climb",
                          "Walk into the rock face to climb it.  Shift climbs faster.",
                          cx - ux * 70.0, cy - uy * 70.0, creation["base_elevation_m"], yaw_deg(ux, uy)))
        spots.append(Spot("summit", "Creation Rock summit",
                          "300 ft up.  Run off the edge and press Space for the wingsuit.",
                          cx, cy, creation["top_elevation_m"], yaw_deg(-ux, -uy), place="summit"))

    goal = to_xy(*town)
    route = cart_route(osm, to_xy, elevation, goal)
    carts = None
    if len(route.points) >= 2:
        sx, sy, sz = route.points[0]
        nx_, ny_, _ = route.points[min(3, len(route.points) - 1)]
        heading = yaw_deg(nx_ - sx, ny_ - sy)
        spots.append(Spot("carts", "Shopping carts", "E to climb in.  The road runs all the way down to Morrison.",
                          sx, sy, sz, heading))
        fine = densify(route)
        carts = {
            "start": {"east_m": round(sx, 2), "north_m": round(sy, 2), "elevation_m": round(sz, 2),
                      "yaw_deg": round(heading, 1)},
            "count": 3,
            "destination": "Morrison",
            "route_length_m": round(route.length_m, 1),
            "route_drop_m": round(route.drop_m, 1),
            "route": [[round(p[0], 1), round(p[1], 1), round(p[2], 2)] for p in fine.points],
        }

    sled = sled_hill(osm, to_xy, elevation, half_extent)
    sleds = None
    if sled:
        (sx, sy, sz), yaw, run, drop, line = sled
        spots.append(Spot("sleds", "Sled hill", "Press N for snow, then E to hop on a sled.", sx, sy, sz, yaw))
        sleds = {"start": {"east_m": round(sx, 2), "north_m": round(sy, 2), "elevation_m": round(sz, 2),
                           "yaw_deg": round(yaw, 1)},
                 "count": 3, "run_m": round(run, 1), "drop_m": round(drop, 1),
                 "run": [[round(p[0], 1), round(p[1], 1), round(p[2], 2)] for p in line]}

    (rx, ry, rz), ryaw = high_ridge(elevation, half_extent)
    spots.append(Spot("ridge", "The high ridge",
                      f"{rz:.0f} m.  Jump off and wingsuit down to the rocks.", rx, ry, rz, ryaw))

    if carts:
        ex, ey, ez = route.points[-1]
        px, py, _ = route.points[-2]
        spots.append(Spot("morrison", "Morrison", "Bear Creek Avenue.  The carts end up here.",
                          ex, ey, ez, yaw_deg(ex - px, ey - py)))

    return {
        "spots": [s.as_dict() for s in spots],
        "carts": carts,
        "sleds": sleds,
        "morrison": {"lat": town[0], "lon": town[1],
                     "east_m": round(goal[0], 2), "north_m": round(goal[1], 2)},
    }
