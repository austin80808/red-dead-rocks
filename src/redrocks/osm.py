"""Minimal OSM XML reader: nodes, ways, tags, and closed-ring assembly."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import requests

OSM_API = "https://api.openstreetmap.org/api/0.6/map"
USER_AGENT = "redrocks-ue/0.1 (terrain prototype)"


@dataclass
class Way:
    id: int
    node_ids: list[int]
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def is_closed(self) -> bool:
        return len(self.node_ids) > 2 and self.node_ids[0] == self.node_ids[-1]

    def kind(self) -> str:
        """A coarse gameplay class derived from the tags."""
        t = self.tags
        if "highway" in t:
            return "road"
        if t.get("amenity") == "parking":
            return "parking"
        if t.get("amenity") == "theatre" or t.get("leisure") == "stadium":
            return "venue"
        if "building" in t:
            return "building"
        if t.get("natural") in ("bare_rock", "stone", "scree", "cliff"):
            return "rock"
        if t.get("natural") == "tree" or t.get("landuse") == "forest":
            return "tree"
        if t.get("landuse") in ("grass", "meadow", "recreation_ground"):
            return "grass"
        if t.get("barrier"):
            return "barrier"
        return "other"


@dataclass
class OsmData:
    nodes: dict[int, tuple[float, float]]  # id -> (lat, lon)
    ways: list[Way]

    def way_coords(self, way: Way) -> list[tuple[float, float]]:
        return [self.nodes[n] for n in way.node_ids if n in self.nodes]

    def by_kind(self, kind: str) -> list[Way]:
        return [w for w in self.ways if w.kind() == kind]


def fetch_osm(bbox, timeout: float = 180.0) -> OsmData:
    """Download raw OSM data for a bbox. Uses the main API, not Overpass."""
    params = {"bbox": f"{bbox.west},{bbox.south},{bbox.east},{bbox.north}"}
    response = requests.get(
        OSM_API,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_osm(response.content)


def parse_osm(xml_bytes: bytes) -> OsmData:
    root = ET.fromstring(xml_bytes)

    nodes: dict[int, tuple[float, float]] = {}
    ways: list[Way] = []

    for element in root:
        if element.tag == "node":
            nodes[int(element.get("id"))] = (
                float(element.get("lat")),
                float(element.get("lon")),
            )
        elif element.tag == "way":
            ways.append(
                Way(
                    id=int(element.get("id")),
                    node_ids=[int(nd.get("ref")) for nd in element.findall("nd")],
                    tags={t.get("k"): t.get("v") for t in element.findall("tag")},
                )
            )

    return OsmData(nodes=nodes, ways=ways)
