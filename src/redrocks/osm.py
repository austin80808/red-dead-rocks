"""Minimal OSM XML reader: nodes, ways, relations, tags, and closed-ring assembly.

Two sources: the main OSM API's ``map`` call for the park (``fetch_osm``), and
the Overpass API for a filtered query over a bigger area (``fetch_overpass``,
downtown Denver's buildings), whose answer is cached on disk.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import requests

OSM_API = "https://api.openstreetmap.org/api/0.6/map"
# Overpass's public instances throttle hard (the park uses the main API for that
# reason); each is tried in turn, with backoff, until one answers.
OVERPASS_APIS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
OVERPASS_ATTEMPTS = 4
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
class Relation:
    id: int
    members: list[tuple[str, int, str]]  # (type, ref, role)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class OsmData:
    nodes: dict[int, tuple[float, float]]  # id -> (lat, lon)
    ways: list[Way]
    relations: list[Relation] = field(default_factory=list)

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


def fetch_overpass(query: str, cache: Path | None = None, timeout: float = 300.0) -> OsmData:
    """Run an Overpass QL query that asks for ``[out:xml]`` and parse the answer.

    With ``cache`` the raw answer is kept in that file, and a later call with the
    file present reads it instead of the network, so re-runs work offline. An
    answer Overpass flags as failed (it reports timeouts and out-of-memory as a
    ``remark`` with HTTP 200) raises and is never cached.
    """
    if cache is not None and Path(cache).is_file():
        return parse_osm(Path(cache).read_bytes())
    last: Exception | None = None
    for attempt in range(OVERPASS_ATTEMPTS):
        for url in OVERPASS_APIS:
            try:
                response = requests.post(
                    url, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=timeout
                )
                response.raise_for_status()
                body = response.content
                _check_overpass(body)
            except (requests.RequestException, RuntimeError) as exc:
                last = exc
                continue
            if cache is not None:
                Path(cache).parent.mkdir(parents=True, exist_ok=True)
                Path(cache).write_bytes(body)
            return parse_osm(body)
        if attempt < OVERPASS_ATTEMPTS - 1:
            delay = 10.0 * 2 ** attempt
            print(f"  Overpass busy ({last}); retry {attempt + 1}/{OVERPASS_ATTEMPTS - 1} in {delay:.0f}s")
            time.sleep(delay)
    raise RuntimeError(f"Overpass query failed after {OVERPASS_ATTEMPTS} rounds: {last}")


def _check_overpass(body: bytes) -> None:
    """Raise if an Overpass answer is not a complete OSM document."""
    head = body[:2000].lstrip()
    if not head.startswith(b"<?xml") and not head.startswith(b"<osm"):
        raise RuntimeError(f"Overpass answered with something other than OSM XML: {head[:120]!r}")
    start = body.find(b"<remark>")
    while start != -1:
        end = body.find(b"</remark>", start)
        text = body[start + len(b"<remark>"):end].decode("utf-8", "replace").strip()
        if "error" in text.lower():
            raise RuntimeError(f"Overpass: {text}")
        start = body.find(b"<remark>", end)


def parse_osm(xml_bytes: bytes) -> OsmData:
    root = ET.fromstring(xml_bytes)

    nodes: dict[int, tuple[float, float]] = {}
    ways: list[Way] = []
    relations: list[Relation] = []

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
        elif element.tag == "relation":
            relations.append(
                Relation(
                    id=int(element.get("id")),
                    members=[(m.get("type"), int(m.get("ref")), m.get("role") or "")
                             for m in element.findall("member")],
                    tags={t.get("k"): t.get("v") for t in element.findall("tag")},
                )
            )

    return OsmData(nodes=nodes, ways=ways, relations=relations)
