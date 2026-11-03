import pytest

from redrocks.osm import Way, parse_osm

SAMPLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="1" lat="39.6655" lon="-105.2062"/>
  <node id="2" lat="39.6650" lon="-105.2050"/>
  <node id="3" lat="39.6656" lon="-105.2053"/>
  <way id="10">
    <nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="1"/>
    <tag k="amenity" v="theatre"/>
    <tag k="name" v="Red Rocks Amphitheatre"/>
  </way>
  <way id="11">
    <nd ref="1"/><nd ref="2"/>
    <tag k="highway" v="service"/>
  </way>
  <way id="12">
    <nd ref="2"/><nd ref="3"/><nd ref="1"/>
    <tag k="amenity" v="parking"/>
  </way>
  <way id="13">
    <nd ref="1"/><nd ref="3"/><nd ref="1"/>
    <tag k="natural" v="bare_rock"/>
  </way>
</osm>
"""


@pytest.fixture
def osm():
    return parse_osm(SAMPLE)


def test_parses_nodes(osm):
    assert len(osm.nodes) == 3
    assert osm.nodes[1] == (39.6655, -105.2062)


def test_parses_ways_with_tags(osm):
    assert len(osm.ways) == 4
    venue = next(w for w in osm.ways if w.id == 10)
    assert venue.tags["name"] == "Red Rocks Amphitheatre"
    assert venue.node_ids == [1, 2, 3, 1]


def test_is_closed(osm):
    venue = next(w for w in osm.ways if w.id == 10)
    road = next(w for w in osm.ways if w.id == 11)
    assert venue.is_closed
    assert not road.is_closed


def test_way_coords_resolves_nodes(osm):
    venue = next(w for w in osm.ways if w.id == 10)
    coords = osm.way_coords(venue)
    assert coords == [(39.6655, -105.2062), (39.665, -105.205), (39.6656, -105.2053), (39.6655, -105.2062)]


def test_kind_classification(osm):
    kinds = {w.id: w.kind() for w in osm.ways}
    assert kinds[10] == "venue"
    assert kinds[11] == "road"
    assert kinds[12] == "parking"
    assert kinds[13] == "rock"


def test_by_kind(osm):
    assert len(osm.by_kind("road")) == 1
    assert len(osm.by_kind("venue")) == 1


def test_kind_priority_theatre_beats_building():
    way = Way(id=1, node_ids=[1, 2, 3, 1], tags={"amenity": "theatre", "building": "yes"})
    assert way.kind() == "venue"


def test_kind_priority_road_beats_landuse():
    way = Way(id=1, node_ids=[1, 2], tags={"highway": "path", "landuse": "grass"})
    assert way.kind() == "road"


def test_unknown_tags_are_other():
    way = Way(id=1, node_ids=[1, 2], tags={"random": "value"})
    assert way.kind() == "other"
