import pytest
import Converter.Internal as I
from maia.utils.yaml   import parse_yaml_cgns

import maia.pytree as PT
from maia.pytree.sids import explore as EX

def test_getZoneDonorPath():
  jn1 = I.newGridConnectivity1to1(donorName='BaseXX/ZoneYY')
  jn2 = I.newGridConnectivity1to1(donorName='ZoneYY')
  assert EX.getZoneDonorPath('BaseXX', jn1) == 'BaseXX/ZoneYY'
  assert EX.getZoneDonorPath('BaseXX', jn2) == 'BaseXX/ZoneYY'


def test_getSubregionExtent():
  yt = """
Zone Zone_t:
  ZBC ZoneBC_t:
    BC BC_t:
    BC2 BC_t:
  ZGC1 ZoneGridConnectivity_t:
    GCA GridConnectivity_t:
    GCB GridConnectivity_t:
  ZGC2 ZoneGridConnectivity_t:
    GC1to1A GridConnectivity1to1_t:
    GC1to1B GridConnectivity1to1_t:
  UnLinkedZSR ZoneSubRegion_t:
    GridLocation GridLocation_t "Vertex":
    PointList IndexArray_t [[]]:
  BCLinkedZSR ZoneSubRegion_t:
    BCRegionName Descriptor_t "BC2":
  GCLinkedZSR ZoneSubRegion_t:
    GridConnectivityRegionName Descriptor_t "GC1to1B":
  OrphelanZSR ZoneSubRegion_t:
    BCRegionName Descriptor_t "BC9":
  WrongZSR WrongType_t:
    BCRegionName Descriptor_t "BC":
  """
  zone = parse_yaml_cgns.to_node(yt)

  assert EX.getSubregionExtent(I.getNodeFromName(zone, 'UnLinkedZSR'), zone) == 'UnLinkedZSR'
  assert EX.getSubregionExtent(I.getNodeFromName(zone, 'BCLinkedZSR'), zone) == 'ZBC/BC2'
  assert EX.getSubregionExtent(I.getNodeFromName(zone, 'GCLinkedZSR'), zone) == 'ZGC2/GC1to1B'

  with pytest.raises(ValueError):
    EX.getSubregionExtent(I.getNodeFromName(zone, 'OrphelanZSR'), zone)
  with pytest.raises(PT.CGNSLabelNotEqualError):
    EX.getSubregionExtent(I.getNodeFromName(zone, 'WrongZSR'), zone)

def test_find_connected_zones():
  yt = """
  BaseA CGNSBase_t:
    Zone1 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "Zone3":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
    Zone2 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "Zone4":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
    Zone3 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match1 GridConnectivity_t "BaseA/Zone1":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
        match2 GridConnectivity_t "BaseB/Zone6":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
    Zone4 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "Zone2":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
  BaseB CGNSBase_t:
    Zone5 Zone_t:
    Zone6 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "BaseA/Zone3":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
  """
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  connected_path = EX.find_connected_zones(tree)
  assert len(connected_path) == 3
  for zones in connected_path:
    if len(zones) == 1:
      assert zones == ['BaseB/Zone5']
    if len(zones) == 2:
      assert sorted(zones) == ['BaseA/Zone2', 'BaseA/Zone4']
    if len(zones) == 3:
      assert sorted(zones) == ['BaseA/Zone1', 'BaseA/Zone3', 'BaseB/Zone6']

