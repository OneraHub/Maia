import pytest
import numpy as np

import Converter.Internal as I
from maia.sids import Internal_ext as IE

from maia.sids.pytree    import CGNSLabelNotEqualError
from   maia.utils        import parse_yaml_cgns

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

  assert IE.getSubregionExtent(I.getNodeFromName(zone, 'UnLinkedZSR'), zone) == 'UnLinkedZSR'
  assert IE.getSubregionExtent(I.getNodeFromName(zone, 'BCLinkedZSR'), zone) == 'ZBC/BC2'
  assert IE.getSubregionExtent(I.getNodeFromName(zone, 'GCLinkedZSR'), zone) == 'ZGC2/GC1to1B'

  with pytest.raises(ValueError):
    IE.getSubregionExtent(I.getNodeFromName(zone, 'OrphelanZSR'), zone)
  with pytest.raises(CGNSLabelNotEqualError):
    IE.getSubregionExtent(I.getNodeFromName(zone, 'WrongZSR'), zone)

def test_getZoneDonorPath():
  jn1 = I.newGridConnectivity1to1(donorName='BaseXX/ZoneYY')
  jn2 = I.newGridConnectivity1to1(donorName='ZoneYY')
  assert IE.getZoneDonorPath('BaseXX', jn1) == 'BaseXX/ZoneYY'
  assert IE.getZoneDonorPath('BaseXX', jn2) == 'BaseXX/ZoneYY'

def test_enforceDonorAsPath():
  yt = """
  Base CGNSBase_t:
    ZoneA Zone_t:
      ZGC ZoneGridConnectivity_t:
        match1 GridConnectivity_t "ZoneA":
        match2 GridConnectivity_t "Base/ZoneA":
        match3 GridConnectivity_t "ZoneB":
    ZoneB Zone_t:
      ZGC1 ZoneGridConnectivity_t:
        match4 GridConnectivity_t "ZoneA":
  """
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  IE.enforceDonorAsPath(tree)
  assert I.getValue(I.getNodeFromName(tree, "match1")) == "Base/ZoneA"
  assert I.getValue(I.getNodeFromName(tree, "match2")) == "Base/ZoneA"
  assert I.getValue(I.getNodeFromName(tree, "match3")) == "Base/ZoneB"
  assert I.getValue(I.getNodeFromName(tree, "match4")) == "Base/ZoneA"

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
  connected_path = IE.find_connected_zones(tree)
  assert len(connected_path) == 3
  for zones in connected_path:
    if len(zones) == 1:
      assert zones == ['BaseB/Zone5']
    if len(zones) == 2:
      assert sorted(zones) == ['BaseA/Zone2', 'BaseA/Zone4']
    if len(zones) == 3:
      assert sorted(zones) == ['BaseA/Zone1', 'BaseA/Zone3', 'BaseB/Zone6']

def test_newDistribution():
  distri = IE.newDistribution()
  assert I.getName(distri) == ':CGNS#Distribution'
  assert I.getType(distri) == 'UserDefinedData_t'

  zone = I.newZone('zone')
  distri = IE.newDistribution(parent=zone)
  assert I.getNodeFromName(zone, ':CGNS#Distribution') is not None

  zone = I.newZone('zone')
  distri_arrays = {'Cell' : [0,15,30], 'Vertex' : [100,1000,1000]}
  distri = IE.newDistribution(distri_arrays, zone)
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Cell')[1] == [0,15,30]).all()
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Vertex')[1] == [100, 1000, 1000]).all()

  zone = I.newZone('zone')
  distri_arrayA = {'Cell' : [0,15,30]}
  distri_arrayB = {'Vertex' : [100,1000,1000]}
  distri = IE.newDistribution(distri_arrayA, parent=zone)
  distri = IE.newDistribution(distri_arrayB, parent=zone)
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Cell')[1] == [0,15,30]).all()
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Vertex')[1] == [100, 1000, 1000]).all()
  assert len(I.getNodesFromName(zone, ':CGNS#Distribution')) == 1

def test_newGlobalNumbering():
  gnum = IE.newGlobalNumbering()
  assert I.getName(gnum) == ':CGNS#GlobalNumbering'
  assert I.getType(gnum) == 'UserDefinedData_t'

  zone = I.newZone('zone')
  gnum = IE.newGlobalNumbering(parent=zone)
  assert I.getNodeFromName(zone, ':CGNS#GlobalNumbering') is not None

  zone = I.newZone('zone')
  gnum_arrays = {'Cell' : [4,21,1,2,8,12], 'Vertex' : None}
  gnum = IE.newGlobalNumbering(gnum_arrays, zone)
  assert (I.getNodeFromPath(zone, ':CGNS#GlobalNumbering/Cell')[1] == [4,21,1,2,8,12]).all()
  assert I.getNodeFromPath(zone, ':CGNS#GlobalNumbering/Vertex')[1] == None

def test_getDistribution():
  zone = I.newZone()
  distri_arrays = {'Cell' : [0,15,30], 'Vertex' : [100,1000,1000]}
  distri = IE.newDistribution(distri_arrays, zone)
  assert IE.getDistribution(zone) is distri
  assert (I.getVal(IE.getDistribution(zone, 'Cell')) == [0,15,30]).all()
  assert (I.getVal(IE.getDistribution(zone, 'Vertex')) == [100,1000,1000]).all()

def test_getGlobalNumbering():
  zone = I.newZone()
  gnum_arrays = {'Cell' : [4,21,1,2,8,12], 'Vertex' : None}
  gnum_node = IE.newGlobalNumbering(gnum_arrays, zone)
  assert IE.getGlobalNumbering(zone) is gnum_node
  assert (I.getVal(IE.getGlobalNumbering(zone, 'Cell')) == [4,21,1,2,8,12]).all()
  assert  I.getVal(IE.getGlobalNumbering(zone, 'Vertex')) == None

