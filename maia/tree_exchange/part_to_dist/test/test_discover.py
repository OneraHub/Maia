from pytest_mpi_check._decorator import mark_mpi_test

import Converter.Internal as I

from maia.sids          import conventions as conv
from maia.utils         import parse_yaml_cgns
from maia.tree_exchange.part_to_dist import discover as disc

@mark_mpi_test(3)
class Test_discover_nodes_of_kind:
  pt = [\
  """
Zone.P0.N0 Zone_t:
  ZBC ZoneBC_t:
    BCA BC_t "wall":
      Family FamilyName_t "myfamily":
  """,
  """
Zone.P1.N0 Zone_t:
  ZGC ZoneGridConnectivity_t:
    match.0 GridConnectivity_t:
    match.1 GridConnectivity_t:
  """,
  """
Zone.P2.N0 Zone_t:
  ZBC ZoneBC_t:
    BCB BC_t "farfield":
      GridLocation GridLocation_t "FaceCenter":
Zone.P2.N1 Zone_t:
  ZBC ZoneBC_t:
    BCA BC_t "wall":
      Family FamilyName_t "myfamily":
  """]
  def test_simple(self, sub_comm):
    dist_zone = I.newZone('Zone')
    part_tree = parse_yaml_cgns.to_complete_pytree(self.pt[sub_comm.Get_rank()])

    disc.discover_nodes_of_kind(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm)
    assert (I.getType(I.getNodeFromPath(dist_zone, 'ZBC/BCA')) == "BC_t")
    assert (I.getType(I.getNodeFromPath(dist_zone, 'ZBC/BCB')) == "BC_t")

  def test_short(self, sub_comm):
    dist_node = I.createNode('SomeName', 'UserDefinedData_t')
    part_tree = parse_yaml_cgns.to_complete_pytree(self.pt[sub_comm.Get_rank()])
    part_nodes = [I.getNodeFromPath(zone, 'ZBC') for zone in I.getZones(part_tree)\
        if I.getNodeFromPath(zone, 'ZBC') is not None]
    disc.discover_nodes_of_kind(dist_node, part_nodes, 'BC_t', sub_comm)
    assert I.getNodeFromPath(dist_node, 'BCA') is not None
    assert I.getNodeFromPath(dist_node, 'BCB') is not None

  def test_getvalue(self, sub_comm):
    part_tree = parse_yaml_cgns.to_complete_pytree(self.pt[sub_comm.Get_rank()])
    for zbc in I.getNodesFromName(part_tree, 'ZBC'):
      I.setValue(zbc, 'test')

    dist_zone = I.newZone('Zone')
    disc.discover_nodes_of_kind(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm)
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC')) == 'test'
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC/BCA')) == None
    dist_zone = I.newZone('Zone')
    disc.discover_nodes_of_kind(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm, get_value='none')
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC')) == None
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC/BCA')) == None
    dist_zone = I.newZone('Zone')
    disc.discover_nodes_of_kind(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm, get_value='all')
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC')) == 'test'
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC/BCA')) == 'wall'

  def test_with_childs(self, sub_comm):
    dist_zone = I.newZone('Zone')
    part_tree = parse_yaml_cgns.to_complete_pytree(self.pt[sub_comm.Get_rank()])

    disc.discover_nodes_of_kind(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm,
        child_list=['FamilyName_t', 'GridLocation'])
    assert (I.getValue(I.getNodeFromPath(dist_zone, 'ZBC/BCA/Family')) == "myfamily")
    assert (I.getType(I.getNodeFromPath(dist_zone, 'ZBC/BCA/Family')) == "FamilyName_t")
    assert (I.getValue(I.getNodeFromPath(dist_zone, 'ZBC/BCB/GridLocation')) == "FaceCenter")
    assert (I.getType(I.getNodeFromPath(dist_zone, 'ZBC/BCB/GridLocation')) == "GridLocation_t")

  def test_with_rule(self, sub_comm):
    part_tree = parse_yaml_cgns.to_complete_pytree(self.pt[sub_comm.Get_rank()])

    #Exclude from node name
    dist_zone = I.newZone('Zone')
    disc.discover_nodes_of_kind(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm,
        child_list=['FamilyName_t', 'GridLocation'],
        skip_rule=lambda node: 'A' in I.getName(node))
    assert I.getNodeFromPath(dist_zone, 'ZBC/BCA') is None
    assert I.getNodeFromPath(dist_zone, 'ZBC/BCB') is not None

    #Exclude from node content
    dist_zone = I.newZone('Zone')
    disc.discover_nodes_of_kind(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm,
        skip_rule=lambda node: I.getNodeFromType1(node, 'FamilyName_t') is None)
    assert I.getNodeFromPath(dist_zone, 'ZBC/BCA') is not None
    assert I.getNodeFromPath(dist_zone, 'ZBC/BCB') is None

  def test_multiple(self, sub_comm):
    gc_path = 'ZoneGridConnectivity_t/GridConnectivity_t'
    part_tree = parse_yaml_cgns.to_complete_pytree(self.pt[sub_comm.Get_rank()])

    #Exclude from node name
    dist_zone = I.newZone('Zone')
    disc.discover_nodes_of_kind(dist_zone, I.getZones(part_tree), gc_path, sub_comm)
    assert I.getNodeFromPath(dist_zone, 'ZGC/match.0') is not None
    assert I.getNodeFromPath(dist_zone, 'ZGC/match.1') is not None

    dist_zone = I.newZone('Zone')
    disc.discover_nodes_of_kind(dist_zone, I.getZones(part_tree), gc_path, sub_comm,\
        merge_rule=lambda path : conv.get_split_prefix(path))
    assert I.getNodeFromPath(dist_zone, 'ZGC/match.0') is None
    assert I.getNodeFromPath(dist_zone, 'ZGC/match.1') is None
    assert I.getNodeFromPath(dist_zone, 'ZGC/match') is not None

  def test_zones(self, sub_comm):
    part_tree = I.newCGNSTree()
    if sub_comm.Get_rank() == 0:
      part_base = I.newCGNSBase('BaseA', parent=part_tree)
      I.newZone('Zone.P0.N0', parent=part_base)
    elif sub_comm.Get_rank() == 1:
      part_base = I.newCGNSBase('BaseB', parent=part_tree)
      I.newZone('Zone.withdot.P1.N0', parent=part_base)
    elif sub_comm.Get_rank() == 2:
      part_base = I.newCGNSBase('BaseA', parent=part_tree)
      I.newZone('Zone.P2.N0', parent=part_base)
      I.newZone('Zone.P2.N1', parent=part_base)

    dist_tree = I.newCGNSTree()
    disc.discover_nodes_of_kind(dist_tree, [part_tree], 'CGNSBase_t/Zone_t', sub_comm,\
        merge_rule=lambda zpath : conv.get_part_prefix(zpath))

    assert len(I.getZones(dist_tree)) == 2
    assert I.getNodeFromPath(dist_tree, 'BaseA/Zone') is not None
    assert I.getNodeFromPath(dist_tree, 'BaseB/Zone.withdot') is not None
