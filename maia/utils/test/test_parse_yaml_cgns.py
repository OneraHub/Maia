from maia.utils import parse_yaml_cgns
import Converter.Internal as I
import numpy as np


def test_empty_tree():
  yt = ""
  nodes = parse_yaml_cgns.to_nodes(yt)
  assert nodes == []

  node = parse_yaml_cgns.to_node(yt)
  assert node is None

  complete_t = parse_yaml_cgns.to_cgns_tree(yt)
  assert complete_t == ["CGNSTree",None,[['CGNSLibraryVersion', [3.1], [], 'CGNSLibraryVersion_t']],"CGNSTree_t"]


def test_simple_tree():
  yt = """
Base0 CGNSBase_t [3,3]:
  Zone0 Zone_t [[24],[6],[0]]:
  Zone1 Zone_t R8 [[4,3,2],[3,2,1],[0,0,0]]:
"""
  t = parse_yaml_cgns.to_cgns_tree(yt)
  bs = I.getNodesFromType1(t,"CGNSBase_t")
  assert len(bs) == 1
  assert I.getName(bs[0]) == "Base0"
  assert I.getType(bs[0]) == "CGNSBase_t"
  assert np.all(I.getValue(bs[0]) == [3,3])

  zs = I.getNodesFromType1(bs[0],"Zone_t")
  assert np.all(I.getValue(zs[0]) == [[24],[6],[0]])
  assert I.getChildren(zs[0]) == []
  assert I.getNodeFromName(t, 'Zone1')[1].dtype == np.float64

  yt = """
  Zone0 Zone_t [[24],[6],[0]]:
"""
  t = parse_yaml_cgns.to_cgns_tree(yt)
  bs = I.getNodesFromType1(t,"CGNSBase_t")
  assert len(bs) == 1
  assert I.getName(bs[0]) == "Base"
  assert I.getType(bs[0]) == "CGNSBase_t"


def test_multi_line_value():
  yt = """
CoordinateX DataArray_t:
  R8 : [ 0,1,2,3,
         0,1,2,3 ]
"""
  nodes = parse_yaml_cgns.to_nodes(yt)
  assert len(nodes) == 1

  node = nodes[0]
  assert I.getName(node) == "CoordinateX"
  assert I.getType(node) == "DataArray_t"
  assert (I.getValue(node) == np.array([0,1,2,3,0,1,2,3],dtype=np.float64)).all()
  assert I.getValue(node).dtype == np.float64
  assert I.getChildren(node) == []

  single_node = parse_yaml_cgns.to_node(yt)
  assert single_node[0] == node[0]
  assert (single_node[1] == node[1]).all()
  assert single_node[2] == node[2]
  assert single_node[3] == node[3]

