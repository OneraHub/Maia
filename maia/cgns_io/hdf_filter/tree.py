import Converter.Internal as I

from maia.cgns_io.hdf_filter import elements as HEF

def create_tree_hdf_filter(dist_tree, hdf_filter):
  """
  """
  for base_tree in I.getNodesFromType1(dist_tree, 'CGNSBase_t'):
    for zone_tree in I.getNodesFromType1(base_tree, 'Zone_t'):
      zone_path = "/"+base_tree[0]+"/"+zone_tree[0]
      HEF.create_zone_std_elements_filter(zone_tree, zone_path, hdf_filter)