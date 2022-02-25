import pytest
from   pytest_mpi_check._decorator import mark_mpi_test
import os

import Converter.Internal     as I

from maia.cgns_io import cgns_io_tree as IOT

from maia.sids  import pytree     as PT
from maia.utils import test_utils as TU

from maia.transform import merge

ref_dir = os.path.join(os.path.dirname(__file__), 'references')

@mark_mpi_test([2])
def test_merge_all(sub_comm, write_output):

  mesh_file = os.path.join(TU.mesh_dir, 'U_Naca0012_multizone.yaml')

  dist_tree = IOT.file_to_dist_tree(mesh_file, sub_comm)

  # Get the path of all the zones to merge
  zones_path = [f"{I.getName(base)}/{I.getName(zone)}" for base,zone in \
      PT.iter_nodes_from_predicates(dist_tree, ['CGNSBase_t', 'Zone_t'], ancestors=True)]

  # Merge the zones indicated in zones_path
  merge.merge_zones(dist_tree, zones_path, sub_comm)
  merged_1 = I.copyTree(dist_tree) # Do copy for further comparaisons
  assert len(I.getZones(merged_1)) == 1
  assert I.getNodeFromPath(merged_1, 'BaseA/MergedZone') is not None #By default take same name than current base
  assert len(I.getNodesFromType(merged_1, 'ZoneGridConnectivity_t')) == 0

  # Note that by default, the BC (and more generally, subsets) of same name (across blocks) are concatenated :
  # thus, there is 4 BCs in the merged block
  assert len(I.getNodesFromType(merged_1, 'BC_t')) == 4

  #We can change this using the subset_merge arg:
  dist_tree = IOT.file_to_dist_tree(mesh_file, sub_comm)
  merge.merge_zones(dist_tree, zones_path, sub_comm, output_path="MyMergedBase/MyMergedZone", subset_merge="None")
  merged_2 = I.copyTree(dist_tree) # Do copy for further comparaisons
  assert len(I.getZones(merged_2)) == 1
  assert len(I.getNodesFromType(merged_2, 'BC_t')) == 4*3

  #Note that we also used the output_path arg to specify the name of the merge zone and of
  # the base containing it
  assert I.getNodeFromPath(merged_2, "MyMergedBase/MyMergedZone") is not None

  
  # One more thing : the function has an API that can detect the zones that are connected by GCs and merge it :
  dist_tree = IOT.file_to_dist_tree(mesh_file, sub_comm)
  merge.merge_connected_zones(dist_tree, sub_comm)
  merged_zone = I.getZones(dist_tree)[0]
  merged_zone[0] = I.getName(I.getZones(merged_1)[0]) #Name differ so change before compare

  assert PT.is_same_tree(merged_1, dist_tree)

  if write_output:
    out_dir = TU.create_pytest_output_dir(sub_comm)
    IOT.dist_tree_to_file(merged_1, os.path.join(out_dir, 'merged_subset_name.hdf'), sub_comm)
    IOT.dist_tree_to_file(merged_2, os.path.join(out_dir, 'merged_subset_none.hdf'), sub_comm)

  # Compare to reference solution (since distributions can differ in // we write/load file to
  # reequilibrate it avoid)
  tmp_dir = TU.create_collective_tmp_dir(sub_comm)
  tmp_file = os.path.join(tmp_dir, 'out.hdf')
  IOT.dist_tree_to_file(merged_1, tmp_file, sub_comm)
  test_tree = IOT.file_to_dist_tree(tmp_file, sub_comm)
  ref_file  = os.path.join(ref_dir, 'U_Naca0012_multizone_merged.yaml')
  reference_tree = IOT.file_to_dist_tree(ref_file, sub_comm)
  for tree in reference_tree, test_tree:
    I._rmNodesByName(tree, '*#Size') #This is only related to IO
    I._rmNodesByName1(tree, 'CGNSLibraryVersion')
  assert PT.is_same_tree(reference_tree, test_tree, type_tol=True) #Somehow reader reput tree in int32 so use type_tol before correcint that
  TU.rm_collective_dir(tmp_dir, sub_comm)

@mark_mpi_test([3])
def test_merge_partial(sub_comm, write_output):

  mesh_file = os.path.join(TU.mesh_dir, 'U_Naca0012_multizone.yaml')

  dist_tree = IOT.file_to_dist_tree(mesh_file, sub_comm)

  # Here we will merge just two zones over three
  zones_path = ['BaseA/blk1', 'BaseA/blk3']

  merge.merge_zones(dist_tree, zones_path, sub_comm)
  merged_1 = I.copyTree(dist_tree) # Do copy for further comparaisons
  assert len(I.getZones(merged_1)) == 2
  merged_zone = I.getNodeFromPath(merged_1, 'BaseA/MergedZone')
  unmerged_zone = I.getNodeFromPath(merged_1, 'BaseB/blk2')
  assert unmerged_zone is not None and merged_zone is not None
  #By default, remaining GridConnectivity pointing to same zone are concatenated
  assert len(I.getNodesFromType(merged_zone, 'GridConnectivity_t')) == 1
  assert len(I.getNodesFromType(unmerged_zone, 'GridConnectivity_t')) == 1

  # This behaviour can be changed by setting argument concatenate_jns to False :
  dist_tree = IOT.file_to_dist_tree(mesh_file, sub_comm)
  merge.merge_zones(dist_tree, zones_path, sub_comm, concatenate_jns=False)
  merged_2 = I.copyTree(dist_tree) # Do copy for further comparaisons
  assert len(I.getZones(merged_1)) == 2
  merged_zone = I.getNodeFromPath(merged_2, 'BaseA/MergedZone')
  unmerged_zone = I.getNodeFromPath(merged_2, 'BaseB/blk2')

  #In both cases, the name of donor zone are updated for each join
  assert len(I.getNodesFromType(merged_zone, 'GridConnectivity_t')) == 2
  for jn in I.getNodesFromType(merged_zone, 'GridConnectivity_t'):
    assert I.getValue(jn) == 'BaseB/blk2'
  assert len(I.getNodesFromType(unmerged_zone, 'GridConnectivity_t')) == 2
  for jn in I.getNodesFromType(unmerged_zone, 'GridConnectivity_t'):
    assert I.getValue(jn) == 'BaseA/MergedZone'

  if write_output:
    out_dir = TU.create_pytest_output_dir(sub_comm)
    IOT.dist_tree_to_file(merged_1, os.path.join(out_dir, 'merged_concat_yes.hdf'), sub_comm)
    IOT.dist_tree_to_file(merged_2, os.path.join(out_dir, 'merged_concat_no.hdf'), sub_comm)