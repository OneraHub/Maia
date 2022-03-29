import pytest
from   pytest_mpi_check._decorator import mark_mpi_test
import os
import numpy as np

import Converter.Internal     as I
import maia.sids.Internal_ext as IE
import maia.sids.pytree       as PT
import maia.sids.sids         as sids

from   maia.utils              import test_utils   as TU
from   maia.cgns_io            import cgns_io_tree as IOT
from   maia.partitioning       import part                             as PPA
from   maia.partitioning.load_balancing import setup_partition_weights as SPW

from maia.tree_exchange import dist_to_part
from maia.tree_exchange import part_to_dist

def _load_dist_tree(sub_comm):
  # Load the distributed tree
  mesh_file = os.path.join(TU.mesh_dir, 'U_ATB_45.yaml')
  dist_tree = IOT.file_to_dist_tree(mesh_file, sub_comm)
  return dist_tree

def _create_dist_sol(dist_tree, sub_comm):
  # Create artificial fields on the distributed zone
  dist_zone = I.getZones(dist_tree)[0] #This mesh is single zone
  fs = I.newFlowSolution('FlowSolution', gridLocation='CellCenter', parent=dist_zone)
  cell_distri = IE.getDistribution(dist_zone, 'Cell')[1]
  n_cell_dist = cell_distri[1] - cell_distri[0]
  I.newDataArray('RankId', sub_comm.Get_rank() * np.ones(n_cell_dist), parent=fs)
  I.newDataArray('CellId', np.arange(cell_distri[0], cell_distri[1]) + 1, parent=fs)
  I.newDataArray('CstField', np.ones(n_cell_dist), parent=fs)

def _create_dist_dataset(dist_tree, sub_comm):
  bc_amont = PT.get_node_from_name_and_label(dist_tree, 'amont', 'BC_t')
  bc_aval  = PT.get_node_from_name_and_label(dist_tree, 'aval',  'BC_t')

  for i, bc in enumerate([bc_amont, bc_aval]):
    patch_distri = IE.getDistribution(bc, "Index")[1]
    patch_size   = sids.Subset.n_elem(bc) #This is local
    bcds = I.newBCDataSet(parent=bc)
    bcdata = I.newBCData(parent=bcds)
    I.newDataArray('BCId', (i+1) * np.ones(patch_size), parent=bcdata)
    I.newDataArray('FaceId', np.arange(patch_distri[0], patch_distri[1])+1, parent=bcdata)

def _split(dist_tree, sub_comm):
  zone_to_parts = SPW.npart_per_zone(dist_tree, sub_comm, 2)
  part_tree = PPA.partitioning(dist_tree, sub_comm, zone_to_parts=zone_to_parts)
  return part_tree


@mark_mpi_test([3])
class Test_fields_exchange:

  # Shared code for each case, who create the configuration
  def get_trees(self, sub_comm):
    dist_tree = _load_dist_tree(sub_comm)   #Load tree
    _create_dist_sol(dist_tree, sub_comm)   # Create artificial fields on the distributed zone
    part_tree = _split(dist_tree, sub_comm) # Split to get the partitioned tree
  
    # For now, we have no solution on the partitioned tree : 
    assert I.getNodeFromType(part_tree, 'FlowSolution_t') is None
    return dist_tree, part_tree


  def test_zone_level(self, sub_comm):
    dist_tree, part_tree = self.get_trees(sub_comm)
    # The lowest level api to exchange field works at zone level
    dist_zone  = I.getZones(dist_tree)[0]
    part_zones = I.getZones(part_tree)
    dist_to_part.data_exchange.dist_sol_to_part_sol(dist_zone, part_zones, sub_comm)
    # We retrieve our fields in each partition:
    for part_zone in part_zones:
      assert I.getNodeFromPath(part_zone, 'FlowSolution/RankId')   is not None
      assert I.getNodeFromPath(part_zone, 'FlowSolution/CellId')   is not None
      assert I.getNodeFromPath(part_zone, 'FlowSolution/CstField') is not None

    # Modify fields on partitions
    for part_zone in part_zones:
      for field in PT.iter_nodes_from_predicates(part_zone, 'FlowSolution_t/DataArray_t'):
        field[1] = 2*field[1]

    #We have the opposite function to send back data to the distributed zone :
    bck_zone = I.copyTree(dist_zone)
    part_to_dist.data_exchange.part_sol_to_dist_sol(dist_zone, part_zones, sub_comm)
    for field in PT.iter_nodes_from_predicates(dist_zone, 'FlowSolution_t/DataArray_t'):
      assert np.allclose(I.getNodeFromName(bck_zone, I.getName(field))[1] * 2, field[1])

  def test_created_sol(self, sub_comm):
    dist_tree, part_tree = self.get_trees(sub_comm)
    dist_zone  = I.getZones(dist_tree)[0]
    part_zones = I.getZones(part_tree)
    dist_to_part.data_exchange.dist_sol_to_part_sol(dist_zone, part_zones, sub_comm)

    # Note that if a FlowSolution or some fields are created on the partitioned zones, they will be transfered to the
    # distributed zone as well
    for part_zone in part_zones:
      part_fs = PT.get_child_from_name(part_zone, 'FlowSolution')
      I.newDataArray('PartRankId', sub_comm.Get_rank() * np.ones(sids.Zone.n_cell(part_zone)), part_fs)
      part_fs_new = I.newFlowSolution('CreatedFS', gridLocation='Vertex', parent=part_zone)
      I.newDataArray('PartRankId', sub_comm.Get_rank() * np.ones(sids.Zone.n_vtx(part_zone)), part_fs_new)

    part_to_dist.data_exchange.part_sol_to_dist_sol(dist_zone, part_zones, sub_comm)
    assert I.getNodeFromPath(dist_zone, 'FlowSolution/RankId')     is not None
    assert I.getNodeFromPath(dist_zone, 'FlowSolution/PartRankId') is not None
    assert I.getNodeFromPath(dist_zone, 'CreatedFS/PartRankId')    is not None

  def test_zone_level_with_filters(self, sub_comm):
    # Each low level function accepts an include or exclude argument, allowing a smoother control
    # of field to exchange :
    dist_tree, part_tree = self.get_trees(sub_comm)
    dist_zone  = I.getZones(dist_tree)[0]
    part_zones = I.getZones(part_tree)

    dist_to_part.data_exchange.dist_sol_to_part_sol(dist_zone, part_zones, sub_comm, exclude=['FlowSolution/CellId'])
    for part_zone in part_zones:
      assert I.getNodeFromPath(part_zone, 'FlowSolution/RankId')   is not None
      assert I.getNodeFromPath(part_zone, 'FlowSolution/CellId')   is None
      assert I.getNodeFromPath(part_zone, 'FlowSolution/CstField') is not None

    # Modify fields on partitions (on part_zones we have RankId and CstField)
    for part_zone in part_zones:
      for field in PT.iter_nodes_from_predicates(part_zone, 'FlowSolution_t/DataArray_t'):
        field[1] = 2*field[1]

    # Wildcard are also accepted
    bck_zone = I.copyTree(dist_zone)
    part_to_dist.data_exchange.part_sol_to_dist_sol(dist_zone, part_zones, sub_comm, include=['FlowSolution/*Id'])
    assert np.allclose(I.getNodeFromName(bck_zone, "RankId")[1] * 2, I.getNodeFromName(dist_zone, "RankId")[1]) #Only this one had a way-and-back transfert
    assert np.allclose(I.getNodeFromName(bck_zone, "CellId")[1]    , I.getNodeFromName(dist_zone, "CellId")[1])
    assert np.allclose(I.getNodeFromName(bck_zone, "CstField")[1]  , I.getNodeFromName(dist_zone, "CstField")[1])

  def test_tree_level(self, sub_comm):
    dist_tree, part_tree = self.get_trees(sub_comm)
    # We also provide a tree-level api who loop over all the zones to exchange data of all kind
    dist_to_part.dist_tree_to_part_tree_all(dist_tree, part_tree, sub_comm)
    for part_zone in I.getZones(part_tree):
      assert I.getNodeFromPath(part_zone, 'FlowSolution/RankId')   is not None
      assert I.getNodeFromPath(part_zone, 'FlowSolution/CellId')   is not None
      assert I.getNodeFromPath(part_zone, 'FlowSolution/CstField') is not None

@mark_mpi_test([2])
class Test_multiple_labels_exchange:

  # Shared code for each case, who create the configuration
  def get_trees(self, sub_comm):
    dist_tree = _load_dist_tree(sub_comm)     #Load tree
    _create_dist_sol(dist_tree, sub_comm)     # Create artificial fields on the distributed zone
    _create_dist_dataset(dist_tree, sub_comm) # Create artificial BCDataSet
    part_tree = _split(dist_tree, sub_comm)   # Split to get the partitioned tree
    return dist_tree, part_tree

  def _cleanup(self, part_tree):
    for label in ['FlowSolution_t', 'BCDataSet_t']:
      I._rmNodesByType(part_tree, label)

  def test_zone_level(self, sub_comm):
    dist_tree, part_tree = self.get_trees(sub_comm)
    dist_zone  = I.getZones(dist_tree)[0]
    part_zones = I.getZones(part_tree)
    # At tree level API, one can use the _all and _only versions to select only
    # some labels to exchange. Note that we can also filter the fields using the paths
    dist_to_part.dist_zone_to_part_zones_only(dist_zone, part_zones, sub_comm, \
        include_dict = {'FlowSolution_t' : ['*/*Id']})
    for part_zone in part_zones:
      assert I.getNodeFromPath(part_zone, 'FlowSolution/RankId')   is not None
      assert I.getNodeFromPath(part_zone, 'FlowSolution/CstField') is None
      assert I.getNodeFromType(part_zone, 'BCDataSet_t') is None
    self._cleanup(part_tree)

    dist_to_part.dist_zone_to_part_zones_only(dist_zone, part_zones, sub_comm, \
        include_dict = {'FlowSolution_t' : ['FlowSolution/*Id'], 'BCDataSet_t' : ['amont/*/*/*', 'aval/*/*/FaceId']})
    for part_zone in part_zones:
      assert I.getNodeFromPath(part_zone, 'FlowSolution/RankId')   is not None
      assert I.getNodeFromPath(part_zone, 'FlowSolution/CstField') is None
      bc_amont = PT.request_node_from_name_and_label(part_zone, 'amont', 'BC_t')
      if bc_amont is not None:
        assert I.getNodeFromName(bc_amont, 'BCId') is not None
        assert I.getNodeFromName(bc_amont, 'FaceId') is not None
      bc_aval = PT.request_node_from_name_and_label(part_zone, 'aval', 'BC_t')
      if bc_aval is not None:
        assert I.getNodeFromName(bc_aval, 'BCId') is None
        assert I.getNodeFromName(bc_aval, 'FaceId') is not None
    self._cleanup(part_tree)

    dist_to_part.dist_zone_to_part_zones_all(dist_zone, part_zones, sub_comm, \
        exclude_dict = {'FlowSolution_t' : ['*']}) #Everything, excepted FS
    for part_zone in part_zones:
      assert I.getNodeFromPath(part_zone, 'FlowSolution') is None
      bc_amont = PT.request_node_from_name_and_label(part_zone, 'amont', 'BC_t')
      bc_aval  = PT.request_node_from_name_and_label(part_zone, 'aval',  'BC_t')
      for bc in [bc_amont, bc_aval]:
        if bc is not None: #BC can be absent from partition
          assert I.getNodeFromName(bc, 'BCId') is not None
          assert I.getNodeFromName(bc, 'FaceId') is not None
    self._cleanup(part_tree)

  def test_tree_level(self, sub_comm):
    dist_tree, part_tree = self.get_trees(sub_comm)
    # At tree level API, one can select only some labels to exchange
    dist_to_part.dist_tree_to_part_tree_only_labels(dist_tree, part_tree, ['BCDataSet_t'], sub_comm)
    assert I.getNodeFromType(part_tree, 'FlowSolution_t') is None
    for part in I.getZones(part_tree):
      bc_amont = PT.request_node_from_name_and_label(part, 'amont', 'BC_t')
      bc_aval  = PT.request_node_from_name_and_label(part, 'aval',  'BC_t')
      for bc in [bc_amont, bc_aval]:
        if bc is not None: #BC can be absent from partition
          assert I.getNodeFromName(bc, 'BCId') is not None
          assert I.getNodeFromName(bc, 'FaceId') is not None