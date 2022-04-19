import pytest
from pytest_mpi_check._decorator import mark_mpi_test
from maia.utils import test_utils as TU
import os
import numpy as np

import Converter.Internal as I
import maia.cgns_io.cgns_io_tree
import maia.transform


@mark_mpi_test([1,4])
def test_add_nfaces(sub_comm, write_output):
  mesh_file = os.path.join(TU.mesh_dir, 'Uelt_M6Wing.yaml')
  dist_tree = maia.cgns_io.cgns_io_tree.file_to_dist_tree(mesh_file, sub_comm)

  # Note: `std_elements_to_ngons` is supposed to work, because it is tested in another test
  maia.transform.std_elements_to_ngons(dist_tree, sub_comm)
  I._rmNodesByName(dist_tree, 'NFACE_n')

  maia.transform.add_nfaces(dist_tree,sub_comm)
  print(dist_tree)
  I.printTree(dist_tree)

  # > Poly sections appear
  nface_node = I.getNodeFromName(dist_tree, 'NFACE_n')
  assert nface_node is not None

  # > Some non-regression checks
  assert np.all(I.getVal(I.getNodeFromName(nface_node, 'ElementRange')) == [2695,3990])

  if write_output:
    out_dir = TU.create_pytest_output_dir(sub_comm)
    maia.cgns_io.cgns_io_tree.dist_tree_to_file(dist_tree, os.path.join(out_dir, 'U_M6Wing_ngon.cgns'), sub_comm)
