import numpy as np
import pytest
from pytest_mpi_check._decorator import mark_mpi_test

import Converter.Internal as I

from maia.transform.dist_tree import vertex_list as VL
from maia.generate import dcube_generator
from maia.distribution.distribution_function import uniform_distribution
from maia.sids import Internal_ext as IE

def test_is_subset_l():
  L = [2,8,10,3,3]
  assert VL._is_subset_l([2],        L) == True
  assert VL._is_subset_l([10,3],     L) == True
  assert VL._is_subset_l([10,3,3],   L) == True
  assert VL._is_subset_l([3,2,8],    L) == True
  assert VL._is_subset_l([1],        L) == False
  assert VL._is_subset_l([3,8,2],    L) == False
  assert VL._is_subset_l([10,3,3,1], L) == False

def test_is_before():
  L = [2,8,10,3,3]
  assert VL._is_before(L, 8, 10) == True
  assert VL._is_before(L, 8, 9 ) == True
  assert VL._is_before(L, 8, 2 ) == False
  assert VL._is_before(L, 7, 2 ) == False
  assert VL._is_before(L, 7, 14) == False

def test_roll_from():
  assert (VL._roll_from(np.array([2,4,8,16]), start_idx = 1) == [4,8,16,2]).all()
  assert (VL._roll_from(np.array([2,4,8,16]), start_value = 4) == [4,8,16,2]).all()
  assert (VL._roll_from(np.array([2,4,8,16]), start_value = 8, reverse=True) == [8,4,2,16]).all()
  with pytest.raises(AssertionError):
    VL._roll_from(np.array([2,4,8,16]), start_idx = 1, start_value = 8)

@mark_mpi_test([1,3])
def test_get_pl_face_vtx_local(sub_comm):
  tree = dcube_generator.dcube_generate(3,1.,[0,0,0], sub_comm)
  ngon = I.getNodeFromName(tree, "NGonElements")
  face_vtx, face_vtx_d, offset = VL.get_pl_face_vtx_local(np.array([3,6,2]), np.array([1,4,5]), ngon, sub_comm)
  assert (offset == np.arange(0,(3+1)*4,4)).all()
  assert (face_vtx == [5,8,7,4, 11,14,15,12, 3,6,5,2]).all()
  assert (face_vtx_d == [2,5,4,1, 6,9,8,5, 10,13,14,11]).all()

@mark_mpi_test(2)
def test_get_extended_pl(sub_comm):
  tree = dcube_generator.dcube_generate(3,1.,[0,0,0], sub_comm)
  ngon = I.getNodeFromName(tree, "NGonElements")
  if sub_comm.Get_rank() == 0:
    pl   = np.array([1,2,3])
    pl_d = np.array([9,10,11])
    pl_idx = np.array([0,4,8,12])
    pl_vtx = np.array([2,5,4,1, 3,6,5,2, 5,8,7,4])
    skip_f = np.array([True, True, True])
  else:
    pl   = np.array([4])
    pl_d = np.array([12])
    pl_idx = np.array([0,4])
    pl_vtx = np.array([6,9,8,5])
    skip_f = np.array([False])
  ext_pl, ext_pl_d = VL.get_extended_pl(pl, pl_d, pl_idx, pl_vtx, sub_comm)
  assert (ext_pl   == [1,2,3,4]).all()
  assert (ext_pl_d == [9,10,11,12]).all()

  ext_pl, ext_pl_d = VL.get_extended_pl(pl, pl_d, pl_idx, pl_vtx, sub_comm, skip_f)
  if sub_comm.Get_rank() == 0:
    assert len(ext_pl) == len(ext_pl_d) == 0
  else:
    assert (ext_pl   == [1,2,3,4]).all()
    assert (ext_pl_d == [9,10,11,12]).all()

def test_search_by_intersection():
  empty = np.empty(0, np.int)
  plv, plv_opp, face_is_treated = VL._search_by_intersection(np.array([0]), empty, empty)
  assert (plv == empty).all()
  assert (plv_opp == empty).all()
  assert (face_is_treated == empty).all()

  #Some examples from cube3, top/down jns
  #Single face can not be treated
  plv, plv_opp, face_is_treated = VL._search_by_intersection([0,4], [5,8,7,4], [22,25,26,23])
  assert (plv == [4,5,7,8]).all()
  assert (plv_opp == [0,0,0,0]).all()
  assert (face_is_treated == [False]).all()

  #Example from Cube4 : solo face will not be treated
  plv, plv_opp, face_is_treated = VL._search_by_intersection([0,4,8,12], \
      [6,10,9,5, 12,16,15,11, 3,7,6,2], [37,41,42,38, 43,47,48,44, 34,38,39,35])
  assert (plv == [2,3,5,6,7,9,10,11,12,15,16]).all()
  assert (plv_opp == [34,35,37,38,39,41,42,0,0,0,0]).all()
  assert (face_is_treated == [True, False, True]).all()

  #Here is a crazy exemple where third face can not be treated on first pass,
  # but is the treated thanks to already treated faces
  plv, plv_opp, face_is_treated = VL._search_by_intersection([0,4,9,14], \
      [2,4,3,1, 10,9,6,8,7, 5,6,3,4,7], [12,11,13,14, 19,20,17,18,16, 13,16,15,17,14])
  assert (plv == np.arange(1,10+1)).all()
  assert (plv_opp == np.arange(11,20+1)).all()
  assert (face_is_treated == [True, True, True]).all()

@mark_mpi_test(3)
def test_search_with_geometry(sub_comm):
  empty = np.empty(0, np.int)
  tree = dcube_generator.dcube_generate(4,1.,[0,0,0], sub_comm)
  zone = I.getZones(tree)[0]
  if sub_comm.Get_rank() == 0:
    pl_face_vtx = [2,6,7,3]
    pld_face_vtx = [51,55,54,50]
  else:
    pl_face_vtx = empty
    pld_face_vtx = empty

  plv, plvd = VL._search_with_geometry(zone, pl_face_vtx, pld_face_vtx, 0, sub_comm)

  if sub_comm.Get_rank() == 0:
    assert (plv  == [2,3,6,7]).all()
    assert (plvd == [50,51,54,55]).all()
  else:
    assert (plv == plvd == empty).all()

class Test_generate_jn_vertex_list():
  @mark_mpi_test([1,3,4])
  def test_single_zone_topo(self, sub_comm):
    #With this configuration, we have isolated faces when np=4
    tree = dcube_generator.dcube_generate(4,1.,[0,0,0], sub_comm)
    zone = I.getZones(tree)[0]
    ngon = I.getNodeFromName(zone, "NGonElements")
    I._rmNodesByType(zone, 'ZoneBC_t')
    #Create fake jn
    zgc = I.newZoneGridConnectivity(parent=zone)
    gcA = I.newGridConnectivity('matchA', I.getName(zone), 'Abutting1to1', zgc)
    full_pl     = np.array([1,2,3,4,5,6,7,8,9])
    full_pl_opp = np.array([28,29,30,31,32,33,34,35,36])
    distri_pl   = uniform_distribution(9, sub_comm)
    I.newGridLocation('FaceCenter', gcA)
    I.newPointList('PointList', full_pl[distri_pl[0]:distri_pl[1]].reshape(1,-1), gcA)
    I.newPointList('PointListDonor', full_pl_opp[distri_pl[0]:distri_pl[1]].reshape(1,-1), gcA)
    IE.newDistribution({'Index' : distri_pl}, gcA)


    pl_vtx, pl_vtx_opp, distri_jn_vtx = VL.generate_jn_vertex_list(zone, ngon, gcA, sub_comm)

    expt_full_pl_vtx     = np.arange(1,16+1)
    expt_full_pl_vtx_opp = np.arange(49,64+1)
    assert (distri_jn_vtx == uniform_distribution(16, sub_comm)).all()
    assert (pl_vtx == expt_full_pl_vtx[distri_jn_vtx[0]:distri_jn_vtx[1]]).all()
    assert (pl_vtx_opp == expt_full_pl_vtx_opp[distri_jn_vtx[0]:distri_jn_vtx[1]]).all()

  @mark_mpi_test(2)
  def test_single_zone_geo(self, sub_comm):
    tree = dcube_generator.dcube_generate(4,1.,[0,0,0], sub_comm)
    zone = I.getZones(tree)[0]
    ngon = I.getNodeFromName(zone, "NGonElements")
    I._rmNodesByType(zone, 'ZoneBC_t')
    zgc = I.newZoneGridConnectivity(parent=zone)
    gcA = I.newGridConnectivity('matchA', I.getName(zone), 'Abutting1to1', zgc)
    I.newGridLocation('FaceCenter', gcA)
    if sub_comm.Get_rank() == 1:
      I.newPointList('PointList', [[2]], gcA)
      I.newPointList('PointListDonor', [[29]], gcA)
      IE.newDistribution({'Index' : [0,1,1]}, gcA)
    else:
      I.newPointList('PointList', np.empty((1,0), dtype=np.int), gcA)
      I.newPointList('PointListDonor', np.empty((1,0), dtype=np.int), gcA)
      IE.newDistribution({'Index' : [0,0,1]}, gcA)

    pl_vtx, pld_vtx, distri_jn_vtx = VL.generate_jn_vertex_list(zone, ngon, gcA, sub_comm)

    expt_full_pl_vtx  = [2,3,6,7]
    expt_full_pld_vtx = [50,51,54,55]
    assert (distri_jn_vtx == uniform_distribution(4, sub_comm)).all()
    assert (pl_vtx == expt_full_pl_vtx[distri_jn_vtx[0]:distri_jn_vtx[1]]).all()
    assert (pld_vtx == expt_full_pld_vtx[distri_jn_vtx[0]:distri_jn_vtx[1]]).all()

