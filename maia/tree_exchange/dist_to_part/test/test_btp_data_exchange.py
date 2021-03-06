import pytest
import numpy      as np
import Converter.Internal as I

from pytest_mpi_check._decorator import mark_mpi_test

from   maia.utils        import parse_yaml_cgns
import maia.tree_exchange.dist_to_part.data_exchange as BTP
from maia import npy_pdm_gnum_dtype as pdm_dtype

dtype = 'I4' if pdm_dtype == np.int32 else 'I8'

dt0 = """
ZoneU Zone_t [[6,0,0]]:
  GridCoordinates GridCoordinates_t:
    CX DataArray_t [1,2,3]:
    CY DataArray_t [2,2,2]:
  ZBC ZoneBC_t:
    BC BC_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[18, 22]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {0} [0,2,6]:
      BCDSWithoutPL BCDataSet_t:
        DirichletData BCData_t:
          field DataArray_t [1,2]:
      BCDSWithPL BCDataSet_t:
        DirichletData BCData_t:
          field DataArray_t R8 [100]:
        PointList IndexArray_t [[10]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {0} [0,1,1]:
  FlowSolution FlowSolution_t:
    GridLocation GridLocation_t "Vertex":
    field1 DataArray_t I4 [0,0,0]:
    field2 DataArray_t R8 [6.,5.,4.]:
  FlowSolWithPL FlowSolution_t:
    GridLocation GridLocation_t "CellCenter":
    PointList IndexArray_t [[2]]:
    :CGNS#Distribution UserDefinedData_t:
      Index DataArray_t {0} [0,1,3]:
    field1 DataArray_t [10]:
  ZSRWithoutPL ZoneSubRegion_t:
    GridLocation GridLocation_t "FaceCenter":
    BCRegionName Descriptor_t "BC":
    field DataArray_t R8 [100,200]:
  ZSRWithPL ZoneSubRegion_t:
    GridLocation GridLocation_t "Vertex":
    PointList IndexArray_t [[6]]:
    field DataArray_t I4 [42]:
    :CGNS#Distribution UserDefinedData_t:
      Index DataArray_t [0,1,2]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t {0} [0,3,6]:
ZoneS Zone_t [[2,0,0],[3,0,0],[1,0,0]]:
  GridCoordinates GridCoordinates_t:
    CX DataArray_t [1,2,3]:
    CY DataArray_t [2,2,2]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t {0} [0,3,6]:
""".format(dtype)

dt1 = """
ZoneU Zone_t [[6,0,0]]:
  GridCoordinates GridCoordinates_t:
    CX DataArray_t [4,5,6]:
    CY DataArray_t [1,1,1]:
  ZBC ZoneBC_t:
    BC BC_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[13, 39, 41, 9]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {0} [2,6,6]:
      BCDSWithoutPL BCDataSet_t:
        DirichletData BCData_t:
          field DataArray_t [4,3,2,1]:
      BCDSWithPL BCDataSet_t:
        DirichletData BCData_t:
          field DataArray_t R8 []:
        PointList IndexArray_t [[]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {0} [1,1,1]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t {0} [3,6,6]:
  FlowSolution FlowSolution_t:
    GridLocation GridLocation_t "Vertex":
    field1 DataArray_t I4 [1,1,1]:
    field2 DataArray_t R8 [3.,2.,1.]:
  FlowSolWithPL FlowSolution_t:
    GridLocation GridLocation_t "CellCenter":
    PointList IndexArray_t [[6,4]]:
    :CGNS#Distribution UserDefinedData_t:
      Index DataArray_t {0} [1,3,3]:
    field1 DataArray_t [20,30]:
  ZSRWithoutPL ZoneSubRegion_t:
    GridLocation GridLocation_t "FaceCenter":
    BCRegionName Descriptor_t "BC":
    field DataArray_t R8 [300,400,500,600]:
  ZSRWithPL ZoneSubRegion_t:
    GridLocation GridLocation_t "Vertex":
    PointList IndexArray_t [[2]]:
    field DataArray_t I4 [24]:
    :CGNS#Distribution UserDefinedData_t:
      Index DataArray_t [1,2,2]:
ZoneS Zone_t [[2,0,0],[3,0,0],[1,0,0]]:
  GridCoordinates GridCoordinates_t:
    CX DataArray_t [4,5,6]:
    CY DataArray_t [1,1,1]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t {0} [3,6,6]:
""".format(dtype)

@mark_mpi_test(2)
def test_dist_to_part(sub_comm):
  dist_data = dict()
  expected_part_data = dict()
  if sub_comm.Get_rank() == 0:
    partial_distri = np.array([0, 5, 10], dtype=pdm_dtype)
    ln_to_gn_list = [np.array([2,4,6,10], dtype=pdm_dtype)]
    dist_data["field"] = np.array([1., 2., 3., 4., 5.])
    expected_part_data["field"] = [np.array([2., 4., 6., 1000.])]
  else:
    partial_distri = np.array([5, 10, 10], dtype=pdm_dtype)
    ln_to_gn_list = [np.array([9,7,5,3,1], dtype=pdm_dtype),
                     np.array([8], dtype=pdm_dtype),
                     np.array([1], dtype=pdm_dtype)]
    dist_data["field"] = np.array([6., 7., 8., 9., 1000.])
    expected_part_data["field"] = [np.array([9., 7., 5., 3., 1.]), np.array([8.]), np.array([1.])]

  part_data = BTP.dist_to_part(partial_distri, dist_data, ln_to_gn_list, sub_comm)
  assert len(part_data["field"]) == len(ln_to_gn_list)
  for i_part in range(len(ln_to_gn_list)):
    assert part_data["field"][i_part].dtype == np.float64
    assert (part_data["field"][i_part] == expected_part_data["field"][i_part]).all()

@mark_mpi_test(2)
def test_dist_to_part_with_void(sub_comm):
  dist_data = dict()
  expected_part_data = dict()
  if sub_comm.Get_rank() == 0:
    partial_distri = np.array([0, 5, 10], dtype=pdm_dtype)
    ln_to_gn_list = [np.array([10,8], dtype=pdm_dtype)]
    dist_data["field"] = np.array([1., 2., 3., 4., 5.])
    expected_part_data["field"] = [np.array([1000., 8.])]
  else:
    partial_distri = np.array([5, 10, 10], dtype=pdm_dtype)
    ln_to_gn_list = list()
    dist_data["field"] = np.array([6., 7., 8., 9., 1000.])
    expected_part_data["field"] = list()

  part_data = BTP.dist_to_part(partial_distri, dist_data, ln_to_gn_list, sub_comm)
  assert len(part_data["field"]) == len(ln_to_gn_list)
  for i_part in range(len(ln_to_gn_list)):
    assert part_data["field"][i_part].dtype == np.float64
    assert (part_data["field"][i_part] == expected_part_data["field"][i_part]).all()

@mark_mpi_test(2)
def test_dist_coords_to_part_coords_U(sub_comm):
  if sub_comm.Get_rank() == 0:
    dt = dt0
    pt = """
  ZoneU.P0.N0 Zone_t [[2,0,0]]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [1,6]:
    """.format(dtype)
  elif sub_comm.Get_rank() == 1:
    dt = dt1
    pt = """
  ZoneU.P1.N0 Zone_t [[2,0,0]]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [5,2]:
  ZoneU.P1.N1 Zone_t [[2,0,0]]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [3,4]:
  """.format(dtype)

  dist_tree = parse_yaml_cgns.to_cgns_tree(dt)
  part_tree = parse_yaml_cgns.to_cgns_tree(pt)

  dist_zone  = I.getZones(dist_tree)[0]
  part_zones = I.getZones(part_tree)
  BTP.dist_coords_to_part_coords(dist_zone, part_zones, sub_comm)

  if sub_comm.Get_rank() == 0:
    assert (I.getNodeFromPath(part_zones[0], 'GridCoordinates/CX')[1] == [1,6]).all()
    assert (I.getNodeFromPath(part_zones[0], 'GridCoordinates/CY')[1] == [2,1]).all()
  elif sub_comm.Get_rank() == 1:
    assert (I.getNodeFromPath(part_zones[0], 'GridCoordinates/CX')[1] == [5,2]).all()
    assert (I.getNodeFromPath(part_zones[0], 'GridCoordinates/CY')[1] == [1,2]).all()
    assert (I.getNodeFromPath(part_zones[1], 'GridCoordinates/CX')[1] == [3,4]).all()
    assert (I.getNodeFromPath(part_zones[1], 'GridCoordinates/CY')[1] == [2,1]).all()

@mark_mpi_test(2)
def test_dist_coords_to_part_coords_S(sub_comm):
  if sub_comm.Get_rank() == 0:
    dt = dt0
    pt = """
  ZoneS.P0.N0 Zone_t [[2,0,0],[1,0,0],[1,0,0]]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [1,2]:
    """.format(dtype)
  elif sub_comm.Get_rank() == 1:
    dt = dt1
    pt = """
  ZoneS.P1.N0 Zone_t [[2,0,0],[2,0,0],[1,0,0]]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [5,6,3,4]:
  """.format(dtype)

  dist_tree = parse_yaml_cgns.to_cgns_tree(dt)
  part_tree = parse_yaml_cgns.to_cgns_tree(pt)

  dist_zone  = I.getZones(dist_tree)[1]
  part_zones = I.getZones(part_tree)
  BTP.dist_coords_to_part_coords(dist_zone, part_zones, sub_comm)

  if sub_comm.Get_rank() == 0:
    assert (I.getNodeFromPath(part_zones[0], 'GridCoordinates/CX')[1] == \
        np.array([1,2]).reshape((2,1,1), order='F')).all()
    assert (I.getNodeFromPath(part_zones[0], 'GridCoordinates/CY')[1] == \
        np.array([2,2]).reshape((2,1,1), order='F')).all()
  elif sub_comm.Get_rank() == 1:
    assert (I.getNodeFromPath(part_zones[0], 'GridCoordinates/CX')[1] == \
        np.array([5,6,3,4]).reshape((2,2,1),order='F')).all()
    assert (I.getNodeFromPath(part_zones[0], 'GridCoordinates/CY')[1] == \
        np.array([1,1,2,1]).reshape((2,2,1),order='F')).all()

@mark_mpi_test(2)
def test_dist_sol_to_part_sol_allvtx(sub_comm):
  if sub_comm.Get_rank() == 0:
    dt = dt0
    pt = """
  ZoneU.P0.N0 Zone_t [[2,0,0]]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [1,6]:
    """.format(dtype)
  elif sub_comm.Get_rank() == 1:
    dt = dt1
    pt = """
  ZoneU.P1.N0 Zone_t [[2,0,0]]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [5,2]:
  ZoneU.P1.N1 Zone_t [[2,0,0]]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [3,4]:
  """.format(dtype)

  dist_tree = parse_yaml_cgns.to_cgns_tree(dt)
  part_tree = parse_yaml_cgns.to_cgns_tree(pt)

  dist_zone  = I.getZones(dist_tree)[0]
  part_zones = I.getZones(part_tree)
  BTP.dist_sol_to_part_sol(dist_zone, part_zones, sub_comm)

  for zone in part_zones:
    assert I.getNodeFromPath(zone, 'FlowSolution/field1')[1].dtype == np.int32
    assert I.getNodeFromPath(zone, 'FlowSolution/field2')[1].dtype == np.float64
  if sub_comm.Get_rank() == 0:
    assert (I.getNodeFromPath(part_zones[0], 'FlowSolution/field1')[1] == [0,1]).all()
    assert (I.getNodeFromPath(part_zones[0], 'FlowSolution/field2')[1] == [6,1]).all()
  elif sub_comm.Get_rank() == 1:
    assert (I.getNodeFromPath(part_zones[0], 'FlowSolution/field1')[1] == [1,0]).all()
    assert (I.getNodeFromPath(part_zones[0], 'FlowSolution/field2')[1] == [2,5]).all()
    assert (I.getNodeFromPath(part_zones[1], 'FlowSolution/field1')[1] == [0,1]).all()
    assert (I.getNodeFromPath(part_zones[1], 'FlowSolution/field2')[1] == [4,3]).all()

@mark_mpi_test(2)
def test_dist_sol_to_part_sol_pl(sub_comm):
  if sub_comm.Get_rank() == 0:
    dt = dt0
    pt = """
  ZoneU.P0.N0 Zone_t [[2,0,0]]:
    FlowSolWithPL FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      PointList IndexArray_t [[1]]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Index DataArray_t {0} [2]:
    """.format(dtype)
  elif sub_comm.Get_rank() == 1:
    dt = dt1
    pt = """
  ZoneU.P1.N0 Zone_t [[2,0,0]]:
  ZoneU.P1.N1 Zone_t [[2,0,0]]:
    FlowSolWithPL FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      PointList IndexArray_t [[12,15]]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Index DataArray_t {0} [3,1]:
  """.format(dtype)

  dist_tree = parse_yaml_cgns.to_cgns_tree(dt)
  I._rmNodesByName(dist_tree, 'FlowSolution') #Test only pl sol here
  part_tree = parse_yaml_cgns.to_cgns_tree(pt)

  dist_zone  = I.getZones(dist_tree)[0]
  part_zones = I.getZones(part_tree)
  BTP.dist_sol_to_part_sol(dist_zone, part_zones, sub_comm)

  if sub_comm.Get_rank() == 0:
    assert (I.getNodeFromPath(part_zones[0], 'FlowSolWithPL/field1')[1] == [20]).all()
  elif sub_comm.Get_rank() == 1:
    assert I.getNodeFromPath(part_zones[0], 'FlowSolWithPL/field1') is None
    assert I.getNodeFromPath(part_zones[1], 'FlowSolWithPL/field1')[1].shape == (2,)
    assert (I.getNodeFromPath(part_zones[1], 'FlowSolWithPL/field1')[1] == [30,10]).all()

@mark_mpi_test(2)
def test_dist_dataset_to_part_dataset(sub_comm):
  if sub_comm.Get_rank() == 0:
    dt = dt0
    pt = """
  ZoneU.P0.N0 Zone_t [[2,0,0]]:
    ZBC ZoneBC_t:
      BC BC_t:
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[1, 12, 21]]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Index DataArray_t {0} [2,5,1]:
    """.format(dtype)
  elif sub_comm.Get_rank() == 1:
    dt = dt1
    pt = """
  ZoneU.P1.N0 Zone_t [[2,0,0]]:
    ZBC ZoneBC_t:
  ZoneU.P1.N1 Zone_t [[2,0,0]]:
    ZBC ZoneBC_t:
      BC BC_t:
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[1, 29, 108]]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Index DataArray_t {0} [6,3,4]:
        BCDSWithPL BCDataSet_t:
          PointList IndexArray_t [[108]]:
          :CGNS#GlobalNumbering UserDefinedData_t:
            Index DataArray_t {0} [1]:
  """.format(dtype)

  dist_tree = parse_yaml_cgns.to_cgns_tree(dt)
  part_tree = parse_yaml_cgns.to_cgns_tree(pt)

  dist_zone  = I.getZones(dist_tree)[0]
  part_zones = I.getZones(part_tree)
  BTP.dist_dataset_to_part_dataset(dist_zone, part_zones, sub_comm)

  if sub_comm.Get_rank() == 0:
    assert (I.getNodeFromPath(part_zones[0], 'ZBC/BC/BCDSWithoutPL/DirichletData/field')[1] == [2,2,1]).all()
    assert I.getNodeFromPath(part_zones[0], 'ZBC/BC/BCDSWitPL/DirichletData/field') is None
  elif sub_comm.Get_rank() == 1:
    assert I.getNodeFromPath(part_zones[0], 'ZBC/BC') is None
    assert (I.getNodeFromPath(part_zones[1], 'ZBC/BC/BCDSWithoutPL/DirichletData/field')[1] == [1,4,3]).all()
    assert (I.getNodeFromPath(part_zones[1], 'ZBC/BC/BCDSWithPL/DirichletData/field')[1] == [100.]).all()

@mark_mpi_test(2)
def test_dist_subregion_to_part_subregion(sub_comm):
  if sub_comm.Get_rank() == 0:
    dt = dt0
    pt = """
  ZoneU.P0.N0 Zone_t [[2,0,0]]:
    ZBC ZoneBC_t:
      BC BC_t:
        PointList IndexArray_t [[1, 12, 21]]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Index DataArray_t {0} [2,5,1]:
    """.format(dtype)
  elif sub_comm.Get_rank() == 1:
    dt = dt1
    pt = """
  ZoneU.P1.N0 Zone_t [[2,0,0]]:
    ZSRWithPL ZoneSubRegion_t:
      GridLocation GridLocation_t "Vertex":
      PointList IndexArray_t [[1,2]]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Index DataArray_t {0} [1,2]:
  ZoneU.P1.N1 Zone_t [[2,0,0]]:
    ZBC ZoneBC_t:
      BC BC_t:
        PointList IndexArray_t [[1, 29, 108]]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Index DataArray_t {0} [6,3,4]:
  """.format(dtype)

  dist_tree = parse_yaml_cgns.to_cgns_tree(dt)
  part_tree = parse_yaml_cgns.to_cgns_tree(pt)

  dist_zone  = I.getZones(dist_tree)[0]
  part_zones = [zone for zone in I.getZones(part_tree) if 'ZoneU' in I.getName(zone)]
  BTP.dist_subregion_to_part_subregion(dist_zone, part_zones, sub_comm)

  if sub_comm.Get_rank() == 0:
    assert (I.getNodeFromPath(part_zones[0], 'ZSRWithoutPL/field')[1] == [200,500,100]).all()
    assert I.getNodeFromPath(part_zones[0], 'ZSRWithPL') is None
  elif sub_comm.Get_rank() == 1:
    assert I.getNodeFromPath(part_zones[0], 'ZSRWithoutPL') is None
    assert (I.getNodeFromPath(part_zones[0], 'ZSRWithPL/field')[1] == [42,24]).all()
    assert I.getNodeFromPath(part_zones[1], 'ZSRWithPL') is None
    assert (I.getNodeFromPath(part_zones[1], 'ZSRWithoutPL/field')[1] == [600,300,400]).all()
