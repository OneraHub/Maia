import pytest
import itertools as ITT

def inc(x):
    return x + 1

def test_answer():
    assert inc(3) == 4
    assert inc(4) == 5
    assert inc(5) == 6
    # assert inc(6) == 8

# --------------------------------------------------------------------------
def assert_mpi(comm, rank, cond ):
  if(comm.rank == rank):
    print(cond)
    assert(cond == True)
  else:
    pass

# --------------------------------------------------------------------------
def test_first_step():
  """
  """
  # print("test_first_step")
  from maia.utils import boundary_algorihms as MUB
  # print(MUB.__file__)
  from maia.utils import first_step as MUF

  base = MUF.cgns_base("cgns_base", 1)

  zone_u1 = MUF.zone_unstructured("zone_name", 1)
  assert(zone_u1.global_id == 1          );
  assert(zone_u1.name      == "zone_name");

  zone_s1 = MUF.zone_structured("cartesian", 2)
  assert(zone_s1.global_id == 2          );
  assert(zone_s1.name      == "cartesian");

  # MUF.add_zone_to_base(base, zone_u1);
  # MUF.add_zone_to_base(base, zone_s1);

  print(zone_u1.global_id)
  print(zone_s1.global_id)
  print(base)

# --------------------------------------------------------------------------
def test_mpi():
  """
  """
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  print("hello", comm.size)
  assert comm.size > 0
  assert_mpi(comm, 0, comm.rank == 0)
  assert_mpi(comm, 1, comm.rank == 1)

# --------------------------------------------------------------------------
@pytest.mark.mpi
def test_size():
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  assert comm.size > 0

# --------------------------------------------------------------------------
@pytest.mark.mpi(min_size=2)
def test_size():
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  print("hello", comm.size, comm.rank)
  assert comm.size >= 2

# --------------------------------------------------------------------------
# def pytest_generate_tests(metafunc):
#   idlist = []
#   argvalues = []
#   for scenario in metafunc.cls.scenarios:
#       idlist.append(scenario[0])
#       items = scenario[1].items()
#       argnames = [x[0] for x in items]
#       argvalues.append([x[1] for x in items])
#   metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


# scenario1 = ("basic"   , {"attribute": "value" })
# scenario2 = ("advanced", {"attribute": "value2"})

# class TestSampleWithScenarios:
#   scenarios = [scenario1, scenario2]

#   def test_demo1(self, attribute):
#     assert isinstance(attribute, str)

#   def test_demo2(self, attribute):
#     assert isinstance(attribute, str)
