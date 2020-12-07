from .range_to_slab          import compute_slabs
import Converter.Internal as I

def create_data_array_filterS(data_shape, distrib):
  slab_list  = compute_slabs(data_shape, distrib[0:2])
  dn_da    = distrib[1] - distrib[0]
  DSFILEDA = []
  for slab in slab_list:
    iS,iE, jS,jE, kS,kE = [item for bounds in slab for item in bounds]
    # For DSFile, each slab must be of form
    # [[offsetI, offsetJ, offsetK], [1,1,1], [nbI, nbJ, nbK], [1,1,1]]
    # such that we have a list looking like (here i=0...4 pour j=k=0 then i=0...2 for j=k=1)
    # DataSpaceFILE = [[[0,0,0], [1,1,1], [5,1,1], [1,1,1],
                      # [0,1,1], [1,1,1], [3,1,1], [1,1,1]]]
    DSFILEDA.extend([[iS,jS,kS], [1,1,1], [iE-iS, jE-jS, kE-kS], [1,1,1]])
  DSMMRYDA = [[0]    , [1]    , [dn_da], [1]]
  DSFILEDA = list([list(DSFILEDA)])
  DSGLOBDA = [list(data_shape)]
  DSFORMDA = [[0]]
  return DSMMRYDA + DSFILEDA + DSGLOBDA + DSFORMDA

def create_data_array_filterU(distrib):
  dn_da    = distrib[1] - distrib[0]
  DSMMRYDA = [[0         ], [1], [dn_da], [1]]
  DSFILEDA = [[distrib[0]], [1], [dn_da], [1]]
  DSGLOBDA = [[distrib[2]]]
  DSFORMDA = [[0]]
  return DSMMRYDA + DSFILEDA + DSGLOBDA + DSFORMDA

def create_point_list_filter(distrib):
  dn_pl    = distrib[1] - distrib[0]
  DSMMRYPL = [[0,0          ], [1, 1], [1, dn_pl], [1, 1]]
  DSFILEPL = [[0, distrib[0]], [1, 1], [1, dn_pl], [1, 1]]
  DSGLOBPL = [[1, distrib[2]]]
  DSFORMPL = [[0]]
  return DSMMRYPL + DSFILEPL + DSGLOBPL + DSFORMPL

def create_data_array_filter(distrib, data_shape=None):
  """
  """
  if data_shape is None or len(data_shape) == 1: #Unstructured
    hdf_data_space = create_data_array_filterU(distrib)
  elif len(data_shape) == 2 and data_shape[0] == 1:
    hdf_data_space = create_point_list_filter(distrib)
  else: #Structured
    hdf_data_space = create_data_array_filterS(data_shape, distrib)

  return hdf_data_space
