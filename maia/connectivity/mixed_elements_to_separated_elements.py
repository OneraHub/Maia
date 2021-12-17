from mpi4py                   import MPI
import Converter.Internal     as I
import numpy                  as np
from maia.sids                import sids
import maia.sids.Internal_ext as IE

from maia.sids.elements_utils import elements_properties
from maia import npy_pdm_gnum_dtype as pdm_dtype
from cmaia.connectivity.mixed_elements_to_separated_elements import count_type_in_mixed_elements, sort_by_type, idx_from_count, redistribute_mixed_to_separated

def mixed_elements_to_separated_elements_zone(dist_zone, dist_zone_mixed, comm):
  """
  """

  # > Hook all MIXED Elements
  count_by_type            = np.zeros(len(elements_properties), dtype=pdm_dtype, order='F')
  count_elmt_vtx_n_by_type = np.zeros(len(elements_properties), dtype=pdm_dtype, order='F')
  n_elmt_vtx_tot = 0;
  for elmt in I.getNodesFromType1(dist_zone, "Elements_t"):
    if(I.getValue(elmt)[0] != 20):
      raise NotImplementedError(f"mixed_elements_to_separated_elements_zone only works with pure MIXED connectivity")
    else:

      er_n  = I.getNodeFromName1(elmt, "ElementRange"       )
      eso_n = I.getNodeFromName1(elmt, "ElementStartOffset" )
      ec_n  = I.getNodeFromName1(elmt, "ElementConnectivity")

      er  = I.getValue(er_n );
      eso = I.getValue(eso_n);
      ec  = I.getValue(ec_n );
      distrib_elmt = I.getVal(IE.getDistribution(elmt, 'Element'))

      n_elmt_vtx_tot += count_type_in_mixed_elements(count_by_type, count_elmt_vtx_n_by_type, er, eso, ec, distrib_elmt)

  # Reset
  g_count_by_type = comm.allreduce(count_by_type, op=MPI.SUM)

  count_by_type_idx            = idx_from_count(count_by_type)
  g_count_by_type_idx          = idx_from_count(g_count_by_type)
  count_elmt_vtx_n_by_type_idx = idx_from_count(count_elmt_vtx_n_by_type)

  print("g_count_by_type              : ", g_count_by_type             )
  print("count_by_type_idx            : ", count_by_type_idx           )
  print("g_count_by_type_idx          : ", g_count_by_type_idx         )
  print("count_elmt_vtx_n_by_type_idx : ", count_elmt_vtx_n_by_type_idx)

  n_elmt_concat = np.sum(count_by_type)

  count_by_type           .fill(0)
  count_elmt_vtx_n_by_type.fill(0)

  concat_elmt_vtx   = np.empty(n_elmt_vtx_tot, dtype=pdm_dtype, order='F')
  concat_elmt_vtx_n = np.empty(n_elmt_concat , dtype='int32'  , order='F')
  dnew_to_old       = np.empty(n_elmt_concat , dtype=pdm_dtype, order='F')
  print("n_elmt_concat : ", n_elmt_concat)

  for elmt in I.getNodesFromType1(dist_zone, "Elements_t"):
    er_n  = I.getNodeFromName1(elmt, "ElementRange"       )
    eso_n = I.getNodeFromName1(elmt, "ElementStartOffset" )
    ec_n  = I.getNodeFromName1(elmt, "ElementConnectivity")

    er  = I.getValue(er_n );
    eso = I.getValue(eso_n);
    ec  = I.getValue(ec_n );
    distrib_elmt = I.getVal(IE.getDistribution(elmt, 'Element'))

    # Je pense qu'on peut generer le old_to_new ou le new_to_old inside
    sort_by_type(count_by_type, count_by_type_idx,
                 count_elmt_vtx_n_by_type, count_elmt_vtx_n_by_type_idx,
                 er, eso, ec,
                 distrib_elmt,
                 dnew_to_old,
                 concat_elmt_vtx_n,
                 concat_elmt_vtx)

  # > On a plus qu'un seul tableau par rang concatener par type
  #   On repartie la charge pour tout les types d'elements

  print("concat_elmt_vtx   : ", concat_elmt_vtx)
  print("concat_elmt_vtx_n : ", concat_elmt_vtx_n)
  print("dnew_to_old       : ", dnew_to_old)

  separated_section = redistribute_mixed_to_separated(comm,
                                                      count_by_type, count_by_type_idx,
                                                      count_elmt_vtx_n_by_type, count_elmt_vtx_n_by_type_idx,
                                                      g_count_by_type,
                                                      dnew_to_old,
                                                      concat_elmt_vtx_n,
                                                      concat_elmt_vtx)

  for section in separated_section:
    print("separated_section : ", section)

def mixed_elements_to_separated_elements(dist_tree, comm):
  """
  Transform a dist tree with Mixed elements into a tree with separated elements
  This methode change to global ordering of the mesh and reorder it to have one section by element type
  """

  dist_tree_mixed = I.newCGNSTree()

  for base in I.getNodesFromType(dist_tree, 'CGNSBase_t'):
    zones_u = [zone for zone in I.getZones(base) if sids.Zone.Type(zone) == "Unstructured"]
    dist_base_mixed = I.createNode(I.getName(base), 'CGNSBase_t', I.getValue(base), parent=dist_tree_mixed)
    for i_zone, zone in enumerate(zones_u):

      dist_zone_mixed = I.newZone(name  = I.getName(zone),
                                  zsize = [I.getValue(zone)],
                                  ztype = 'Unstructured',
                                  parent = dist_base_mixed)

      mixed_elements_to_separated_elements_zone(zone, dist_zone_mixed, comm)

  return dist_tree_mixed
