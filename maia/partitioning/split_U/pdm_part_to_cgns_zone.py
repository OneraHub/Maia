import Converter.Internal as I
import numpy              as np

from maia.connectivity import connectivity_transform as CNT

def dump_pdm_output(p_zone, dims, data):
  """
  Write PDM output in part_tree (for debug)
  """
  ppart_node = I.createUniqueChild(p_zone, ':CGNS#Ppart', 'UserDefinedData_t')
  for dim_key, dim_val in dims.items():
    I.newDataArray(dim_key, dim_val, parent=ppart_node)
  for data_key, data_val in data.items():
    if isinstance(data_val, np.ndarray):
      I.newDataArray(data_key, np.copy(data_val), parent=ppart_node)

def zgc_created_pdm_to_cgns(p_zone, d_zone, dims, data, grid_loc='FaceCenter', zgc_name='ZoneGridConnectivity'):
  """
  Create by splitting
  """
  if grid_loc not in ['FaceCenter', 'Vertex']:
    raise NotImplementedError("Unvalid specified entity")
  entity = 'face' if grid_loc == 'FaceCenter' else 'vtx'

  entity_part_bound_proc_idx = data['np_{0}_part_bound_proc_idx'.format(entity)]
  entity_part_bound_part_idx = data['np_{0}_part_bound_part_idx'.format(entity)]
  entity_part_bound_tmp      = data['np_{0}_part_bound'         .format(entity)]

  entity_part_bound = entity_part_bound_tmp.reshape((4, entity_part_bound_tmp.shape[0]//4), order='F')
  entity_part_bound = entity_part_bound.transpose()

  zgc_n = I.newZoneGridConnectivity(name=zgc_name, parent=p_zone)

  n_internal_join = entity_part_bound_part_idx.shape[0]-1
  for i_join in range(n_internal_join):

    beg_pl = entity_part_bound_part_idx[i_join  ]
    end_pl = entity_part_bound_part_idx[i_join+1]

    if( beg_pl != end_pl):

      pl_size = end_pl - beg_pl
      pl      = np.empty((1, pl_size), order='F', dtype=np.int32)
      pl[0]   = np.copy(entity_part_bound[beg_pl:end_pl, 0])

      pld    = np.empty((1, pl_size), order='F', dtype=np.int32)
      pld[0] = np.copy(entity_part_bound[beg_pl:end_pl, 3])

      opp_rank = entity_part_bound[beg_pl, 1]
      opp_part = entity_part_bound[beg_pl, 2]-1

      cur_rank = I.getName(p_zone).split('.')[-2][1:]
      cur_part = I.getName(p_zone).split('.')[-1][1:]
      gcname = 'JN.P{0}.N{1}.LT.P{2}.N{3}'.format(cur_rank, cur_part, opp_rank, opp_part, i_join)
      join_n = I.newGridConnectivity(name      = gcname,
                                     donorName = I.getName(d_zone)+'.P{0}.N{1}'.format(opp_rank, opp_part),
                                     ctype     = 'Abutting1to1',
                                     parent    = zgc_n)

      I.newGridLocation(grid_loc, parent=join_n)
      I.newPointList(name='PointList'     , value=pl , parent=join_n)
      I.newPointList(name='PointListDonor', value=pld, parent=join_n)


def pdm_vtx_to_cgns_grid_coordinates(p_zone, dims, data):
  """
  """
  grid_c = I.newGridCoordinates(parent=p_zone)
  I.newDataArray('CoordinateX', data['np_vtx_coord'][0::3], parent=grid_c)
  I.newDataArray('CoordinateY', data['np_vtx_coord'][1::3], parent=grid_c)
  I.newDataArray('CoordinateZ', data['np_vtx_coord'][2::3], parent=grid_c)

def pdm_elmt_to_cgns_elmt(p_zone, d_zone, dims, data):
  """
  """
  if (dims['n_section']==0):
    n_face        = dims['n_face']
    n_cell        = dims['n_cell']
    pdm_face_cell = data['np_face_cell']
    pe = np.empty((n_face, 2), dtype=pdm_face_cell.dtype, order='F')
    CNT.pdm_face_cell_to_pe_cgns(pdm_face_cell, pe)

    ngon_n = I.createUniqueChild(p_zone, 'NGonElements', 'Elements_t', value=[22,0])
    I.newDataArray('ElementConnectivity', data['np_face_vtx']    , parent=ngon_n)
    I.newDataArray('ElementStartOffset' , data['np_face_vtx_idx'], parent=ngon_n)
    I.newDataArray('ParentElements'     , pe                     , parent=ngon_n)
    I.newPointRange('ElementRange'      , [1, n_face]            , parent=ngon_n)
    lngn_elmt = I.createUniqueChild(ngon_n, ':CGNS#GlobalNumbering', 'UserDefinedData_t')
    I.newDataArray('Element', data['np_face_ln_to_gn'], parent=lngn_elmt)

    nface_n = I.createUniqueChild(p_zone, 'NFaceElements', 'Elements_t', value=[23,0])
    I.newDataArray('ElementConnectivity', data['np_cell_face']     , parent=nface_n)
    I.newDataArray('ElementStartOffset' , data['np_cell_face_idx'] , parent=nface_n)
    I.newPointRange('ElementRange'      , [n_face+1, n_face+n_cell], parent=nface_n)
    lngn_elmt = I.createUniqueChild(nface_n, ':CGNS#GlobalNumbering', 'UserDefinedData_t')
    I.newDataArray('Element', data['np_cell_ln_to_gn'], parent=lngn_elmt)

  else:
    elt_section_nodes = I.getNodesFromType(d_zone, "Elements_t")
    assert len(elt_section_nodes) == dims['n_section']

    n_elt_cum = 0
    for i_elt, elt_section_node in enumerate(elt_section_nodes):
      elt_name = I.getName(elt_section_node)
      elt_type = I.getValue(elt_section_node)[0]
      n_i_elt    = dims['n_elt'][i_elt]
      elt_n = I.createUniqueChild(p_zone, elt_name, 'Elements_t', value=[elt_type,0])
      I.newDataArray('ElementConnectivity', data['np_elt_vtx'][i_elt]       , parent=elt_n)
      I.newPointRange('ElementRange'      , [n_elt_cum+1, n_elt_cum+n_i_elt], parent=elt_n)
      n_elt_cum += n_i_elt
      lngn_elmt = I.createUniqueChild(elt_n, ':CGNS#GlobalNumbering', 'UserDefinedData_t')
      I.newDataArray('Element', data['np_elt_section_ln_to_gn'][i_elt], parent=lngn_elmt)


def pdm_part_to_cgns_zone(dist_zone, l_dims, l_data, comm, options):
  """
  """
  #Dims and data should be related to the dist zone and of size n_parts
  part_zones = list()
  for i_part, (dims, data) in enumerate(zip(l_dims, l_data)):

    part_zone = I.newZone(name  = '{0}.P{1}.N{2}'.format(I.getName(dist_zone), comm.Get_rank(), i_part),
                          zsize = [[dims['n_vtx'],dims['n_cell'],0]],
                          ztype = 'Unstructured')

    dump_pdm_output(part_zone, dims, data)
    pdm_vtx_to_cgns_grid_coordinates(part_zone, dims, data)
    pdm_elmt_to_cgns_elmt(part_zone, dist_zone, dims, data)

    output_loc = options['part_interface_loc']
    zgc_name = 'ZoneGridConnectivity#Vertex' if output_loc == 'Vertex' else 'ZoneGridConnectivity'
    zgc_created_pdm_to_cgns(part_zone, dist_zone, dims, data, output_loc, zgc_name)

    if options['save_ghost_data']:
      vtx_ghost_info = data['np_vtx_ghost_information']
      if vtx_ghost_info is not None:
        first_ghost_idx = np.searchsorted(vtx_ghost_info, 2)
        n_ghost_node = len(vtx_ghost_info) - first_ghost_idx
        coord_node   = I.getNodeFromName(part_zone, "GridCoordinates")
        I.newUserDefinedData("FSDM#n_ghost", value=[n_ghost_node], parent=coord_node)


    lngn_zone = I.createUniqueChild(part_zone, ':CGNS#GlobalNumbering', 'UserDefinedData_t')
    I.newDataArray('Vertex', data['np_vtx_ln_to_gn'], parent=lngn_zone)
    I.newDataArray('Cell', data['np_cell_ln_to_gn'], parent=lngn_zone)

    part_zones.append(part_zone)

  return part_zones