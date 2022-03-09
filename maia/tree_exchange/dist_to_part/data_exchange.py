import numpy              as np
import Converter.Internal as I
import Pypdm.Pypdm        as PDM

import maia.sids.sids     as SIDS
import maia.sids.Internal_ext as IE
from maia.utils.parallel import utils as par_utils
from maia.tree_exchange import utils as te_utils
from maia.sids import pytree as PT

def dist_to_part(partial_distri, dist_data, ln_to_gn_list, comm):
  """
  Helper function calling PDM.BlockToPart
  """
  pdm_distrib = par_utils.partial_to_full_distribution(partial_distri, comm)

  part_data = dict()
  for data_name in dist_data:
    npy_type = dist_data[data_name].dtype
    part_data[data_name] = [np.empty(ln_to_gn.shape[0], dtype=npy_type) for ln_to_gn in ln_to_gn_list]

  BTP = PDM.BlockToPart(pdm_distrib, comm, ln_to_gn_list, len(ln_to_gn_list))
  BTP.BlockToPart_Exchange(dist_data, part_data)

  return part_data

def dist_coords_to_part_coords(dist_zone, part_zones, comm):
  """
  Transfert all the data included in GridCoordinates_t nodes from a distributed
  zone to the partitioned zones
  """
  #Get distribution
  distribution_vtx = te_utils.get_cgns_distribution(dist_zone, 'Vertex')

  #Get data
  dist_data = dict()
  dist_gc = I.getNodeFromType1(dist_zone, "GridCoordinates_t")
  for grid_co in I.getNodesFromType1(dist_gc, 'DataArray_t'):
    dist_data[I.getName(grid_co)] = grid_co[1] #Prevent np->scalar conversion


  vtx_lntogn_list = te_utils.collect_cgns_g_numbering(part_zones, 'Vertex')
  part_data = dist_to_part(distribution_vtx, dist_data, vtx_lntogn_list, comm)

  for ipart, part_zone in enumerate(part_zones):
    part_gc = I.newGridCoordinates(parent=part_zone)
    for data_name, data in part_data.items():
      #F is mandatory to keep shared reference. Normally no copy is done
      shaped_data = data[ipart].reshape(SIDS.Zone.VertexSize(part_zone), order='F')
      I.newDataArray(data_name, shaped_data, parent=part_gc)



def _dist_to_part_sollike(dist_zone, part_zones, mask_tree, comm):
  """
  Shared code for FlowSolution_t and DiscreteData_t
  """
  #Get distribution
  for mask_sol in I.getChildren(mask_tree):
    d_sol = I.getNodeFromName1(dist_zone, I.getName(mask_sol)) #True container
    location = SIDS.GridLocation(d_sol)
    has_pl   = I.getNodeFromName1(d_sol, 'PointList') is not None
    if has_pl:
      distribution = te_utils.get_cgns_distribution(d_sol, 'Index')
      lntogn_list  = te_utils.collect_cgns_g_numbering(part_zones, 'Index', I.getName(d_sol))
    else:
      assert location in ['Vertex', 'CellCenter']
      if location == 'Vertex':
        distribution = te_utils.get_cgns_distribution(dist_zone, 'Vertex')
        lntogn_list  = te_utils.collect_cgns_g_numbering(part_zones, 'Vertex')
      elif location == 'CellCenter':
        distribution = te_utils.get_cgns_distribution(dist_zone, 'Cell')
        lntogn_list  = te_utils.collect_cgns_g_numbering(part_zones, 'Cell')

    #Get data
    fields = [I.getName(n) for n in I.getChildren(mask_sol)]
    dist_data = {field : I.getNodeFromName1(d_sol, field)[1] for field in fields}

    #Exchange
    part_data = dist_to_part(distribution, dist_data, lntogn_list, comm)

    for ipart, part_zone in enumerate(part_zones):
      #Skip void flow solution (can occur with point lists)
      if lntogn_list[ipart].size > 0:
        if has_pl:
          p_sol = I.getNodeFromName1(part_zone, I.getName(d_sol))
          shape = I.getNodeFromName1(p_sol, 'PointList')[1].shape[1]
        else:
          p_sol = I.createChild(part_zone, I.getName(d_sol), I.getType(d_sol))
          I.newGridLocation(location, parent=p_sol)
          shape = SIDS.Zone.VertexSize(part_zone) if location == 'Vertex' else SIDS.Zone.CellSize(part_zone)
        for data_name, data in part_data.items():
          #F is mandatory to keep shared reference. Normally no copy is done
          shaped_data = data[ipart].reshape(shape, order='F')
          I.newDataArray(data_name, shaped_data, parent=p_sol)

def dist_sol_to_part_sol(dist_zone, part_zones, comm, include=[], exclude=[]):
  """
  Transfert all the data included in FlowSolution_t nodes from a distributed
  zone to the partitioned zones
  """
  mask_tree = create_mask_tree(dist_zone, ['FlowSolution_t', 'DataArray_t'], include, exclude)
  _dist_to_part_sollike(dist_zone, part_zones, mask_tree, comm)

def dist_discdata_to_part_discdata(dist_zone, part_zones, comm, include=[], exclude=[]):
  """
  Transfert all the data included in DiscreteData_t nodes from a distributed
  zone to the partitioned zones
  """
  mask_tree = create_mask_tree(dist_zone, ['DiscreteData_t', 'DataArray_t'], include, exclude)
  _dist_to_part_sollike(dist_zone, part_zones, mask_tree, comm)

def pathes_to_tree(pathes, root_name='CGNSTree'):
  """
  Convert a list of pathes to a CGNSTreeLike
  """
  def unroll(root):
    """ Internal recursive function """
    if root[1] is None:
      return
    for path in root[1]:
      first = path.split('/')[0]
      others = '/'.join(path.split('/')[1:])
      node = I.getNodeFromName1(root, first)
      if node is None:
        if others:
          root[2].append([first, [others], [], None])
        else:
          root[2].append([first, None, [], None])
      else:
        node[1].append(others)
    root[1] = None
    for child in root[2]:
      unroll(child)

  path_tree = [root_name, pathes, [], None]
  unroll(path_tree)
  return path_tree

def predicates_to_pathes(root, predicates):
  """
  An utility function searching node descendance from root matching given predicates,
  and returning the path of these nodes (instead of the nodes itselves)
  """
  pathes = []
  for nodes in PT.iter_nodes_from_predicates(root, predicates, depth=[1,1], ancestors=True):
    pathes.append('/'.join([I.getName(n) for n in nodes]))
  return pathes

def concretise_pathes(root, wanted_path_list, labels):
  """
  """
  all_pathes = []
  for path in wanted_path_list:
    names = path.split('/')
    assert len(names) == len(labels)
    predicates = [lambda n, _name=name, _label=label: PT.match_name_label(n, _name, _label) \
        for (name, label) in zip(names,labels)] 
    pathes = predicates_to_pathes(root, predicates)
    all_pathes.extend(pathes)

  return sorted(list(set(all_pathes))) #Unique + sort

def create_mask_tree(root, labels, include, exclude):
  """
  Create a mask tree from root using either the include or exclude list + hints on searched labels
  """
  if len(include) * len(exclude) != 0:
    raise ValueError("Include and exclude args are mutally exclusive")

  if len(include) > 0:
    to_include = concretise_pathes(root, include, labels)
  elif len(exclude) > 0:
    #In exclusion mode, we get all the pathes matching labels and exclude the one founded
    all_pathes = predicates_to_pathes(root, labels)
    to_exclude = concretise_pathes(root, exclude, labels)
    to_include = [p for p in all_pathes if not p in to_exclude]
  else:
    to_include = predicates_to_pathes(root, labels)

  return pathes_to_tree(to_include, I.getName(root))

def queries_to_pathes(root, query_list):
  """OLD"""
  def explore(root, i, queries, path='', pp=0):
    try:
      child_label = labels[i]
      #print(2*pp*' ', "Exploring", root[0], "searching for labels", child_label)
      for child in I.getNodesFromType1(root, child_label):
        next_queries = [data_n for (container_n, *data_n) in queries if PT.match_name(child, container_n)]
        # print(2*pp*' ', "For child", child[0], "next queries", next_queries)
        explore(child, i+1, next_queries, f'{path}/{I.getName(child)}', pp+1)
    except IndexError:
      #print(2*pp*' ', "Exploring last level", root[0], [path+'/'+q[0] for q in queries])
      all_path.extend( [f'{path}/{q[0]}' for q in queries])

  all_path = []
  labels = ["BC_t", "BCDataSet_t", "BCData_t"]
  start_query = [query.split('/') for query in include]
  if root is not None:
    explore(root, 0, start_query)
    all_path = [path[1:] for path in all_path] #Remove first "/"

  return all_path


def dist_dataset_to_part_dataset(dist_zone, part_zones, comm, include=[], exclude=[]):
  """
  Transfert all the data included in BCDataSet_t/BCData_t nodes from a distributed
  zone to the partitioned zones
  """
  for d_zbc in I.getNodesFromType1(dist_zone, "ZoneBC_t"):
    labels = ['BC_t', 'BCDataSet_t', 'BCData_t', 'DataArray_t']
    mask_tree = create_mask_tree(d_zbc, labels, include, exclude)
    for mask_bc in I.getChildren(mask_tree):
      bc_path = I.getName(d_zbc) + '/' + I.getName(mask_bc)
      d_bc = I.getNodeFromPath(dist_zone, bc_path) #True BC
      for mask_dataset in I.getChildren(mask_bc):
        ds_path = bc_path + '/' + I.getName(mask_dataset)
        d_dataset = I.getNodeFromPath(dist_zone, ds_path) #True DataSet
        #If dataset has its own PointList, we must override bc distribution and lngn
        if IE.getDistribution(d_dataset) is not None:
          distribution = te_utils.get_cgns_distribution(d_dataset, 'Index')
          lngn_list    = te_utils.collect_cgns_g_numbering(part_zones, 'Index', ds_path)
        else: #Fallback to bc distribution
          distribution = te_utils.get_cgns_distribution(d_bc, 'Index')
          lngn_list    = te_utils.collect_cgns_g_numbering(part_zones, 'Index', bc_path)
        #Get data
        data_pathes = predicates_to_pathes(mask_dataset, ['*', '*'])
        dist_data = {data_path : I.getNodeFromPath(d_dataset, data_path)[1] for data_path in data_pathes}

        #Exchange
        part_data = dist_to_part(distribution, dist_data, lngn_list, comm)

        #Put part data in tree
        for ipart, part_zone in enumerate(part_zones):
          part_bc = I.getNodeFromPath(part_zone, bc_path)
          # Skip void bcs
          if lngn_list[ipart].size > 0:
            # Create dataset if no existing
            part_ds = I.createUniqueChild(part_bc, I.getName(d_dataset), I.getType(d_dataset), I.getValue(d_dataset))
            # Add data
            for data_name, data in part_data.items():
              container_name, field_name = data_name.split('/')
              p_container = I.createUniqueChild(part_ds, container_name, 'BCData_t')
              I.newDataArray(field_name, data[ipart], parent=p_container)


def dist_subregion_to_part_subregion(dist_zone, part_zones, comm, include=[], exclude=[]):
  """
  Transfert all the data included in ZoneSubRegion_t nodes from a distributed
  zone to the partitioned zones
  """
  mask_tree = create_mask_tree(dist_zone, ['ZoneSubRegion_t', 'DataArray_t'], include, exclude)
  for mask_zsr in I.getChildren(mask_tree):
    d_zsr = I.getNodeFromName1(dist_zone, I.getName(mask_zsr)) #True ZSR
    # Search matching region
    matching_region_path = IE.getSubregionExtent(d_zsr, dist_zone)
    matching_region = I.getNodeFromPath(dist_zone, matching_region_path)
    assert matching_region is not None

    #Get distribution and lngn
    distribution = te_utils.get_cgns_distribution(matching_region, 'Index')
    lngn_list    = te_utils.collect_cgns_g_numbering(part_zones, 'Index', matching_region_path)

    #Get Data
    fields = [I.getName(n) for n in I.getChildren(mask_zsr)]
    dist_data = {field : I.getNodeFromName1(d_zsr, field)[1] for field in fields}

    #Exchange
    part_data = dist_to_part(distribution, dist_data, lngn_list, comm)

    #Put part data in tree
    for ipart, part_zone in enumerate(part_zones):
      # Skip void zsr
      if lngn_list[ipart].size > 0:
        # Create ZSR if not existing (eg was defined by bc/gc)
        p_zsr = I.createUniqueChild(part_zone, I.getName(d_zsr), I.getType(d_zsr), I.getValue(d_zsr))
        for field_name, data in part_data.items():
          I.newDataArray(field_name, data[ipart], parent=p_zsr)
