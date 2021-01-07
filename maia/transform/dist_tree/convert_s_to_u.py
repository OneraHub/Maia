#coding:utf-8
import Converter.Internal as I
import numpy              as np
import copy

import Pypdm.Pypdm as PDM

from maia.distribution       import distribution_function           as MDIDF
from maia.cgns_io.hdf_filter import range_to_slab                   as HFR2S

###############################################################################
def convert_ijk_to_index(i,j,k,Ni,Nj,Nk):
  """
  Convert (i,j,k) indices from structured grid to unstructured index
  This fonction allows (i,j,k) that defines node or cell
  Ni (resp. j,k) is the number of nodes or cells in the direction i (resp. j,k)
  WARNING : (i,j,k) and index begins at 1
  """
  return(i+(j-1)*Ni+(k-1)*Ni*Nj)
###############################################################################

###############################################################################
def convert_ijk_to_faceiIndex(i,j,k,nCell,nVtx):
  """
  Convert (i,j,k) indices from structured grid to unstructured index of face with
  normal in direction i
  (i,j,k) defines start node of the structured face defined by :
  (i,j,k), (i,j+1,k), (i,j+1,k+1) and (i,j,k+1)
  nCell = [Ni,Nj,Nk]
  nVtx  = [Ni,Nj,Nk]
  Ni (resp. j,k) is the number of nodes or cells in the direction i (resp. j,k)
  WARNING : (i,j,k) and index begins at 1
  CONVENTION : the face numerotation starts with all faces with normal in direction 
               i, then continues with all faces with normal in direction j and ends 
               with all faces with normal in direction k
  """
  return(i + (j-1)*nVtx[0] + (k-1)*nVtx[0]*nCell[1])
###############################################################################

###############################################################################
def convert_ijk_to_facejIndex(i,j,k,nCell,nVtx):
  """
  Convert (i,j,k) indices from structured grid to unstructured index of face with
  normal in direction j
  (i,j,k) defines start node of the structured face defined by :
  (i,j,k), (i,j,k+1), (i+1,j,k+1) and (i+1,j,k)
  nCell = [Ni,Nj,Nk]
  nVtx  = [Ni,Nj,Nk]
  Ni (resp. j,k) is the number of nodes or cells in the direction i (resp. j,k)
  WARNING : (i,j,k) and index begins at 1
  CONVENTION : the face numerotation starts with all faces with normal in direction 
               i, then continues with all faces with normal in direction j and ends 
               with all faces with normal in direction k
  nbFacesi is the total number of faces with normal in direction i
  """
  nbFacesi = nVtx[0]*nCell[1]*nCell[2]
  return(i + (j-1)*nCell[0] + (k-1)*nVtx[1]*nCell[0] + nbFacesi)
###############################################################################

###############################################################################
def convert_ijk_to_facekIndex(i,j,k,nCell,nVtx):
  """
  Convert (i,j,k) indices from structured grid to unstructured index of face with
  normal in direction k
  (i,j,k) defines start node of the structured face defined by :
  (i,j,k), (i+1,j,k), (i+1,j+1,k) and (i,j+1,k)
  nCell = [Ni,Nj,Nk]
  nVtx  = [Ni,Nj,Nk]
  Ni (resp. j,k) is the number of nodes or cells in the direction i (resp. j,k)
  WARNING : (i,j,k) and index begins at 1
  CONVENTION : the face numerotation starts with all faces with normal in direction 
               i, then continues with all faces with normal in direction j and ends 
               with all faces with normal in direction k
  nbFacesi (resp. j) is the total number of faces with normal in direction i (resp. j)
  """
  nbFacesi = nVtx[0]*nCell[1]*nCell[2]
  nbFacesj = nVtx[1]*nCell[0]*nCell[2]
  return(i + (j-1)*nCell[0] + (k-1)*nCell[0]*nCell[1] + nbFacesi + nbFacesj)
###############################################################################

###############################################################################
def convert_ijk_to_faceIndices(i,j,k,nCell,nVtx):
  """
  Convert (i,j,k) indices from structured grid to unstructured index of faces with
  normal in direction i, j and k
  (i,j,k) defines start node of the structured face defined by :
  (i,j,k), (i,j+1,k), (i,j+1,k+1) and (i,j,k+1) for fi
  (i,j,k), (i,j,k+1), (i+1,j,k+1) and (i+1,j,k) for fj
  (i,j,k), (i+1,j,k), (i+1,j+1,k) and (i,j+1,k) for fk
  nCell = [Ni,Nj,Nk]
  nVtx  = [Ni,Nj,Nk]
  Ni (resp. j,k) is the number of nodes or cells in the direction i (resp. j,k)
  WARNING : (i,j,k) and index begins at 1
  CONVENTION : the face numerotation starts with all faces with normal in direction 
               i, then continues with all faces with normal in direction j and ends 
               with all faces with normal in direction k
  """
  fi = convert_ijk_to_faceiIndex(i,j,k,nCell,nVtx)
  fj = convert_ijk_to_facejIndex(i,j,k,nCell,nVtx)
  fk = convert_ijk_to_facekIndex(i,j,k,nCell,nVtx)
  return(fi,fj,fk)
###############################################################################

###############################################################################
def compute_fi_from_ijk(i,j,k, is_min=False, is_max=False):
  """
  Compute from structured indices (i,j,k) of structured nodes indices that compose 
  face with normal in direction i and structured left and right cells indices of 
  this face
  (i,j,k) defines start node of the structured face defined by :
  (i,j,k), (i,j+1,k), (i,j+1,k+1) and (i,j,k+1) if i is not min
  (i,j,k), (i,j,k+1), (i,j+1,k+1) and (i,j+1,k) if i is min
  WARNING : (i,j,k) begins at (1,1,1)
  """
  # Nodes of the face
  n1 = (i,j  ,k  )
  n2 = (i,j+1,k  )
  n3 = (i,j+1,k+1)
  n4 = (i,j  ,k+1)
  # Neighbour cells of the face
  left  = (i-1,j,k)
  right = (i  ,j,k)
  if is_min:
    n2, n4 = n4, n2
    left = (i,j,k)
  if is_min or is_max:
    right = 0
  return(n1,n2,n3,n4,left,right)
###############################################################################

###############################################################################
def compute_fj_from_ijk(i,j,k, is_min=False, is_max=False):
  """
  Compute from structured indices (i,j,k) of structured nodes indices that compose 
  face with normal in direction j and structured left and right cells indices of 
  this face
  (i,j,k) defines start node of the structured face defined by :
  (i,j,k), (i,j,k+1), (i+1,j,k+1) and (i+1,j,k) if j is not min
  (i,j,k), (i+1,j,k), (i+1,j,k+1) and (i,j,k+1) if j is min
  WARNING : (i,j,k) begins at (1,1,1)
  """
  # Nodes of the face
  n1 = (i  ,j,k  )
  n2 = (i  ,j,k+1)
  n3 = (i+1,j,k+1)
  n4 = (i+1,j,k  )
  # Neighbour cells of the face
  left  = (i,j-1,k)
  right = (i,j  ,k)
  if is_min:
    n2, n4 = n4, n2
    left = (i,j,k)
  if is_min or is_max:
    right = 0
  return(n1,n2,n3,n4,left,right)
###############################################################################

###############################################################################
def compute_fk_from_ijk(i,j,k, is_min=False, is_max=False):
  """
  Compute from structured indices (i,j,k) of structured nodes indices that compose 
  face with normal in direction k and structured left and right cells indices of 
  this face
  (i,j,k) defines start node of the structured face defined by :
  (i,j,k), (i+1,j,k), (i+1,j+1,k) and (i,j+1,k) if k is not min
  (i,j,k), (i,j+1,k), (i+1,j+1,k) and (i+1,j,k) if k is min
  WARNING : (i,j,k) begins at (1,1,1)
  """
  # Nodes of the face
  n1 = (i  ,j  ,k)
  n2 = (i+1,j  ,k)
  n3 = (i+1,j+1,k)
  n4 = (i  ,j+1,k)
  # Neighbour cells of the face
  left  = (i,j,k-1)
  right = (i,j,k  )
  if is_min:
    n2, n4 = n4, n2
    left = (i,j,k)
  if is_min or is_max:
    right = 0
  return(n1,n2,n3,n4,left,right)
###############################################################################

###############################################################################
def fill_faceNgon_leftCell_rightCell(counter,n1ijk,n2ijk,n3ijk,n4ijk,
                                     leftijk,rightijk,nVtx,nCell,
                                     faceNgon,faceLeftCell,faceRightCell):
  """
  Convert to unstructured indices for nodes and cells for a face and fill associated
  tabs :
  faceNgon refers to non sorted NGonConnectivity
  faceLeftCell refers to non sorted ParentElement[:][0]
  faceRightCell refers to non sorted ParentElement[:][1]
  WARNING : (i,j,k) begins at (1,1,1)
  """
  # Convert (i,j,k) structured indices of node to unstructured indices
  n1 = convert_ijk_to_index(n1ijk[0],n1ijk[1],n1ijk[2],nVtx[0],nVtx[1],nVtx[2])
  n2 = convert_ijk_to_index(n2ijk[0],n2ijk[1],n2ijk[2],nVtx[0],nVtx[1],nVtx[2])
  n3 = convert_ijk_to_index(n3ijk[0],n3ijk[1],n3ijk[2],nVtx[0],nVtx[1],nVtx[2])
  n4 = convert_ijk_to_index(n4ijk[0],n4ijk[1],n4ijk[2],nVtx[0],nVtx[1],nVtx[2])
  # Fill NGon connectivity with nodes
  faceNgon[4*counter:4*(counter+1)] = [n1,n2,n3,n4]
  # Convert leftt structured cell (i,j,k) to unstructured index
  left = convert_ijk_to_index(leftijk[0],leftijk[1],leftijk[2],nCell[0],nCell[1],nCell[2])
  # Fill LeftCell with nodes (LeftCell equal ParentElement[:][0])
  faceLeftCell[counter] = left
  # Convert right structured cell (i,j,k) to unstructured index
  if rightijk == 0:
    right = 0
  else:
    right = convert_ijk_to_index(rightijk[0],rightijk[1],rightijk[2],nCell[0],nCell[1],nCell[2])
  # Fill RightCell with nodes (RightCell equal ParentElement[:][1])
  faceRightCell[counter] = right    
###############################################################################

###############################################################################
def vtx_slab_to_n_face(vtx_slab, n_vtx):
  """
  Compute the number of faces to create for a zone by a proc with distributed info
  from a vertex slab
  """
  iS,iE, jS,jE, kS,kE = [item for bounds in vtx_slab for item in bounds]
  # Number of vertices of the slab in each direction
  nx = iE - iS
  ny = jE - jS
  nz = kE - kS

  # Number of edges of the slab in each direction : exclude last edge if slab
  # is the end of the block
  ex = nx - 1 if iE == n_vtx[0] else nx
  ey = ny - 1 if jE == n_vtx[1] else ny
  ez = nz - 1 if kE == n_vtx[2] else nz

  # In each direction, number of faces is n_vtx * n_edge1 * n_edge2
  return nx*ey*ez + ny*ex*ez + nz*ex*ey

###############################################################################

###############################################################################
def compute_all_ngon_connectivity(slabListVtx,nVtx,nCell,
                                  faceNumber,faceNgon,
                                  faceLeftCell,faceRightCell):
  """
  Compute the numerotation, the nodes and the cells linked to all face traited for
  zone by a proc and fill associated tabs :
  faceNumber refers to non sorted numerotation of each face
  faceNgon refers to non sorted NGonConnectivity
  faceLeftCell refers to non sorted ParentElement[:][0]
  faceRightCell refers to non sorted ParentElement[:][1]
  Remark : all tabs are defined in the same way i.e. for the fth face, information are
  located in faceNumber[f], faceNgon[4*f:4*(f+1)], faceLeftCell[f] and faceRightCell[f]
  WARNING : (i,j,k) begins at (1,1,1)
  """
  counter = 0
  for slabVtx in slabListVtx:
    iS,iE, jS,jE, kS,kE = [item+1 for bounds in slabVtx for item in bounds]
      
    for i in range(iS, iE):
      for j in range(jS, jE):
        for k in range(kS, kE):
          if (j != nVtx[1] and k != nVtx[2]):
            faceNumber[counter] = convert_ijk_to_faceiIndex(i,j,k,nCell,nVtx)
            (n1ijk,n2ijk,n3ijk,n4ijk,leftijk,rightijk) = compute_fi_from_ijk(i,j,k, i==1, i==nVtx[0])
            fill_faceNgon_leftCell_rightCell(counter,n1ijk,n2ijk,n3ijk,n4ijk,
                                             leftijk,rightijk,nVtx,nCell,
                                             faceNgon,faceLeftCell,faceRightCell)
            counter += 1
          if (i != nVtx[0] and k != nVtx[2]):
            faceNumber[counter] = convert_ijk_to_facejIndex(i,j,k,nCell,nVtx)
            (n1ijk,n2ijk,n3ijk,n4ijk,leftijk,rightijk) = compute_fj_from_ijk(i,j,k, j==1, j==nVtx[1])
            fill_faceNgon_leftCell_rightCell(counter,n1ijk,n2ijk,n3ijk,n4ijk,
                                             leftijk,rightijk,nVtx,nCell,
                                             faceNgon,faceLeftCell,faceRightCell)
            counter += 1
          if (i != nVtx[0] and j != nVtx[1]):
            faceNumber[counter] = convert_ijk_to_facekIndex(i,j,k,nCell,nVtx)
            (n1ijk,n2ijk,n3ijk,n4ijk,leftijk,rightijk) = compute_fk_from_ijk(i,j,k, k==1, k==nVtx[2])
            fill_faceNgon_leftCell_rightCell(counter,n1ijk,n2ijk,n3ijk,n4ijk,
                                             leftijk,rightijk,nVtx,nCell,
                                             faceNgon,faceLeftCell,faceRightCell)
            counter += 1

###############################################################################

###############################################################################
def compute_pointList_from_vertexRange(pointRange, sub_pr_list, nVtxS, output_loc):
  """
  Transform structured PointRange with 'GridLocation'='Vertex' to unstructured
  PointList with 'GridLocation'='FaceCenter', 'Vertex' or 'CellCenter'
  """

  nCellS = [nv - 1 for nv in nVtxS]
  #Find constant direction
  cst_axes = np.nonzero(pointRange[:,0] == pointRange[:,1])[0]
  if len(cst_axes) != 1:
    raise ValueError("The PointRange '{}' is bad defined".format(pointRange))
  cst_axe = cst_axes[0]

  # The lambda func ijk_to_func redirect to the good indexing function depending
  # on the output grid location
  convert_ijk_to_faceIndex = [convert_ijk_to_faceiIndex, convert_ijk_to_facejIndex, convert_ijk_to_facekIndex]
  if output_loc == 'FaceCenter':
    ijk_to_func = lambda i,j,k : convert_ijk_to_faceIndex[cst_axe](i, j, k, nCellS, nVtxS)
  elif output_loc == 'Vertex':
    ijk_to_func = lambda i,j,k : convert_ijk_to_index(i, j, k, *nVtxS)
  elif output_loc == 'CellCenter':
    ijk_to_func = lambda i,j,k : convert_ijk_to_index(i, j, k, *nCellS)
  else:
    raise ValueError("Wrong output location : '{}'".format(output_loc))

  # The lambda func ijk_to_vect_func is a wrapping to ijk_to_func (and so to the good indexing func)
  # but with args expressed as numpy arrays : this allow vectorial call of indexing function as if we did an
  # imbricated loop
  ijk_to_vect_func = lambda i_idx, j_idx, k_idx : ijk_to_func(i_idx, j_idx.reshape(-1,1), k_idx.reshape(-1,1,1))

  sub_range_sizes = [(np.abs(pr[:,1] - pr[:,0]) + 1).prod() for pr in sub_pr_list]
  pointList = np.empty((1, sum(sub_range_sizes)), dtype=np.int32)
  counter = 0

  for ipr, pr in enumerate(sub_pr_list):
    inc = 2*(pr[:,0] <= pr[:,1]).astype(int) - 1 #In each direction, 1 if pr[l,0] <= pr[l,1] else - 1

    # Here we build for each direction a looping array range(start, end+1) if pr is increasing
    # or range(start, end-1, -1) if pr is decreasing
    np_idx_arrays = []
    for l in range(pr.shape[0]):
      np_idx_arrays.append(np.arange(pr[l,0], pr[l,1] + inc[l], inc[l]))

    pointList[0][counter:counter+sub_range_sizes[ipr]] = ijk_to_vect_func(*np_idx_arrays).flatten()
    counter += sub_range_sizes[ipr]

  return pointList
###############################################################################

###############################################################################
def compute_faceList_from_faceRange(pointRange, vtx_slabs, nVtxS, gridLocationS):
  """
  Transform structured PointRange with 'GridLocation'='IFaceCenter' or 'JFaceCenter' or
  'KFaceCenter' to unstructured PointList with 'GridLocation'='FaceCenter'
  """
  nCellS = [nv - 1 for nv in nVtxS]
  # Prepare lambda func depending of const idx -> this allow vectorial call of
  # convert_ijk_to_faceLIndex as if we did double for loop
  if gridLocationS == 'IFaceCenter':
    ijk_to_faceidx = lambda i_idx, j_idx, k_idx : \
        convert_ijk_to_faceiIndex(pointRange[0,0], j_idx, k_idx.reshape(-1,1), nCellS, nVtxS)
  elif gridLocationS == 'JFaceCenter':
    ijk_to_faceidx = lambda i_idx, j_idx, k_idx : \
        convert_ijk_to_facejIndex(i_idx, pointRange[1,0], k_idx.reshape(-1,1), nCellS, nVtxS)
  elif gridLocationS == 'KFaceCenter':
    ijk_to_faceidx = lambda i_idx, j_idx, k_idx : \
        convert_ijk_to_facekIndex(i_idx, j_idx.reshape(-1,1), pointRange[2,0], nCellS, nVtxS)
  else:
    raise ValueError("The GridLocation '{}' is bad defined".format(gridLocationS))

  sizeU = sum([np.prod([d[1] - d[0] for d in slab]) for slab in vtx_slabs])
  faceList = np.empty((1,sizeU), dtype=np.int32)
  counter = 0

  for slabS in vtx_slabs:
    iS,iE, jS,jE, kS,kE = [item+1 for bounds in slabS for item in bounds]
    n_faces = (iE-iS)*(jE-jS)*(kE-kS)
    faceList[0][counter:counter+n_faces] = ijk_to_faceidx(
        np.arange(iS, iE), np.arange(jS, jE), np.arange(kS, kE)).flatten()
    counter += n_faces

  return faceList
###############################################################################

###############################################################################
def isSameAxis(x,y):
  """
  This function is the implementation of the 'del' function defined in the SIDS
  of CGNS (https://cgns.github.io/CGNS_docs_current/sids/cnct.html) as :
  del(x−y) ≡ +1 if |x| = |y|
  """
  return (np.abs(x) == np.abs(y)).astype(int)
###############################################################################

###############################################################################
def compute_transformMatrix(transform):
  """
  This function compute the matrix to convert current indices to opposit indices
  The definition of this matrix is given in the SIDS of CGNS 
  (https://cgns.github.io/CGNS_docs_current/sids/cnct.html)
  """
  transform_np = np.asarray(transform)
  del_matrix = isSameAxis(transform_np, np.array([[1],[2],[3]]))
  return np.sign(transform_np) * del_matrix
###############################################################################

###############################################################################
def apply_transformation(index1, start1, start2, T):
  """
  This function compute indices from current to oppposit or from opposit to current
  by using the transform matrix.
  As defined in the SIDS of CGNS (https://cgns.github.io/CGNS_docs_current/sids/cnct.html) :
  Index2 = T.(Index1 - Begin1) + Begin2
  """
  return np.matmul(T, (index1 - start1)) + start2
###############################################################################

def fix_cell_point_ranges(pointRange, nCell, point_range_list):
  """
  Correct the cells sub pointranges for BC/GC (shift max cell if needed)
  """
  cst_axes = np.nonzero(pointRange[:,0] == pointRange[:,1])[0]
  if len(cst_axes) != 1:
    raise ValueError("The PointRange '{}' is bad defined".format(pointRange))
  cst_axe = cst_axes[0]

  if pointRange[cst_axe,0] > nCell[cst_axe]:
    for sub_pr in point_range_list:
      sub_pr[cst_axe,:] -= 1


def transform_bnd_pr_size(point_range, input_loc, output_loc):
  """
  Predict a point_range defined at an input_location if it were defined at an output_location
  """
  size = np.abs(point_range[:,1] - point_range[:,0]) + 1

  if input_loc == 'Vertex' and 'Center' in output_loc:
    size -= (size != 1).astype(int)
  elif 'Center' in input_loc and output_loc == 'Vertex':
    if input_loc in ['IFaceCenter', 'JFaceCenter', 'KFaceCenter']:
      cst_axe = {'I':0, 'J':1, 'K':2}[input_loc[0]]
    elif sum(point_range[:,0] == point_range[:,1]) == 1: #Ambiguity can be resolved
      cst_axe = np.nonzero(point_range[:,0] == point_range[:,1])[0][0]
    else:
      raise ValueError("Ambiguous input location")
    mask = np.zeros(point_range.shape[0], dtype=np.bool)
    mask[cst_axe] = True
    size += (~mask).astype(int)
  return size

###############################################################################
def convert_s_to_u(distTreeS,comm,attendedGridLocationBC="FaceCenter",attendedGridLocationGC="FaceCenter"):

  nRank = comm.Get_size()
  iRank = comm.Get_rank()

  #> Create skeleton of distTreeU
  distTreeU = I.newCGNSTree()
  baseS = I.getNodeFromType1(distTreeS, 'CGNSBase_t')
  baseU = I.createNode(I.getName(baseS), 'CGNSBase_t', I.getValue(baseS), parent=distTreeU)
  for zoneS in I.getZones(distTreeS):
    zoneSName = I.getName(zoneS)
    zoneSDims = I.getValue(zoneS)
    nCellS = zoneSDims[:,1]
    nVtxS  = zoneSDims[:,0]
    nCellTotS = nCellS.prod()
    nVtxTotS  = nVtxS.prod()
  
    #> Calcul du nombre faces totales en i, j et k
    nbFacesi = nVtxS[0]*nCellS[1]*nCellS[2]
    nbFacesj = nVtxS[1]*nCellS[0]*nCellS[2]
    nbFacesk = nVtxS[2]*nCellS[0]*nCellS[1]
    nbFacesTot = nbFacesi + nbFacesj + nbFacesk
  
    #> with Zones
    zoneU = I.newZone(zoneSName, [[nVtxTotS, nCellTotS, 0]], 'Unstructured', None, baseU)
  
    #> with GridCoordinates
    gridCoordinatesS = I.getNodeFromType1(zoneS, "GridCoordinates_t")
    CoordinateXS = I.getNodeFromName1(gridCoordinatesS, "CoordinateX")
    CoordinateYS = I.getNodeFromName1(gridCoordinatesS, "CoordinateY")
    CoordinateZS = I.getNodeFromName1(gridCoordinatesS, "CoordinateZ")
    gridCoordinatesU = I.newGridCoordinates(parent=zoneU)
    I.newDataArray('CoordinateX', I.getValue(CoordinateXS), gridCoordinatesU)
    I.newDataArray('CoordinateY', I.getValue(CoordinateYS), gridCoordinatesU)
    I.newDataArray('CoordinateZ', I.getValue(CoordinateZS), gridCoordinatesU)
  
    #> with FlowSolutions
    for flowSolutionS in I.getNodesFromType1(zoneS, "FlowSolution_t"):
      flowSolutionU = I.newFlowSolution(I.getName(flowSolutionS), parent=zoneU)
      gridLocationS = I.getNodeFromType1(zoneS, "GridLocation_t")
      if gridLocationS:
        I.addChild(flowSolutionU, gridLocationS)
      else:
        I.newGridLocation("CellCenter", flowSolutionU)
      for dataS in I.getNodesFromType1(flowSolutionS, "DataArray_t"):
        I.addChild(flowSolutionU, dataS)
  
    #> with NgonElements
    #>> Definition en non structure des faces
    vtxRangeS  = MDIDF.uniform_distribution_at(nVtxTotS, iRank, nRank)
    slabListVtxS  = HFR2S.compute_slabs(nVtxS, vtxRangeS)
    nbFacesAllSlabsPerZone = sum([vtx_slab_to_n_face(slab, nVtxS) for slab in slabListVtxS])
    faceNumber    = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceNgon      = -np.ones(4*nbFacesAllSlabsPerZone, dtype=np.int32)
    faceLeftCell  = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    faceRightCell = -np.ones(  nbFacesAllSlabsPerZone, dtype=np.int32)
    compute_all_ngon_connectivity(slabListVtxS,nVtxS,nCellS,
                                  faceNumber,faceNgon,
                                  faceLeftCell,faceRightCell)
    #>> PartToBlock pour ordonner et equidistribuer les faces
    #>>> Creation de l'objet partToBlock
    #>>> PDM_part_to_block_distrib_t t_distrib = 0 ! Numerotation recalculee sur tous les procs
    #>>> PDM_part_to_block_post_t    t_post    = 0 ! Pas de traitement sur les valeurs
    #>>> PDM_stride_t                t_stride  = 1 ! Stride variable car variable en sortie...
    partToBlockObject = PDM.PartToBlock(comm, [faceNumber], None, 1, 0, 0, 1)
    #>>> Premier echange pour le ParentElements
    pFieldStride1 = dict()  
    pFieldStride1["faceLeftCell"] = [faceLeftCell]
    pFieldStride1["faceRightCell"] = [faceRightCell]
    pStride1 = [np.ones(nbFacesAllSlabsPerZone, dtype='int32')]
    dFieldStride1 = dict()  
    partToBlockObject.PartToBlock_Exchange(dFieldStride1, pFieldStride1, pStride1)
    #>>> Deuxieme echange pour l'ElementConnectivity
    pFieldStride4 = dict()
    pFieldStride4["faceNgon"] = [faceNgon]
    pStride4 = [4*np.ones(nbFacesAllSlabsPerZone,dtype='int32')]
    dFieldStride4 = dict()
    partToBlockObject.PartToBlock_Exchange(dFieldStride4, pFieldStride4, pStride4)

    #>>> Distribution des faces  
    facesDistribution = partToBlockObject.getDistributionCopy()
    # >> Creation du noeud NGonElements
    ngon = I.newElements('NGonElements', 'NGON', dFieldStride4["faceNgon"],
                         [1, nbFacesTot], parent=zoneU)
    nbFacesLoc = dFieldStride1["faceLeftCell"].shape[0]
    pe = np.array([dFieldStride1["faceLeftCell"],
                   dFieldStride1["faceRightCell"]]).transpose()

    I.newParentElements(pe,ngon)
    startOffset = facesDistribution[iRank]
    endOffset   = startOffset + nbFacesLoc+1
    I.newDataArray("ElementStartOffset", 4*np.arange(startOffset,endOffset), parent=ngon)
    I.newIndexArray('ElementConnectivity#Size', [nbFacesTot*4], parent=ngon)

    #> with ZoneBC
    zoneBCS = I.getNodeFromType1(zoneS, "ZoneBC_t")
    if zoneBCS is not None:
      zoneBCU = I.newZoneBC(zoneU)
      for bcS in I.getNodesFromType1(zoneBCS,"BC_t"):
        gridLocationNodeS = I.getNodeFromType1(bcS, "GridLocation_t")
        gridLocationS = I.getValue(gridLocationNodeS) if gridLocationNodeS is not None else "Vertex"
        pointRange = I.getValue(I.getNodeFromName1(bcS, 'PointRange'))

        #Compute slabs from attended location (better load balance)
        sizeS = transform_bnd_pr_size(pointRange, gridLocationS, attendedGridLocationBC)
        bc_range = MDIDF.uniform_distribution_at(sizeS.prod(), iRank, nRank)
        bc_slabs = HFR2S.compute_slabs(sizeS, bc_range)

        #Prepare sub pointRanges from slabs
        sub_pr_list = [np.asarray(slab) for slab in bc_slabs]
        for sub_pr in sub_pr_list:
          sub_pr[:,0] += pointRange[:,0]
          sub_pr[:,1] += pointRange[:,0] - 1

        if attendedGridLocationBC == 'CellCenter':
          fix_cell_point_ranges(pointRange, nCellS, sub_pr_list)

        if gridLocationS == "Vertex":
          pointList = compute_pointList_from_vertexRange(pointRange,sub_pr_list,nVtxS,attendedGridLocationBC)
        # elif "FaceCenter" in gridLocationS:
          # assert attendedGridLocationBC == "FaceCenter"
          # pointList = compute_faceList_from_faceRange(pointRange,bc_slabs,nCellS,nVtxS,gridLocationS)

        bcU = I.newBC(I.getName(bcS), btype=I.getValue(bcS), parent=zoneBCU)
        I.newGridLocation(attendedGridLocationBC, parent=bcU)
        I.newPointList(value=pointList, parent=bcU)
        I.newIndexArray('PointList#Size', [1, sizeS.prod()], parent=bcU)
        allowed_types = ['FamilyName_t'] #Copy these nodes to bcU
        for allowed_child in [c for c in I.getChildren(bcS) if I.getType(c) in allowed_types]:
          I.addChild(bcU, allowed_child)

    #> with ZoneGC
    zoneName = I.getName(zoneS)
    for zoneGCS in I.getNodesFromType1(zoneS, "ZoneGridConnectivity_t"):
      zoneGCU = I.newZoneGridConnectivity(I.getName(zoneGCS), parent=zoneU)
      for gcS in I.getNodesFromType1(zoneGCS, "GridConnectivity1to1_t"):
        gridLocationNodeS = I.getNodeFromType1(gcS, "GridLocation_t")
        assert gridLocationNodeS is None or I.getValue(gridLocationNodeS) == "Vertex"

        zoneDonorName = I.getValue(gcS)
        zoneDonor     = I.getNodeFromName1(baseS, zoneDonorName)
        nVtxSDonor    = I.getValue(zoneDonor)[:,0]

        transform = I.getValue(I.getNodeFromName1(gcS, 'Transform'))
        T = compute_transformMatrix(transform)

        pointRange      = I.getValue(I.getNodeFromName1(gcS, 'PointRange'))
        pointRangeDonor = I.getValue(I.getNodeFromName1(gcS, 'PointRangeDonor'))

        # One of the two connected zones is choosen to compute the slabs/sub_pointrange and to impose
        # it to the opposed zone.
        if zoneName <= zoneDonorName:
          pointRangeLoc, pointRangeDonorLoc = pointRange, pointRangeDonor
          nVtxLoc, nVtxDonorLoc = nVtxS, nVtxSDonor
        else:
          pointRangeLoc, pointRangeDonorLoc = pointRangeDonor, pointRange
          nVtxLoc, nVtxDonorLoc = nVtxSDonor, nVtxS
          T = T.transpose()
        # Refence PR must be increasing, otherwise we have troubles with slabs->sub_point_range
        # When we swap the PR, we must swap the corresponding dim of the PRD as well
        dir_to_swap     = (pointRangeLoc[:,1] < pointRangeLoc[:,0])
        opp_dir_to_swap = dir_to_swap[abs(transform) - 1]
        pointRangeLoc[dir_to_swap, 0], pointRangeLoc[dir_to_swap, 1] = \
                pointRangeLoc[dir_to_swap, 1], pointRangeLoc[dir_to_swap, 0]
        pointRangeDonorLoc[opp_dir_to_swap,0], pointRangeDonorLoc[opp_dir_to_swap,1] \
            = pointRangeDonorLoc[opp_dir_to_swap,1], pointRangeDonorLoc[opp_dir_to_swap,0]

        #Compute slabs from attended location (better load balance)
        sizeS = transform_bnd_pr_size(pointRangeLoc, "Vertex", attendedGridLocationGC)
        gc_range = MDIDF.uniform_distribution_at(sizeS.prod(), iRank, nRank)
        gc_slabs = HFR2S.compute_slabs(sizeS, gc_range)

        sub_pr_list = [np.asarray(slab) for slab in gc_slabs]
        #Compute sub pointranges from slab
        for sub_pr in sub_pr_list:
          sub_pr[:,0] += pointRangeLoc[:,0]
          sub_pr[:,1] += pointRangeLoc[:,0] - 1

        #Get opposed sub point ranges
        sub_pr_opp_list = []
        for sub_pr in sub_pr_list:
          sub_pr_opp = np.empty((3,2), dtype=np.int32)
          sub_pr_opp[:,0] = apply_transformation(sub_pr[:,0], pointRangeLoc[:,0], pointRangeDonorLoc[:,0], T)
          sub_pr_opp[:,1] = apply_transformation(sub_pr[:,1], pointRangeLoc[:,0], pointRangeDonorLoc[:,0], T)
          sub_pr_opp_list.append(sub_pr_opp)

        #If output location is vertex, sub_point_range are ready. Otherwise, some corrections are required
        if attendedGridLocationGC == 'CellCenter':
          fix_cell_point_ranges(pointRangeLoc, nVtxLoc-1, sub_pr_list)
          fix_cell_point_ranges(pointRangeDonorLoc, nVtxDonorLoc-1, sub_pr_opp_list)

        #When working on cell|face, extra care has to be taken if PR[:,1] < PR[:,0] : the cell|face id
        #is not given by the bottom left corner but by the top right. We can just shift to retrieve casual behaviour
        if 'Center' in attendedGridLocationGC:
          for sub_pr_opp in sub_pr_opp_list:
            reverted = sub_pr_opp[:,0] > sub_pr_opp[:,1]
            sub_pr_opp[reverted,:] -= 1

        pointListLoc      = compute_pointList_from_vertexRange(pointRangeLoc, sub_pr_list, nVtxLoc, attendedGridLocationGC)
        pointListDonorLoc = compute_pointList_from_vertexRange(pointRangeDonorLoc, sub_pr_opp_list, nVtxDonorLoc, attendedGridLocationGC)

        if zoneName <= zoneDonorName:
          pointList, pointListDonor = pointListLoc, pointListDonorLoc
        else:
          pointList, pointListDonor = pointListDonorLoc, pointListLoc

        gcU = I.newGridConnectivity(I.getName(gcS), I.getValue(gcS), 'Abutting1to1', zoneGCU)
        I.newGridLocation(attendedGridLocationGC, gcU)
        I.newPointList('PointList'     , pointList,      parent=gcU)
        I.newPointList('PointListDonor', pointListDonor, parent=gcU)
        I.newIndexArray('PointList#Size', [1, sizeS.prod()], gcU)
        #Copy these nodes to gcU
        allowed_types = ['GridConnectivityProperty_t']
        allowed_names = ['Ordinal', 'OrdinalOpp']
        for child in I.getChildren(gcS):
          if I.getName(child) in allowed_names or I.getType(child) in allowed_types:
            I.addChild(gcU, child)
      
  for flowEquationSetS in I.getNodesFromType1(baseS,"FlowEquationSet_t"):
    I.addChild(baseU,flowEquationSetS)
  
  for referenceStateS in I.getNodesFromType1(baseS,"ReferenceState_t"):
    I.addChild(baseU,referenceStateS)
  
  for familyS in I.getNodesFromType1(baseS,"Family_t"):
    I.addChild(baseU,familyS)

  return distTreeU
