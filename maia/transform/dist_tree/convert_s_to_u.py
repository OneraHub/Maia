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
def compute_pointList_from_vertexRange(pointRange,iRank,nRank,nCellS,nVtxS, output_loc):
  """
  Transform structured PointRange with 'GridLocation'='Vertex' to unstructured
  PointList with 'GridLocation'='FaceCenter', 'Vertex' or 'CellCenter'
  """
  if output_loc == 'FaceCenter':
    sizeS = np.maximum(np.abs(pointRange[:,1] - pointRange[:,0]), 1)
  elif output_loc == 'Vertex':
    sizeS = np.abs(pointRange[:,1] - pointRange[:,0]) + 1
  elif output_loc == 'CellCenter':
    sizeS = np.maximum(np.abs(pointRange[:,1] - pointRange[:,0]), 1)
  else:
    raise ValueError("Wrong output location : '{}'".format(output_loc))

  rangeS = MDIDF.uniform_distribution_at(sizeS.prod(), iRank, nRank)
  slabListS = HFR2S.compute_slabs(sizeS, rangeS)
  sizeU = rangeS[1]-rangeS[0]
  pointList = np.empty((1,sizeU), dtype=np.int32)
  counter = 0

  #Find constant direction
  cst_axes = np.nonzero(pointRange[:,0] == pointRange[:,1])[0]
  if len(cst_axes) != 1:
    raise ValueError("The PointRange '{}' is bad defined".format(pointRange))
  cst_axe = cst_axes[0]
  cst_val = pointRange[cst_axe,0]
  if output_loc == 'CellCenter' and pointRange[cst_axe,0] > nCellS[cst_axe]:
    cst_val -= 1

  # The lambda func ijk_to_func redirect to the good indexing function depending
  # on the output grid location
  convert_ijk_to_faceIndex = [convert_ijk_to_faceiIndex, convert_ijk_to_facejIndex, convert_ijk_to_facekIndex]
  if output_loc == 'FaceCenter':
    ijk_to_func = lambda i,j,k : convert_ijk_to_faceIndex[cst_axe](i, j, k, nCellS, nVtxS)
  elif output_loc == 'Vertex':
    ijk_to_func = lambda i,j,k : convert_ijk_to_index(i, j, k, *nVtxS)
  elif output_loc == 'CellCenter':
    ijk_to_func = lambda i,j,k : convert_ijk_to_index(i, j, k, *nCellS)

  # The lambda func ijk_to_vect_func is a wrapping to ijk_to_func (and so to the good indexing func)
  # but with args expressed as numpy arrays (for 2 of them) : this allow vectorial call of indexing
  # function as if we did a double for loop
  if cst_axe == 0:
    ijk_to_vect_func = lambda i_idx, j_idx, k_idx : ijk_to_func(cst_val, j_idx, k_idx.reshape(-1,1))
  elif cst_axe == 1:
    ijk_to_vect_func = lambda i_idx, j_idx, k_idx : ijk_to_func(i_idx, cst_val, k_idx.reshape(-1,1))
  elif cst_axe == 2:
    ijk_to_vect_func = lambda i_idx, j_idx, k_idx : ijk_to_func(i_idx, j_idx.reshape(-1,1), cst_val)

  for slabS in slabListS:
    iS,iE, jS,jE, kS,kE = [item+1 for bounds in slabS for item in bounds]
    n_faces = (iE-iS)*(jE-jS)*(kE-kS)
    pointList[0][counter:counter+n_faces] = ijk_to_vect_func(
        np.arange(iS, iE), np.arange(jS, jE), np.arange(kS, kE)).flatten()
    counter += n_faces

  return pointList
###############################################################################

###############################################################################
def compute_faceList_from_faceRange(pointRange,iRank,nRank,nCellS,nVtxS,gridLocationS):
  """
  Transform structured PointRange with 'GridLocation'='IFaceCenter' or 'JFaceCenter' or
  'KFaceCenter' to unstructured PointList with 'GridLocation'='FaceCenter'
  """
  sizeS = np.abs(pointRange[:,1] - pointRange[:,0]) + 1
  rangeS = MDIDF.uniform_distribution_at(sizeS.prod(), iRank, nRank)
  slabListS = HFR2S.compute_slabs(sizeS, rangeS)
  sizeU = rangeS[1]-rangeS[0]
  faceList = np.empty((1,sizeU), dtype=np.int32)
  counter = 0

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

  for slabS in slabListS:
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
def convert_i1j1k1_to_i2j2k2(i1,j1,k1,iS1,jS1,kS1,iS2,jS2,kS2,T):
  """
  This function compute indices from current to oppposit or from opposit to current
  by using the transform matrix.
  As defined in the SIDS of CGNS (https://cgns.github.io/CGNS_docs_current/sids/cnct.html) :
  Index2 = T.(Index1 - Begin1) + Begin2
  """
  vector = np.array([i1-iS1,j1-jS1,k1-kS1])
  [i2,j2,k2] = np.matmul(T,vector,order='F') + np.array([iS2,jS2,kS2])
  
  return(i2,j2,k2)
###############################################################################

###############################################################################
def compute_faceList2_from_vertexRanges(pointRange1,pointRange2,T,nCell2,nVtx2):
  [iS1,jS1,kS1] = pointRange1[:,0]
  [iS2,jS2,kS2] = pointRange2[:,0]
  size1 = np.maximum(np.abs(pointRange1[:,1] - pointRange1[:,0]), 1)
  bounds1 = MDIDF.uniform_distribution_at(size1.prod(), iRank, nRank)
  slabList = HFR2S.compute_slabs(size1, bounds1)
  size2 = bounds1[1]-bounds1[0]
  faceList2 = np.empty((1,size2),dtype=np.int32)
  if T[0].sum() < 0:
    correcti2 = -1
  else:
    correcti2 = 0
  if T[1].sum() < 0:
    correctj2 = -1
  else:
    correctj2 = 0
  if T[2].sum() < 0:
    correctk2 = -1
  else:
    correctk2 = 0
  counter = 0
  for slab in slabList:
    iS,iE, jS,jE, kS,kE = [item+1 for bounds in slab for item in bounds]
    if pointRange2[0,0] == pointRange2[0,1]:
    #>> face i2
      for k1 in range(kS,kE):
        for j1 in range(jS,jE):
          for i1 in range(iS,iE):
            (i2,j2,k2) = convert_i1j1k1_to_i2j2k2(i1,j1,k1,iS1,jS1,kS1,iS2,jS2,kS2,T)
            j2 += correctj2
            k2 += correctk2
            faceList2[0][counter] = convert_ijk_to_faceiIndex(i2,j2,k2,nCell2,nVtx2)
            counter += 1
    elif pointRange2[1,0] == pointRange2[1,1]:
    #>> face j2
      for k1 in range(kS,kE):
        for j1 in range(jS,jE):
          for i1 in range(iS,iE):
            (i2,j2,k2) = convert_i1j1k1_to_i2j2k2(i1,j1,k1,iS1,jS1,kS1,iS2,jS2,kS2,T)
            i2 += correcti2
            k2 += correctk2
            faceList2[0][counter] = convert_ijk_to_facejIndex(i2,j2,k2,nCell2,nVtx2)
            counter += 1
    elif pointRange2[2,0] == pointRange2[2,1]:
    #>> face k2
      for k1 in range(kS,kE):
        for j1 in range(jS,jE):
          for i1 in range(iS,iE):
            (i2,j2,k2) = convert_i1j1k1_to_i2j2k2(i1,j1,k1,iS1,jS1,kS1,iS2,jS2,kS2,T)
            i2 += correcti2
            j2 += correctj2
            faceList2[0][counter] = convert_ijk_to_facekIndex(i2,j2,k2,nCell2,nVtx2)
            counter += 1
    else:
      raise ValueError("The PointRange2 '{}' is bad defined".format(pointRange2))    
    
  return(faceList2)
###############################################################################

###############################################################################
def compute_vertexList2_from_vertexRanges(pointRange1,pointRange2,T,nCell2,nVtx2):
  [iS1,jS1,kS1] = pointRange1[:,0]
  [iS2,jS2,kS2] = pointRange2[:,0]
  size1 = np.abs(pointRange1[:,1] - pointRange1[:,0]) + 1
  bounds1 = MDIDF.uniform_distribution_at(size1.prod(), iRank, nRank)
  slabList = HFR2S.compute_slabs(size1, bounds1)
  size2 = bounds1[1]-bounds1[0]
  vertexList2 = np.empty((1,size2),dtype=np.int32)
  counter = 0
  for slab in slabList:
    iS,iE, jS,jE, kS,kE = [item+1 for bounds in slab for item in bounds]
    for k1 in range(kS,kE):
      for j1 in range(jS,jE):
        for i1 in range(iS,iE):
          (i2,j2,k2) = convert_i1j1k1_to_i2j2k2(i1,j1,k1,iS1,jS1,kS1,iS2,jS2,kS2,T)
          vertexList2[0][counter] = convert_ijk_to_index(i2,j2,k2,nVtx2[0],nVtx2[1],nVtx2[2])
          counter += 1
    
  return(vertexList2)
###############################################################################

###############################################################################
def compute_cellList2_from_vertexRanges(pointRange1,pointRange2,T,nCell2,nVtx2):
  [iS1,jS1,kS1] = pointRange1[:,0]
  [iS2,jS2,kS2] = pointRange2[:,0]
  size1 = np.maximum(np.abs(pointRange1[:,1] - pointRange1[:,0]), 1)
  bounds1 = MDIDF.uniform_distribution_at(size1.prod(), iRank, nRank)
  slabList = HFR2S.compute_slabs(size1, bounds1)
  size2 = bounds1[1]-bounds1[0]
  cellList2 = np.empty((1,size2),dtype=np.int32)
  if T[0].sum() < 0:
    correcti2 = -1
  else:
    correcti2 = 0
  if T[1].sum() < 0:
    correctj2 = -1
  else:
    correctj2 = 0
  if T[2].sum() < 0:
    correctk2 = -1
  else:
    correctk2 = 0
  counter = 0
  for slab in slabList:
    iS,iE, jS,jE, kS,kE = [item+1 for bounds in slab for item in bounds]
    if pointRange2[0,0] == pointRange2[0,1]:
    #>> face i2
      for k1 in range(kS,kE):
        for j1 in range(jS,jE):
          for i1 in range(iS,iE):
            (i2,j2,k2) = convert_i1j1k1_to_i2j2k2(i1,j1,k1,iS1,jS1,kS1,iS2,jS2,kS2,T)
            j2 += correctj2
            k2 += correctk2
            cellList2[0][counter] = convert_ijk_to_index(i2,j2,k2,nCell2[0],nCell2[1],nCell2[2])
            counter += 1
    elif pointRange2[1,0] == pointRange2[1,1]:
    #>> face j2
      for k1 in range(kS,kE):
        for j1 in range(jS,jE):
          for i1 in range(iS,iE):
            (i2,j2,k2) = convert_i1j1k1_to_i2j2k2(i1,j1,k1,iS1,jS1,kS1,iS2,jS2,kS2,T)
            i2 += correcti2
            k2 += correctk2
            cellList2[0][counter] = convert_ijk_to_index(i2,j2,k2,nCell2[0],nCell2[1],nCell2[2])
            counter += 1
    elif pointRange2[2,0] == pointRange2[2,1]:
    #>> face k2
      for k1 in range(kS,kE):
        for j1 in range(jS,jE):
          for i1 in range(iS,iE):
            (i2,j2,k2) = convert_i1j1k1_to_i2j2k2(i1,j1,k1,iS1,jS1,kS1,iS2,jS2,kS2,T)
            i2 += correcti2
            j2 += correctj2
            cellList2[0][counter] = convert_ijk_to_index(i2,j2,k2,nCell2[0],nCell2[1],nCell2[2])
            counter += 1
    else:
      raise ValueError("The PointRange2 '{}' is bad defined".format(pointRange2))    
    
  return(cellList2)
###############################################################################

###############################################################################
def compute_dZones2ID(distTree):
  return{I.getName(zone):z for z, zone in enumerate(I.getZones(distTree))}

###############################################################################

###############################################################################
def compute_dJoins2ID(distree,dZones2ID):
  dJoins2ID    = {}
  counterJoins = 0
  for zone in I.getZones(distTree):
    zoneName = I.getName(zone)
    zoneID   = dZones2ID[zoneName]
    zoneGC = I.getNodesFromType1(zone, 'ZoneGridConnectivity_t')
    joins  = I.getNodesFromType1(zoneGC, 'GridConnectivity_t')
    joins += I.getNodesFromType1(zoneGC, 'GridConnectivity1to1_t')
    for join in joins:
      zoneDonorName = I.getValue(join)
      zoneDonorID   = dZones2ID[zoneDonorName]
      if zoneID < zoneDonorID:
        pointRange = I.getValue(I.getNodeFromName1(join, 'PointRange'))
        strPointRange = ""
        for dim1 in range(pointRange.shape[0]):
          for dim2 in range(pointRange.shape[1]):
            strPointRange += "_{0}".format(pointRange[dim1][dim2])
            joinName = str(zoneID)+"_"+str(zoneDonorID)+strPointRange
            dJoins2ID[joinName] = counterJoins
            counterJoins += 1
  return dJoins2ID

###############################################################################

###############################################################################
def correctPointRanges(pointRange,pointRangeDonor,transform,zoneID,zoneDonorID):
  newPointRange      = np.empty(pointRange.shape,dtype=np.int32)
  newPointRangeDonor = np.empty(pointRange.shape,dtype=np.int32)
  for d in range(pointRange.shape[0]):
    dDonor = abs(transform[d])-1
    nbPointd = pointRange[d][1] - pointRange[d][0]
    nbPointDonord = np.sign(transform[d])*(pointRangeDonor[dDonor][1] - pointRangeDonor[dDonor][0])
    if nbPointd == nbPointDonord:
      newPointRange[d][0]      = pointRange[d][0]
      newPointRange[d][1]      = pointRange[d][1]
      newPointRangeDonor[d][0] = pointRangeDonor[dDonor][0]
      newPointRangeDonor[d][1] = pointRangeDonor[dDonor][1]
    else:
      if zoneID < zoneDonorID:
        newPointRange[d][0]      = pointRange[d][0]
        newPointRange[d][1]      = pointRange[d][1]
        newPointRangeDonor[d][0] = pointRangeDonor[dDonor][1]
        newPointRangeDonor[d][1] = pointRangeDonor[dDonor][0]
      else:
        newPointRange[d][0]      = pointRange[d][1]
        newPointRange[d][1]      = pointRange[d][0]
        newPointRangeDonor[d][0] = pointRangeDonor[dDonor][0]
        newPointRangeDonor[d][1] = pointRangeDonor[dDonor][1]
        
  return (newPointRange,newPointRangeDonor)
###############################################################################

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
# Début convertion DistTreeS en DistTreeU

def convert_s_to_u(distTreeS,comm,attendedGridLocationBC="FaceCenter",attendedGridLocationGC="FaceCenter"):

  nRank = comm.Get_size()
  iRank = comm.Get_rank()

  dZones2ID = compute_dZones2ID(distTreeS)
  
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
    zoneBCS = I.getNodeFromType1(zoneS,"ZoneBC_t")
    if zoneBCS is not None:
      zoneBCU = I.newZoneBC(zoneU)
      for bcS in I.getNodesFromType1(zoneBCS,"BC_t"):
        gridLocationNodeS = I.getNodeFromType1(bcS, "GridLocation_t")
        gridLocationS = I.getValue(gridLocationNodeS) if gridLocationNodeS is not None else "Vertex"
        bcU = copy.deepcopy(bcS)
        I._rmNodesByType(bcU,"GridLocation_t")
        I._rmNodesByType(bcU,"IndexRange_t")
        I._rmNodesByName(bcU,":CGNS#Distribution")
        pointRange = I.getValue(I.getNodeFromName1(bcS, 'PointRange'))
        if gridLocationS == "Vertex":
          if attendedGridLocationBC == "FaceCenter":
            sizeS     = np.maximum(np.abs(pointRange[:,1] - pointRange[:,0]), 1)
          elif attendedGridLocationBC == "Vertex":
            sizeS     = np.abs(pointRange[:,1] - pointRange[:,0]) + 1
          elif attendedGridLocationBC == "CellCenter":
            sizeS     = np.maximum(np.abs(pointRange[:,1] - pointRange[:,0]), 1)
          else:
            raise ValueError("attendedGridLocationBC is '{}' but allowed values are \
                'Vertex', 'FaceCenter' or 'CellCenter'".format(attendedGridLocationBC))
          pointList = compute_pointList_from_vertexRange(\
              pointRange,iRank,nRank,nCellS,nVtxS,attendedGridLocationBC)
        elif "FaceCenter" in gridLocationS:
          if attendedGridLocationBC == "FaceCenter":
            pointList = compute_faceList_from_faceRange(pointRange,iRank,nRank,nCellS,nVtxS,gridLocationS)
            sizeS     = np.abs(pointRange[:,1] - pointRange[:,0]) + 1
          # elif attendedGridLocationBC == "Vertex":
          #   pointList = compute_vertexList_from_vertexRange(pointRange,iRank,nRank,nVtxS)
          #   sizeS     = np.abs(pointRange[:,1] - pointRange[:,0]) + 1
          # elif attendedGridLocationBC == "CellCenter":
          #   pointList = compute_cellList_from_vertexRange(pointRange,iRank,nRank,nCellS)
          #   sizeS     = np.maximum(np.abs(pointRange[:,1] - pointRange[:,0]), 1)
          else:
            print("Not yet implemented !!!")
          #   raise ValueError("attendedGridLocationBC is '{}' but allowed values are 'Vertex', 'FaceCenter' or 'CellCenter'".format(attendedGridLocationBC))
        sizeU = sizeS.prod()
        I.newPointList(value=pointList,parent=bcU)
        I.newGridLocation(attendedGridLocationBC, parent=bcU)
        I.newIndexArray('PointList#Size', [1, sizeU], bcU)
        I.addChild(zoneBCU,bcU)

  #> with ZoneGC
    zoneGCS = I.getNodeFromType1(zoneS,"ZoneGridConnectivity_t")
    if zoneGCS is not None:
      zoneGCU = I.newZoneGridConnectivity(parent=zoneU)
      joinsS  = I.getNodesFromType1(zoneGCS, "GridConnectivity1to1_t")
      zoneID  = dZones2ID[zoneSName]
      for gcS in joinsS:
        gridLocationNodeS = I.getNodeFromType1(gcS, "GridLocation_t")
        if gridLocationNodeS is not None:
          if I.getValue(gridLocationNodeS) != "Vertex":
            raise ValueError("'GridLocation' value for a 'GridConnectivity1to1_t' node could only be 'Vertex'.")
        gcU = copy.deepcopy(gcS)
        I._rmNodesByType(gcU,"GridLocation_t")
        I._rmNodesByType(gcU,"IndexRange_t")
        I._rmNodesByName(gcU,":CGNS#Distribution")
        I._rmNodesByName(gcU,"Transform")
        I._setType(gcU, 'GridConnectivity_t')
        I.newGridConnectivityType('Abutting1to1', gcU) 
        pointRange      = I.getValue(I.getNodeFromName1(gcS, 'PointRange'))
        pointRangeDonor = I.getValue(I.getNodeFromName1(gcS, 'PointRangeDonor'))
        transform = I.getValue(I.getNodeFromName1(gcS, 'Transform'))
        zoneDonorName = I.getValue(gcS)
        zoneDonorID   = dZones2ID[zoneDonorName]
        (pointRange,pointRangeDonor) = correctPointRanges(pointRange,pointRangeDonor,transform,zoneID,zoneDonorID)
        I.setValue(I.getNodeFromName1(gcS, 'PointRange')     ,pointRange)
        I.setValue(I.getNodeFromName1(gcS, 'PointRangeDonor'),pointRangeDonor)
        zoneSDimsDonor = I.getValue(I.getNodeFromName1(baseS,zoneDonorName))
        nCellSDonor = zoneSDimsDonor[:,1]
        nVtxSDonor  = zoneSDimsDonor[:,0]
        T = compute_transformMatrix(transform)
        if zoneID == zoneDonorID:
        # raccord périodique sur la même zone
          if attendedGridLocationGC == "FaceCenter":
            sizeS = np.maximum(np.abs(pointRange[:,1] - pointRange[:,0]), 1)
          elif attendedGridLocationGC == "Vertex":
            sizeS = np.abs(pointRange[:,1] - pointRange[:,0]) + 1
          elif attendedGridLocationGC == "CellCenter":
            sizeS = np.maximum(np.abs(pointRange[:,1] - pointRange[:,0]), 1)
          else:
            raise ValueError("attendedGridLocationGC is '{}' but allowed values are \
                'Vertex', 'FaceCenter' or 'CellCenter'".format(attendedGridLocationGC))
          pointList = compute_pointList_from_vertexRange(\
              pointRange,iRank,nRank,nCellS,nVtxS,attendedGridLocationGC)
          pointListDonor = compute_pointList_from_vertexRange(\
              pointRangeDonor,iRank,nRank,nCellS,nVtxS,attendedGridLocationGC)
        elif zoneID < zoneDonorID:
          if attendedGridLocationGC == "FaceCenter":
            pointListDonor = compute_faceList2_from_vertexRanges(pointRange,pointRangeDonor,T,nCellSDonor,nVtxSDonor)
            sizeS          = np.maximum(np.abs(pointRange[:,1] - pointRange[:,0]), 1)
          elif attendedGridLocationGC == "Vertex":
            pointListDonor = compute_vertexList2_from_vertexRanges(pointRange,pointRangeDonor,T,nCellSDonor,nVtxSDonor)
            sizeS          = np.abs(pointRange[:,1] - pointRange[:,0]) + 1
          elif attendedGridLocationGC == "CellCenter":
            pointListDonor = compute_cellList2_from_vertexRanges(pointRange,pointRangeDonor,T,nCellSDonor,nVtxSDonor)
            sizeS = np.maximum(np.abs(pointRange[:,1] - pointRange[:,0]), 1)
          else:
            raise ValueError("attendedGridLocationGC is '{}' but allowed values are \
                'Vertex', 'FaceCenter' or 'CellCenter'".format(attendedGridLocationGC))
          pointList = compute_pointList_from_vertexRange(\
              pointRange,iRank,nRank,nCellS,nVtxS,attendedGridLocationGC)
        else:
          if attendedGridLocationGC == "FaceCenter":
            pointList      =compute_faceList2_from_vertexRanges(pointRangeDonor,pointRange,np.transpose(T),nCellS,nVtxS)
            sizeS          = np.maximum(np.abs(pointRange[:,1] - pointRange[:,0]), 1)
          elif attendedGridLocationGC == "Vertex":
            pointList      = compute_vertexList2_from_vertexRanges(pointRangeDonor,pointRange,np.transpose(T),nCellS,nVtxS)
            sizeS          = np.abs(pointRange[:,1] - pointRange[:,0]) + 1
          elif attendedGridLocationGC == "CellCenter":
            pointList      = compute_cellList2_from_vertexRanges(pointRangeDonor,pointRange,np.transpose(T),nCellS,nVtxS)
            sizeS = np.maximum(np.abs(pointRange[:,1] - pointRange[:,0]), 1)
          else:
            raise ValueError("attendedGridLocationGC is '{}' but allowed values are \
                'Vertex', 'FaceCenter' or 'CellCenter'".format(attendedGridLocationGC))
          pointListDonor = compute_pointList_from_vertexRange(\
              pointRangeDonor,iRank,nRank,nCellSDonor,nVtxSDonor,attendedGridLocationGC)
        sizeU = sizeS.prod()
        I.newPointList(value=pointList,parent=gcU)
        I.newPointList('PointListDonor',value=pointListDonor,parent=gcU)
        I.newGridLocation(attendedGridLocationGC,gcU)
        I.newIndexArray('PointList#Size', [1, sizeU], gcU)
        I.addChild(zoneGCU,gcU)
      
  # #> with ZoneGC
  # TO DO
  
  #> with FlowEquationSet
  for flowEquationSetS in I.getNodesFromType1(baseS,"FlowEquationSet_t"):
    I.addChild(baseU,flowEquationSetS)
  
  #> with ReferenceState
  for referenceStateS in I.getNodesFromType1(baseS,"ReferenceState_t"):
    I.addChild(baseU,referenceStateS)
  
  #> with Family
  for familyS in I.getNodesFromType1(baseS,"Family_t"):
    I.addChild(baseU,familyS)
  
