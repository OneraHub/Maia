import Converter.Internal as I
import numpy as np

def list_or_only_elt(l):
  return l[0] if len(l) == 1 else l

def interweave_arrays(array_list):
  #https://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays
  first  = array_list[0]
  number = len(array_list)
  output = np.empty(number*first.size, first.dtype)
  for i,array in enumerate(array_list):
    output[i::number] = array
  return output

def single_dim_pr_to_pl(pr, distrib=None):
  assert pr.shape[0] == 1
  if distrib is not None:
    return np.arange(pr[0,0]+distrib[0], pr[0,0]+distrib[1], dtype=pr.dtype).reshape((1,-1), order='F')
  else:
    return np.arange(pr[0,0], pr[0,1]+1, dtype=pr.dtype).reshape((1,-1), order='F')

def concatenate_point_list(point_lists, dtype=None):
  """
  Merge all the PointList arrays in point_lists list
  into a flat 1d array and an index array
  """
  sizes = [pl_n.size for pl_n in point_lists]

  merged_pl_idx = sizes_to_indices(sizes, dtype=np.int32)

  if dtype is None:
    dtype = point_lists[0].dtype if point_lists != [] else np.int

  merged_pl = np.empty(sum(sizes), dtype=dtype)
  for ipl, pl in enumerate(point_lists):
    merged_pl[merged_pl_idx[ipl]:merged_pl_idx[ipl+1]] = pl[0,:]

  return merged_pl_idx, merged_pl

def sizes_to_indices(nb_array, dtype=None):
  """ Create and offset array from a size array """
  nptype = dtype if dtype else np.asarray(nb_array).dtype
  offset_array = np.empty(len(nb_array)+1, dtype=nptype)
  offset_array[0] = 0
  np.cumsum(nb_array, out=offset_array[1:])
  return offset_array

def multi_arange(starts, stops):
  """
  Create concatenated ranges of integers for multiple start/stop
  See https://codereview.stackexchange.com/questions/83018/
  vectorized-numpy-version-of-arange-with-multiple-start-stop
  """
  stops = np.asarray(stops)
  l = stops - starts # Lengths of each range.
  return np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())

