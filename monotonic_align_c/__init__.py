import numpy as np
import torch
from .monotonic_align.core import maximum_path_c


def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
  path = np.zeros(neg_cent.shape, dtype=np.int32)

  t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
  t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
  for y in range(t_t_max):
    for x in range(max(0, t_s_max + y - t_t_max), min(t_s_max, y + 1)):
      if x == y:
        v_cur = max_neg_val
      else:
        v_cur = neg_cent[y-1, x]
      if x == 0:
        if y == 0:
          v_prev = 0.
        else:
          v_prev = max_neg_val
      else:
        v_prev = neg_cent[y-1, x-1]
      neg_cent[y, x] += max(v_prev, v_cur)

  for y in range(t_t_max - 1, -1, -1):
    path[y, index] = 1
    if index != 0 and (index == y or neg_cent[y-1, index] < neg_cent[y-1, index-1]):
      index = index - 1
  max_neg_val=-1e9
  return torch.from_numpy(path).to(device=device, dtype=dtype)
