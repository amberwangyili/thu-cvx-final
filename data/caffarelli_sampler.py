# -*- coding: utf-8 -*-
# @Author: Mengfei Xia
# @Date:   2021-05-24 13:53:59
# @Last Modified by:   Mengfei Xia
# @Last Modified time: 2021-05-24 14:47:29

import numpy as np
from .base_sampler import BaseSampler


class CaffarelliSampler(BaseSampler):
  def __init__(self, opt):
    super(CaffarelliSampler, self).__init__(opt)
  
  def sample_weight(self,dim_n, dim_m):
    return self.sample_constant_weight(dim_n, dim_m)

  def sample_position(self,dim,scalar):
    l = 0
    while l <= dim:
      ox = np.random.uniform(-1, 1, dim)
      oy = np.random.uniform(-1, 1, dim)
      mask = ox ** 2 + oy ** 2 < 1 ** 2
      dx, dy = ox[mask], oy[mask]
      dx[dx < 0.] -= scalar
      dx[dx >= 0.] += scalar
      n = dx.size
      x = dx
      y = dy
      p = np.concatenate((x.reshape(n, 1), y.reshape(n, 1)), axis=1)
      if l == 0:
        pos = p
      else:
        pos = np.concatenate((pos, p), axis=0)
      l = len(pos)
    return pos[0:dim, ]

  def sample_cost(self,dim_n, dim_m, scalar=1):
    """default parameters: (x_center, y_center, r, d) = (0, 0, 1, 2)"""
    mu = self.sample_position(dim_n, 0)
    nu = self.sample_position(dim_m, 2)
    return self._l2_loss(mu,nu)