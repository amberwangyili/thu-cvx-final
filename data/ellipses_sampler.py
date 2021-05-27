# -*- coding: utf-8 -*-
# @Author: Mengfei Xia
# @Date:   2021-05-24 13:53:59
# @Last Modified by:   Mengfei Xia
# @Last Modified time: 2021-05-24 14:47:29

import numpy as np, math
from .base_sampler import BaseSampler


class EllipsesSampler(BaseSampler):
  def __init__(self, opt):
    super(EllipsesSampler, self).__init__(opt)
  
  def sample_weight(self,dim_n, dim_m):
    return self.sample_constant_weight(dim_n, dim_m)

  def sample_position(self,dim,scalar):
    r_x, r_y = scalar
    r = np.random.uniform(0, 2. * math.pi, dim)
    dx = np.cos(r) + 0.1 / math.sqrt(2.) * np.random.randn(dim)
    dy = np.sin(r) + 0.1 / math.sqrt(2.) * np.random.randn(dim)
    x = r_x * dx
    y = r_y * dy
    p = np.concatenate((x.reshape(dim, 1), y.reshape(dim, 1)), axis=1)
    return p

  def sample_cost(self,dim_n, dim_m, scalar=1):
    """default parameters: (x_center, y_center, r_x, r_y, eps) = (0, 0, 0.5, 2, 0.1)"""
    mu = self.sample_position(dim_n, (0.5, 2))
    nu = self.sample_position(dim_m, (2, 0.5))
    return self._l2_loss(mu,nu)
