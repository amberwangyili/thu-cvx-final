# -*- coding: utf-8 -*-
# @Author: Amber
# @Date:   2021-05-24 13:53:59
# @Last Modified by:   yiliwang
# @Last Modified time: 2021-05-24 14:47:29
import numpy as np
from .base_sampler import BaseSampler

class NaiveSampler(BaseSampler):
  def __init__(self):
    super(NaiveSampler, self).__init__()

  def sample_weight(self,dim):
    vec = np.random.random(dim)
    return vec/sum(vec)

  def sample_position(self,dim,scalar):
    return scalar * np.random.rand(dim * 2 ).reshape((dim,2))

  def sample_cost(self,dim_n, dim_m, scalar=1):
    mu = self.sample_position(dim_n,scalar)
    nu = self.sample_position(dim_m,scalar)
    return self._l2_loss(mu,nu)