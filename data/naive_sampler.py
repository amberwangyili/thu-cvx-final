# -*- coding: utf-8 -*-
# @Author: amberwangyili
# @Date:   2021-05-24 13:53:59
# @Last Modified by:   amberwangyili
# @Last Modified time: 2021-06-02 16:36:35
import numpy as np
from .base_sampler import BaseSampler

class NaiveSampler(BaseSampler):
  def __init__(self):
    super(NaiveSampler, self).__init__()
    print('Naive sampler has been constructed.')

  def sample_weight(self,dim):
    vec = np.random.random(dim)
    return vec/sum(vec)

  def sample_position(self,dim,scalar):
    return scalar * np.random.rand(dim * 2 ).reshape((dim,2))

  def sample_cost(self,dim_n, dim_m, scalar=1):
    mu = self.sample_position(dim_n,scalar)
    nu = self.sample_position(dim_m,scalar)
    return self._l2_loss(mu,nu)