# -*- coding: utf-8 -*-
# @Author: amberwangyili
# @Date:   2021-05-24 13:53:59
# @Last Modified by:   amberwangyili
# @Last Modified time: 2021-06-02 16:36:41
import numpy as np
from .base_sampler import BaseSampler


class LoadSampler(BaseSampler):
  def __init__(self, opt):
    super(LoadSampler, self).__init__(opt)
    if opt['dim_n'] == 320:
      self.data = np.load('./datasets/Saved/data320.npy', allow_pickle=True).item()
    elif opt['dim_n'] == 1000:
      self.data = np.load('./datasets/Saved/data1000.npy', allow_pickle=True).item()
    else:
      assert 0
    print('Load sampler has been constructed.')

  def sample_weight(self,dim_n, dim_m):
    mu = self.data['mu']
    nu = self.data['nu']
    return mu, nu

  def sample_position(self,dim,scalar):
    return scalar * np.random.rand(dim * 2 ).reshape((dim,2))

  def sample_cost(self,dim_n, dim_m, scalar=1):
    return self.data['C']