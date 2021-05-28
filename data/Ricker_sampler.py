# -*- coding: utf-8 -*-
# @Author: Mengfei Xia
# @Date:   2021-05-24 13:53:59
# @Last Modified by:   Mengfei Xia
# @Last Modified time: 2021-05-24 14:47:29

import numpy as np, math
from .base_sampler import BaseSampler
from scipy.io import loadmat


class RickerSampler(BaseSampler):
  def __init__(self, opt):
    super(RickerSampler, self).__init__(opt)
    self.data = loadmat('./datasets/Ricker/shift_of_Ricker.mat')
  
  def sample_weight(self,dim_n, dim_m):
    mu = self.data['fx'].reshape(-1,)
    nu = self.data['fy'].reshape(-1,)
    return mu, nu

  def sample_position(self,dim,scalar):
    assert 0

  def sample_cost(self,dim_n, dim_m, scalar=1):
    C = self.data['C']
    return C
