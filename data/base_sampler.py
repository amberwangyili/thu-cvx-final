# -*- coding: utf-8 -*-
# @Author: amberwangyili
# @Date:   2021-05-24 13:56:54
# @Last Modified by:   amberwangyili
# @Last Modified time: 2021-06-02 16:36:27

import numpy as np
class BaseSampler():
  def __init__(self, opt):
    pass

  def sample_constant_weight(self, dim_n, dim_m):
    return np.ones(dim_n)/dim_n, np.ones(dim_m)/dim_m

  def _l2_loss(self,mu,nu):
    m, n = mu.shape[0], nu.shape[0]
    ind = np.indices((m,n))
    return (((mu[ind[0]])-(nu[ind[1]]))**2).sum(axis=2)


  def sample_weight(self,dim_n, dim_m):
    pass
  def sample_position(self,dim,scalar):
    pass
  def sample_cost(self,dim_n, dim_m,scalar=1):
    pass