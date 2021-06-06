# -*- coding: utf-8 -*-
# @Author: Mengfei Xia
# @Date:   2021-05-24 13:53:59
# @Last Modified by:   Mengfei Xia
# @Last Modified time: 2021-05-24 14:47:29

import numpy as np
from .base_sampler import BaseSampler
import csv, os, math

data_classes = ['CauchyDensity', 'ClassicImages', 'GRFmoderate', 'GRFrough', 'GRFSmooth', 'LogGRF', 'LogitGRF', 'MicroscopyImages', 'Shapes', 'WhiteNoise']


class DOTSampler(BaseSampler):
  def __init__(self, opt):
    super(DOTSampler, self).__init__(opt)
    assert(opt['class'] in data_classes)
    m, n = opt['dim_m'], opt['dim_n']
    assert((m == n) and m in [32, 64])
    self.data_class = opt['class']
    print('DOTmark sampler has been constructed.')

  def sample_weight(self,dim_n, dim_m):
    path = os.getcwd()
    index = [1, 2]
    w = []
    for i in index:
      with open(path + '/datasets/DOT/' + self.data_class + '/data' + str(dim_n) + '_100' + str(i) + '.csv') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
          w.append(row)
      csvfile.close()
    w = np.array(w, np.float32).reshape(2, dim_n*dim_n)
    mu = w[0, :] / w[0, :].sum()
    nu = w[1, :] / w[1, :].sum()
    return mu, nu

  def sample_position(self, dim, scalar):
    """default parameters: (start_x1, end_x1, start_x2, end_x2) = (0, 1, 0, 1)"""
    step_x1, step_x2 = 1 / dim, 1 / dim
    x1 = np.linspace(step_x1 / 2.0, 1 - step_x1 / 2.0, dim)
    x2 = np.linspace(step_x2 / 2.0, 1 - step_x2 / 2.0, dim)
    x1p, x2p = np.meshgrid(x1, x2) 
    x = np.concatenate((x1p.reshape((int(dim* dim), 1)), x2p.reshape((int(dim * dim), 1))), axis=1)
    return x

  def sample_cost(self, dim_n, dim_m, scalar=1):
    mu = self.sample_position(dim_m, scalar)
    nu = self.sample_position(dim_n, scalar)
    return self._l2_loss(mu,nu)
