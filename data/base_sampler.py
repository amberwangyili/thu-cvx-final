# -*- coding: utf-8 -*-
# @Author: Amber
# @Date:   2021-05-24 13:56:54
# @Last Modified by:   yiliwang
# @Last Modified time: 2021-05-24 14:35:07

import numpy as np
class BaseSampler():
	def __init__(self):
		pass

	def sample_constant_weight(self,dim):
		return np.ones(dim)/dim

	def _l2_loss(self,mu,nu):
		m, n = mu.shape[0], nu.shape[0]
		ind = np.indices((m,n))
		return (((mu[ind[0]])-(mu[ind[1]]))**2).sum(axis=2)


	def sample_weight(self,dim):
		pass
	def sample_position(self,dim,scalar):
		pass
	def sample_cost(self,dim,scalar=1):
		pass