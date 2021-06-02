# -*- coding: utf-8 -*-
# @Author: amberwangyili
# @Date:   2021-06-02 17:40:25
# @Last Modified by:   amberwangyili
# @Last Modified time: 2021-06-02 18:26:30
import math
import numpy as np
import time
from .base_solver import BaseSolver

class SinkhornSolver(BaseSolver):
	def __init__(self, opt):
		super(SinkhornSolver, self).__init__(opt)
		self.m, self.n = self.C.shape
		self.max_iter_step = opt['max_iter_step']
		self.iter_thre = opt['iter_thre']    
		self.epsilon = opt['epsilon']
		self.K = np.exp(-self.C/self.epsilon)
  
	def _init(self):
		m, n = self.m, self.n
		a = np.ones(m)
		b = np.ones(n)  
		return a,b

	def _update(self, a, b):
		a = self.mu / (self.K.dot(b))
		b = self.nu / (self.K.T.dot(a))
		return a, b

	def _recover(self, a, b):
		pi = a.reshape((self.m, 1)) * self.K * b.reshape((1, self.n))
		return pi

	def solve(self):
		a,b = self._init()
		tot_iter = 0
		verbose_step = int(self.iter_thre * self.max_iter_step)
		start = time.time()
		for i in range(self.max_iter_step):
			tot_iter += 1
			a,b = self._update(a, b)
			pi = self._recover(a, b)
			if tot_iter % verbose_step == 0:
				err_mu = np.linalg.norm(pi.sum(axis=1) - self.mu, 1)
				err_nu = np.linalg.norm(pi.sum(axis=0) - self.nu, 1)
				loss = (self.C * pi).sum()
				print('Total iteration: {0}, err_mu: {1}, err_nu: {2}, loss: {3}'.format(tot_iter, err_mu, err_nu, loss))
		loss = (self.C * pi).sum()
		tm = time.time() - start
		print('Time: {0}\nLoss: {1}'.format(tm,loss))
		return pi

