# -*- coding: utf-8 -*-
# @Author: Amber
# @Date:   2021-05-24 12:39:07
# @Last Modified by:   yiliwang
# @Last Modified time: 2021-05-24 14:54:58
import mosek
import time
import math
import numpy as np
from utils.util import streamprinter
from collections import OrderedDict
from .base_solver import BaseSolver


method2paras = {'Primal': [mosek.optimizertype.primal_simplex,mosek.soltype.bas],
				'Dual': [mosek.optimizertype.dual_simplex,mosek.soltype.bas],
				'Intpnt':[mosek.optimizertype.intpnt,mosek.soltype.itr]}



class MosekSolver(BaseSolver):
	def __init__(self,opt):
		super(MosekSolver,self).__init__(opt)
		self.method, self.soltype  = method2paras[opt['method']]
		self.env = mosek.Env()
		self.task = self.env.Task()
		self._set_task()

	def _set_task(self):
		m,n = self.C.shape
		inf = 0.0
		self.task.set_Stream(mosek.streamtype.log, streamprinter)
		self.task.putintparam(mosek.iparam.optimizer,self.method)
		self.task.appendvars(m*n)
		self.task.appendcons(m+n)
		self.task.putvarboundlist(range(m*n),[mosek.boundkey.lo]*(m*n),[0.]*(m*n),[inf]*(m*n))
		for i in range(m):
			self.task.putarow(i,range(i*n, (i+1)*n),[1.]*n)
		self.task.putconboundlist(range(0, m),[mosek.boundkey.fx]*m,self.mu,self.mu)
		for i in range(n):
			self.task.putarow(i+m,range(i, i+m*n, n),[1.]*m)
		self.task.putconboundlist(range(m, m+n),[mosek.boundkey.fx]*n,self.nu,self.nu)
		self.task.putclist(range(m*n), self.C.reshape(m*n))
		self.task.putobjsense(mosek.objsense.minimize)



	def solve(self):
		m,n = self.C.shape
		res = [0.] * (m * n)
		self.task.optimize()
		self.task.getxx(self.soltype, res)
		res = np.array(res).reshape(m, n)
		tm = self.task.getdouinf(mosek.dinfitem.optimizer_time)
		it = self.task.getintinf(mosek.iinfitem.intpnt_iter)
		loss = (self.C * res).sum()
		print('Time: {0}\nIterations: {1}\nLoss: {2}'.format(tm,it,loss))
		return res

