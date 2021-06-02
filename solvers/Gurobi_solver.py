# -*- coding: utf-8 -*-
# @Author: amberwangyili
# @Date:   2021-05-24 12:39:07
# @Last Modified by:   amberwangyili
# @Last Modified time: 2021-06-02 16:35:30
from gurobipy import *
import time
import math
import numpy as np
from .base_solver import BaseSolver


method2paras = {'Primal': 0,'Dual': 1,'Intpnt':2}



class GurobiSolver(BaseSolver):
	def __init__(self,opt):
		super(GurobiSolver,self).__init__(opt)
		self.method = method2paras[opt['method']]
		m,n = self.C.shape		
		self.M = Model("OT")
		self.M.setParam(GRB.Param.Method,self.method)
		self.M.setParam('OutputFlag', False) 
		self.res = self.M.addVars(m, n, lb=0., ub=GRB.INFINITY)
		self.M.addConstrs(LinExpr([(1., self.res[i, j]) for j in range(n)]) == self.mu[i] for i in range(m))
		self.M.addConstrs(LinExpr([(1., self.res[i, j]) for i in range(m)]) == self.nu[j] for j in range(n))
		self.M.setObjective(LinExpr([(self.C[i, j], self.res[i, j]) for i in range(m) for j in range(n)]))

	def solve(self):
		m,n = self.C.shape
		start = time.time()
		self.M.optimize()
		tm = time.time() - start
		sx = self.M.getAttr("x", self.res)
		pi = np.array([sx[i, j] for i in range(m) for j in range(n)]).reshape(m, n)
		loss = (self.C * pi).sum()
		print('Time: {0}\nLoss: {1}'.format(tm,loss))
		return pi

