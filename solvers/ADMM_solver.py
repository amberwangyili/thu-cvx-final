# -*- coding: utf-8 -*-
# @Author: Mengfei Xia
# @Date:   2021-05-25 12:39:07
# @Last Modified by:   amberwangyili
# @Last Modified time: 2021-06-02 18:20:15
import time
import math
import numpy as np
from .base_solver import BaseSolver


class ADMMSolver(BaseSolver):
  def __init__(self, opt):
    super(ADMMSolver, self).__init__(opt)
    self.m, self.n = self.C.shape
    self.method = opt['method']
    self.max_iter_step = opt['max_iter_step']
    self.rhos = opt['rhos']
    self.iter_thre = opt['iter_thre']
    self.alpha = opt['alpha']

  def _init_primal(self):
    m, n = self.m, self.n
    x, x_hat = np.zeros((m, n)), np.zeros((m, n))
    e = np.zeros_like(x)
    lamb = np.zeros(m)
    eta = np.zeros(n)
    return x, x_hat, e, lamb, eta
  
  def _update_primal(self, m, n, mu, nu, C, x, x_hat, e, lamb, eta, rho, alpha):
    r = (-e + lamb.reshape((m, 1)) + eta.reshape((1, n)) - C) / rho + \
      mu.reshape((m, 1)) + \
      nu.reshape((1, n)) + \
      x_hat
    
    x = r - ((r.sum(axis=1) - r.sum() / (m+n+1)) / (n+1)).reshape((m, 1)) - \
      ((r.sum(axis=0) - r.sum() / (m+n+1)) / (m+1)).reshape((1, n))

    x_hat = np.maximum(x + e/rho, 0.)
    lamb = lamb + alpha * rho * (mu - x.sum(axis=1))
    eta = eta + alpha * rho * (nu - x.sum(axis=0))
    e = e + alpha * rho * (x - x_hat)
    return x, x_hat, e, lamb, eta

  def _solve_primal(self):
    x, x_hat, e, lamb, eta = self._init_primal()
    m, n = self.m, self.n
    tot_iter = 0
    for rho in self.rhos:
      for i in range(self.max_iter_step):
        tot_iter += 1
        x, x_hat, e, lamb, eta = self._update_primal(m, n, self.mu, self.nu, self.C, x, x_hat, e, lamb, eta, rho, self.alpha)
        if tot_iter % 500 == 0:
          err_mu = np.linalg.norm(x_hat.sum(axis=1) - self.mu, 1)
          err_nu = np.linalg.norm(x_hat.sum(axis=0) - self.nu, 1)
          loss = (self.C * x_hat).sum()
          print('Total iteration: {0}, err_mu: {1}, err_nu: {2}, loss: {3}'.format(tot_iter, err_mu, err_nu, loss))

    return x_hat

  def _init_dual(self):
    m, n = self.m, self.n
    lamb = np.zeros(m)
    eta = np.zeros(n)
    d = np.zeros((m, n))
    e = self.C - lamb.reshape((m, 1)) - eta.reshape((1, n))
    return d, e, lamb, eta

  def _update_dual(self, m, n, mu, nu, C, d, e, lamb, eta, rho, alpha):
    lamb = ((mu + d.sum(axis=1)) / rho - eta.sum() - e.sum(axis=1) + C.sum(axis=1)) / n
    eta = ((nu + d.sum(axis=0)) / rho - lamb.sum() - e.sum(axis=0) + C.sum(axis=0)) / m
    e = d / rho + C - lamb.reshape((m, 1)) - eta.reshape((1, n))
    e = np.maximum(e, 0.)
    d = d + alpha * (C - lamb.reshape((m, 1)) - eta.reshape((1, n)) - e)
    return d, e, lamb, eta

  def _solve_dual(self):
    d, e, lamb, eta = self._init_dual()
    m, n = self.m, self.n
    tot_iter = 0
    verbose_step = int(self.iter_thre * self.max_iter_step)
    for rho in self.rhos:
      for i in range(self.max_iter_step):
        tot_iter += 1
        d, e, lamb, eta = self._update_dual(m, n, self.mu, self.nu, self.C, d, e, lamb, eta, rho, self.alpha)
        if tot_iter % verbose_step == 0:
          x_hat = -d
          err_mu = np.linalg.norm(x_hat.sum(axis=1) - self.mu, 1)
          err_nu = np.linalg.norm(x_hat.sum(axis=0) - self.nu, 1)
          loss = (self.C * x_hat).sum()
          print('Total iteration: {0}, err_mu: {1}, err_nu: {2}, loss: {3}'.format(tot_iter, err_mu, err_nu, loss))

    return -d

  def solve(self):
    start = time.time()

    ans = self._solve_primal() if self.method == 'Primal' else self._solve_dual()

    loss = (self.C * ans).sum()
    tm = time.time() - start
    print('Time: {0}\nLoss: {1}'.format(tm,loss))
    return ans




class EADMMSolver(ADMMSolver):
  def __init__(self, opt):
    super(EADMMSolver, self).__init__(opt)
    self.epsilon = opt['epsilon']
    self.delta = opt['delta']

  def _update_primal(self, m, n, mu, nu, C, x, x_hat, e, lamb, eta, rho, alpha):
    r = (-e + lamb.reshape((m, 1)) + eta.reshape((1, n)) - C) / rho + \
      mu.reshape((m, 1)) + \
      nu.reshape((1, n)) - \
      self.epsilon * np.log(x_hat + self.delta)+ \
      x_hat
    
    x = r - ((r.sum(axis=1) - r.sum() / (m+n+1)) / (n+1)).reshape((m, 1)) - \
      ((r.sum(axis=0) - r.sum() / (m+n+1)) / (m+1)).reshape((1, n))

    x_hat = np.maximum(x + e/rho, 0.)
    lamb = lamb + alpha * rho * (mu - x.sum(axis=1))
    eta = eta + alpha * rho * (nu - x.sum(axis=0))
    e = e + alpha * rho * (x - x_hat)
    return x, x_hat, e, lamb, eta

  def solve(self):
    x, x_hat, e, lamb, eta = self._init_primal()
    m, n = self.m, self.n
    tot_iter = 0
    verbose_step = int(self.iter_thre * self.max_iter_step)
    start = time.time()
    for rho in self.rhos:
      for i in range(self.max_iter_step):
        tot_iter += 1
        x, x_hat, e, lamb, eta = self._update_primal(m, n, self.mu, self.nu, self.C, x, x_hat, e, lamb, eta, rho, self.alpha)
        if tot_iter % verbose_step == 0:
          err_mu = np.linalg.norm(x_hat.sum(axis=1) - self.mu, 1)
          err_nu = np.linalg.norm(x_hat.sum(axis=0) - self.nu, 1)
          loss = (self.C * x_hat).sum()
          print('Total iteration: {0}, err_mu: {1}, err_nu: {2}, loss: {3}'.format(tot_iter, err_mu, err_nu, loss))
    loss = (self.C * x_hat).sum()
    tm = time.time() - start
    print('Time: {0}\nLoss: {1}'.format(tm,loss))
    return x_hat




