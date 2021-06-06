# -*- coding: utf-8 -*-
# @Author: Amber
# @Date:   2021-05-24 13:50:30
# @Last Modified by:   Mengfei Xia
# @Last Modified time: 2021-05-24 14:30:10



def create_sampler(opt):
  mode = opt['mode']  
  if mode == 'Random':
    from .random_sampler import RandomSampler as G
  elif mode == 'DOT':
    from .DOT_sampler import DOTSampler as G
  elif mode == 'Ellip':
    from .ellipses_sampler import EllipsesSampler as G
  elif mode == 'Caffa':
    from .caffarelli_sampler import CaffarelliSampler as G
  elif mode == 'Load':
    from .load_sampler import LoadSampler as G
  elif mode == 'Ricker':
    from .Ricker_sampler import RickerSampler as G
  else:
    raise NotImplementedError('Mode [{:s}] is not recognized.'.format(mode))
  data = G(opt)
  return data
