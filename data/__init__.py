# -*- coding: utf-8 -*-
# @Author: Amber
# @Date:   2021-05-24 13:50:30
# @Last Modified by:   yiliwang
# @Last Modified time: 2021-05-24 14:30:10



def create_sampler(opt):
    mode = opt['mode']    
    if mode == 'Naive':
        from .naive_sampler import NaiveSampler as G
    else:
        raise NotImplementedError('Mode [{:s}] is not recognized.'.format(mode))
    data = G()
    return data
