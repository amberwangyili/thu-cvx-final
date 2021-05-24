# -*- coding: utf-8 -*-
# @Author: Amber
# @Date:   2021-05-24 12:50:07
# @Last Modified by:   yiliwang
# @Last Modified time: 2021-05-24 15:15:08
import logging
logger = logging.getLogger('base')


def create_solver(opt):
    solver = opt['solver']
    if solver == 'Gurobi':  
        from .Gurobi_solver import GurobiSolver as S
    elif solver == 'Mosek':
        from .Mosek_solver import MosekSolver as S
    else:
        raise NotImplementedError('Solver [{:s}] not recognized.'.format(solver))
    s = S(opt)
    logger.info('Model [{:s}] is created.'.format(s.__class__.__name__))
    return s


