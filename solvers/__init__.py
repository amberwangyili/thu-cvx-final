# -*- coding: utf-8 -*-
# @Author: amberwangyili
# @Date:   2021-05-24 12:50:07
# @Last Modified by:   amberwangyili
# @Last Modified time: 2021-06-02 18:24:17
import logging
logger = logging.getLogger('base')


def create_solver(opt):
    solver = opt['solver']
    is_regularized = opt['is_regularized']

    if solver == 'Gurobi':  
        from .Gurobi_solver import GurobiSolver as S
    elif solver == 'Mosek':
        from .Mosek_solver import MosekSolver as S
    elif solver == 'ADMM':
        if is_regularized:
            from .ADMM_solver import EADMMSolver as S    
        else:
            from .ADMM_solver import ADMMSolver as S
    elif solver == 'Sinkhorn':
        from .Sinkhorn_solver import SinkhornSolver as S    
    else:
        raise NotImplementedError('Solver [{:s}] not recognized.'.format(solver))
    s = S(opt)
    logger.info('Model [{:s}] is created.'.format(s.__class__.__name__))
    return s


