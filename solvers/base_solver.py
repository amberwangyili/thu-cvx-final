# -*- coding: utf-8 -*-
# @Author: Amber
# @Date:   2021-05-24 12:56:29
# @Last Modified by:   yiliwang
# @Last Modified time: 2021-05-24 15:09:46

import os
from collections import OrderedDict

class BaseSolver():
    def __init__(self, opt):
        self.opt = opt
        self.mu = opt['mu']
        self.nu = opt['nu']
        self.C = opt['C']

    def solve(self):
        pass
    


