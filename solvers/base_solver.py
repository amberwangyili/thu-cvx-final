# -*- coding: utf-8 -*-
# @Author: amberwangyili    
# @Date:   2021-05-24 12:56:29
# @Last Modified by:   amberwangyili
# @Last Modified time: 2021-06-02 16:35:20

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
    


