# -*- coding: utf-8 -*-
# @Author: amberwangyili
# @Date:   2021-05-24 13:44:13
# @Last Modified by:   amberwangyili
# @Last Modified time: 2021-06-02 17:14:47
import os
import math
import random
import argparse
from solvers import create_solver
from data import create_sampler
from utils import util

def main():
    #### options
	parser = argparse.ArgumentParser()
	parser.add_argument('-config', type=str, help='Path to YAML config')
	args = parser.parse_args()
	cfg = util.parse(args.config)

	#generate random data
	sampler = create_sampler(cfg)
	cfg["mu"], cfg['nu'] = sampler.sample_weight(cfg['dim_n'], cfg['dim_m'])
	cfg["C"] = sampler.sample_cost(cfg['dim_n'],cfg['dim_m'])

	solver = create_solver(cfg)
	solver.solve()



if __name__ == '__main__':
	main()

    