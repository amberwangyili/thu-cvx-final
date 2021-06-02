# -*- coding: utf-8 -*-
# @Author: amberwangyili
# @Date:   2021-05-24 13:08:19
# @Last Modified by:   amberwangyili
# @Last Modified time: 2021-06-02 16:35:51
import os
import sys
import time
import math
from collections import OrderedDict

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def streamprinter(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()
    

def OrderedYaml():

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(opt_path):
    Loader, Dumper = OrderedYaml()
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    return opt