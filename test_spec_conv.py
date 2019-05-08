#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/15
@Author  : AnNing
"""
import h5py
from util import *


in_file = ''
with h5py.File(in_file, 'r') as h5:
    rad = h5.get('spectrum')[:]
