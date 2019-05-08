#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/15
@Author  : AnNing
"""
from data_loader import *
from hdf5 import write_hdf5_and_compress

in_file = '/home/cali/data/GapFilling/IASI/IASI_xxx_1C_M01_20161101235959Z_20161102000255Z_N_O_20161102003529Z__20161102003712'
out_file = '/home/cali/src/gap_filling/data/iasi_001.h5'

loader_iasi = LoaderIasiL1(in_file)
radiances = loader_iasi.get_spectrum_radiance()

data = {'spectrum': radiances[1]}

write_hdf5_and_compress(out_file, data)


