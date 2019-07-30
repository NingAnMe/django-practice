#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/7/29
@Author  : AnNing
"""
from __future__ import print_function
import os
import sys
import numpy as np
import h5py


class LoaderHirasL1:
    def __init__(self, in_file):
        self.in_file = in_file

    def get_radiance(self, coeff_file=None):
        if coeff_file is None:
            return self.get_spectrum_radiance()
        else:
            return self.get_spectrum_radiance_full(coeff_file)

    def get_spectrum_radiance(self):
        with h5py.File(self.in_file, 'r') as h5r:
            sds_name = '/Data/ES_RealLW'
            real_lw = h5r.get(sds_name)[:]
            sds_name = '/Data/ES_RealMW1'
            real_mw = h5r.get(sds_name)[:]
            sds_name = '/Data/ES_RealMW2'
            real_sw = h5r.get(sds_name)[:]

        # 增加切趾计算
        w0 = 0.23
        w1 = 1 - 2 * w0
        w2 = w0
        real_lw[:, :, :, 1:-1] = w0 * real_lw[:, :, :, :-2] + w1 * real_lw[:, :, :, 1:-1] + w2 * real_lw[:, :, :, 2:]
        real_mw[:, :, :, 1:-1] = w0 * real_mw[:, :, :, :-2] + w1 * real_mw[:, :, :, 1:-1] + w2 * real_mw[:, :, :, 2:]
        real_sw[:, :, :, 1:-1] = w0 * real_sw[:, :, :, :-2] + w1 * real_sw[:, :, :, 1:-1] + w2 * real_sw[:, :, :, 2:]

        real_lw = real_lw[:, :, :, 2:-2]
        real_mw = real_mw[:, :, :, 2:-2]
        real_sw = real_sw[:, :, :, 2:-2]

        response = np.concatenate((real_lw, real_mw, real_sw), axis=3)
        response.reshape(-1, 2275)
        wave_lw = np.arange(650, 1135. + 0.625, 0.625)
        wave_mw = np.arange(1210., 1750. + 0.625, 0.625)
        wave_sw = np.arange(2155., 2550. + 0.625, 0.625)
        wave_number = np.concatenate((wave_lw, wave_mw, wave_sw))

        return wave_number, response

    def get_spectrum_radiance_full(self, coeff_file):
        """
        return 光谱波数和响应值，1维，2维
        """
        s = (3480, 1)
        # 增加切趾计算
        w0 = 0.23
        w1 = 1 - 2 * w0
        w2 = w0
        data_file = self.in_file
        with h5py.File(data_file, 'r') as h5r:
            sds_name = '/Data/ES_RealLW'
            real_lw = h5r.get(sds_name)[:]
            sds_name = '/Data/ES_RealMW1'
            real_mw = h5r.get(sds_name)[:]
            sds_name = '/Data/ES_RealMW2'
            real_sw = h5r.get(sds_name)[:]

        # 切趾计算 w0*n-1 + w1*n + w2*n+1 当作n位置的修正值
        # 开头和结尾不参与计算
        real_lw[:, :, :, 1:-1] = w0 * real_lw[:, :, :, :-2] + w1 * real_lw[:, :, :, 1:-1] + w2 * real_lw[:, :, :, 2:]
        real_mw[:, :, :, 1:-1] = w0 * real_mw[:, :, :, :-2] + w1 * real_mw[:, :, :, 1:-1] + w2 * real_mw[:, :, :, 2:]
        real_sw[:, :, :, 1:-1] = w0 * real_sw[:, :, :, :-2] + w1 * real_sw[:, :, :, 1:-1] + w2 * real_sw[:, :, :, 2:]

        real_lw = real_lw[:, :, :, 2:-2]
        real_mw = real_mw[:, :, :, 2:-2]
        real_sw = real_sw[:, :, :, 2:-2]

        # 波数范围和步长
        wave_number = np.arange(650., 2755.0 + 0.625, 0.625)

        # 响应值拼接起来 30*29*4*n
        response_old = np.concatenate((real_lw, real_mw, real_sw), axis=3)

        last_s = response_old.shape[-1]
        #  30*29*4*n 变成 30*29*4 = 3480 *n
        response_old = response_old.reshape(s[0], last_s)
        #                 self.test_w = wave_number_old
        #                 self.test_r = response_old
        #                 print '23', response_old.shape

        with h5py.File(coeff_file, 'r') as h5r:
            c0 = h5r.get('C0')[:]
            p0 = h5r.get('P0')[:]
            gap_num = h5r.get('GAP_NUM')[:]

        response_new = np.dot(response_old, p0)
        response_new = response_new + c0
        ch_part1 = gap_num[0]
        ch_part2 = gap_num[0] + gap_num[1]
        ch_part3 = gap_num[0] + gap_num[1] + gap_num[2]
        real_lw_e = response_new[:, 0:ch_part1]
        real_mw_e = response_new[:, ch_part1:ch_part2]
        real_sw_e = response_new[:, ch_part2:ch_part3]

        # 把原响应值 维度转成2维
        real_lw = real_lw.reshape(s[0], real_lw.shape[-1])
        real_mw = real_mw.reshape(s[0], real_mw.shape[-1])
        real_sw = real_sw.reshape(s[0], real_sw.shape[-1])
        response = np.concatenate((real_lw, real_lw_e, real_mw, real_mw_e, real_sw, real_sw_e), axis=1)

        return wave_number, response


def gap_filling(l1_file):
    print('输入：{}'.format(l1_file))

