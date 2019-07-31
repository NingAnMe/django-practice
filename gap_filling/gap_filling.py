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
from shutil import copyfile


class LoaderHirasL1:
    def __init__(self, in_file):
        self.in_file = in_file

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
        response = response.reshape((30, 29, 4, -1))

        return wave_number, response


class LoaderGiirsL1:
    def __init__(self, in_file):
        self.in_file = in_file

    def get_spectrum_radiance_full(self, coeff_file):
        """
        return 光谱波数和响应值，1维，2维
        """
        # 增加切趾计算
        w0 = 0.23
        w1 = 1 - 2 * w0
        w2 = w0
        data_file = self.in_file
        with h5py.File(data_file, 'r') as h5r:
            sds_name = 'ES_RealLW'
            real_lw = h5r.get(sds_name)[:].T

            sds_name = 'ES_RealMW'
            real_mw = h5r.get(sds_name)[:].T

        # 切趾计算 w0*n-1 + w1*n + w2*n+1 当作n位置的修正值
        # 开头和结尾不参与计算
        real_lw[:, 1:-1] = w0 * real_lw[:, :-2] + w1 * real_lw[:, 1:-1] + w2 * real_lw[:, 2:]
        real_mw[:, 1:-1] = w0 * real_mw[:, :-2] + w1 * real_mw[:, 1:-1] + w2 * real_mw[:, 2:]

        real_lw = real_lw[:, 2:-2][:, 94:559]
        real_mw = real_mw[:, 2:-2][:, 238:860]

        # 波数范围和步长
        wave_number = np.arange(700., 2250.0 + 0.625, 0.625)

        # 响应值拼接起来
        response_old = np.concatenate((real_lw, real_mw), axis=1)

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
        response = np.concatenate((real_lw_e, real_lw, real_mw_e, real_mw, real_sw_e), axis=1)
        response = response.reshape((128, -1))

        return wave_number, response


def gap_filling_giras(in_file, out_file):
    coeff_file = 'giirs_fs.GapCoeff.model'
    try:
        datas = LoaderGiirsL1(in_file)
        wavenumber, response = datas.get_spectrum_radiance_full(coeff_file)
    except Exception as why:
        print('读取L1数据出错:{}'.format(why))
        return

    if not os.path.isfile(out_file):
        copyfile(in_file, out_file)

    compression = 'gzip'
    compression_opts = 5
    shuffle = True
    try:
        with h5py.File(out_file, 'a') as hdf5:
            hdf5.create_dataset('ES_Full',
                                dtype=np.float32, data=response, compression=compression,
                                compression_opts=compression_opts,
                                shuffle=shuffle)
        with h5py.File(out_file, 'a') as hdf5:
            hdf5.create_dataset('WL_Full',
                                dtype=np.float32, data=wavenumber, compression=compression,
                                compression_opts=compression_opts,
                                shuffle=shuffle)
    except Exception as why:
        print('GapFilling的过程出错： {}'.format(why))
        os.remove(out_file)
        return


def gap_filling_hiras(in_file, out_file):
    coeff_file = 'hiras_fs.GapCoeff.model'
    try:
        datas = LoaderHirasL1(in_file)
        wavenumber, response = datas.get_spectrum_radiance_full(coeff_file)
    except Exception as why:
        print('读取L1数据出错:{}'.format(why))
        return

    if not os.path.isfile(out_file):
        copyfile(in_file, out_file)

    compression = 'gzip'
    compression_opts = 5
    shuffle = True
    try:
        with h5py.File(out_file, 'a') as hdf5:
            hdf5.create_dataset('/Data/ES_Full',
                                dtype=np.float32, data=response, compression=compression,
                                compression_opts=compression_opts,
                                shuffle=shuffle)
        with h5py.File(out_file, 'a') as hdf5:
            hdf5.create_dataset('/Data/WL_Full',
                                dtype=np.float32, data=wavenumber, compression=compression,
                                compression_opts=compression_opts,
                                shuffle=shuffle)
    except Exception as why:
        print('GapFilling的过程出错： {}'.format(why))
        os.remove(out_file)
        return


def gap_filling(l1_file, out_file):
    print('输入：{}'.format(l1_file))
    if not os.path.isfile(l1_file):
        raise '输入文件不存在，请检查:{}'.format(l1_file)
    file_name = os.path.basename(l1_file)
    if 'GIIRS' in file_name:
        gap_filling_giras(l1_file, out_file)
        print('完成：{}'.format(out_file))
    elif 'HIRAS' in file_name:
        gap_filling_hiras(l1_file, out_file)
        print('完成：{}'.format(out_file))
    else:
        print('不支持此L1文件')


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 2:
        print("""
        python gap_filling <l1_file> <out_file>
        在原来的数据文件中添加新的数据集：ES_Full和WL_Full,为切趾之后的数据
        """)
    gap_filling(args[0], args[1])
