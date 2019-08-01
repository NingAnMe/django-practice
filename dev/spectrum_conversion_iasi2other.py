#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2018/12/20
"""
import sys
from spectrum_conversion import *
from util import *
from data_loader import *
from hdf5 import write_hdf5_and_compress
import numpy as np
import matplotlib.pyplot as plt

IBAND = [0, ]  # band 1、 2 or 3，光谱带

NIGHT = False

# #########  仪器参数 #############

IASI_F_NYQUIST = 6912.0  # 频带宽度  cm-1
IASI_RESAMPLE_MAXX = [2.0, ]  # cm OPD
IASI_D_FREQUENCY = [0.25, ]  # v cm-1  光谱分辨率
IASI_BAND_F1 = [645.25, ]  # 光谱带开始
IASI_BAND_F2 = [2760.25, ]  # 光谱带结束
IASI_FILTER_WIDTH = [20.0, ]  # cm-1  # COS过滤器过滤的宽度

GIIRS_F_NYQUIST = 5875.0
GIIRS_RESAMPLE_MAXX = [0.8, ]
GIIRS_D_FREQUENCY = [0.625, ]
GIIRS_BAND_F1 = [645.625, ]
GIIRS_BAND_F2 = [2760., ]
GIIRS_FILTER_WIDTH = [20.0, ]

# IASI_F_NYQUIST = 6912.0  # 频带宽度  cm-1
# IASI_RESAMPLE_MAXX = [2.0, 2.0, 2.0]  # cm OPD
# IASI_D_FREQUENCY = [0.25, 0.25, 0.25]  # v cm-1  光谱分辨率
# IASI_BAND_F1 = [645.00, 1210.00, 2000.0]  # 光谱带开始
# IASI_BAND_F2 = [1209.75, 1999.75, 2760.0]  # 光谱带结束
# IASI_FILTER_WIDTH = [20.0, 20.0, 20.0]

# CRIS_F_NYQUIST = 5866.0
# CRIS_RESAMPLE_MAXX = [0.8, 0.8, 0.8]
# CRIS_D_FREQUENCY = [0.625, 0.625, 0.625]
# CRIS_BAND_F1 = [650.00, 1210.00, 2155.0]
# CRIS_BAND_F2 = [1135.00, 1750.00, 2550.0]
# CRIS_FILTER_WIDTH = [20.0, 20.0, 20.0]

# HIRAS_F_NYQUIST = 5875.0
# HIRAS_RESAMPLE_MAXX = [0.8, 0.8, 0.8]
# HIRAS_D_FREQUENCY = [0.625, 1.25, 2.5]
# HIRAS_BAND_F1 = [650.00, 1210.00, 2155.0]
# HIRAS_BAND_F2 = [1095.00, 1750.00, 2550.0]
# HIRAS_FILTER_WIDTH = [20.0, 24.0, 20.0]


def conv():
    iband = 0
    in_file = r'D:\nsmc\LBL\data\iasi_001.h5'
    with h5py.File(in_file, 'r') as h5:
        radiance = h5.get('spectrum')[:8461]

    spec_iasi2giirs, wavenumber_iasi2giirs, plot_data_iasi2giirs = iasi2hiras(
        radiance, IASI_BAND_F1[iband], IASI_BAND_F2[iband], IASI_D_FREQUENCY[iband],
        GIIRS_BAND_F1[iband], GIIRS_BAND_F2[iband], GIIRS_D_FREQUENCY[iband],
        GIIRS_F_NYQUIST, GIIRS_RESAMPLE_MAXX[iband], GIIRS_FILTER_WIDTH[iband],
        apodization_ori=iasi_apod,)

    print(spec_iasi2giirs.size)

    spec1 = np.loadtxt(r'D:\nsmc\LBL\data\iasi_001.csv', delimiter=',')

    # plt.plot(rad2tbb(spec_iasi2giirs, wavenumber_iasi2giirs))
    # plt.tight_layout()
    # plt.savefig('003.png')
    # plt.show()
    #
    # frequency = np.arange(0, 8461, dtype=np.float64) * 0.25 + 645.25
    #
    # plt.plot(rad2tbb(radiance, frequency))
    # plt.tight_layout()
    # plt.savefig('004.png')
    # plt.show()

    plt.plot(spec_iasi2giirs - spec1)
    plt.tight_layout()
    plt.savefig('004.png')
    plt.show()

    print('sum: ', sum(spec_iasi2giirs - spec1))

    np.savetxt(r'D:\nsmc\LBL\data\iasi_002.csv', spec_iasi2giirs, delimiter=',')


def main(dir_in, dir_out):
    """
    :param dir_in: 输入目录路径
    :param dir_out:  输出目录路径
    :return:
    """
    in_files = os.listdir(dir_in)
    in_files.sort()
    for in_file in in_files:
        print('<<< {}'.format(in_file))
        out_filename = os.path.basename(in_file)
        out_file = os.path.join(dir_out, out_filename)
        if not os.path.isfile(out_file):
            iasi2giirs(in_file, out_file)
        else:
            print("already exist: {}".format(out_file))


def iasi2giirs(in_file, out_file):
    """
    :param in_file: IASI L1原始数据绝对路径，全光谱分辨率
    :param out_file: CRIS 数据绝对路径
    :return:
    """
    loader_iasi = LoaderIasiL1(in_file)
    radiances = loader_iasi.get_spectrum_radiance()
    sun_zenith = loader_iasi.get_sun_zenith()
    iband = 0

    result_out = dict()

    for i, radiance in enumerate(radiances):
        print('Count: ', i)
        radiance = radiance[:8461]  # 后面都是无效值
        # 如果响应值中存在无效值，不进行转换
        condition = radiance <= 0
        idx = np.where(condition)[0]
        if len(idx) > 0:
            print('!!! Origin data has invalid data! continue.')
            continue

        # 如果 night = True 那么只处理晚上数据
        if NIGHT:
            sz = sun_zenith[i]
            if sz <= 120:
                print('!!! Origin data is not night data! continue.')
                continue

        spec_iasi2giirs, wavenumber_iasi2giirs, plot_data_iasi2giirs = iasi2hiras(
            radiance, IASI_BAND_F1[iband], IASI_BAND_F2[iband], IASI_D_FREQUENCY[iband],
            GIIRS_BAND_F1[iband], GIIRS_BAND_F2[iband], GIIRS_D_FREQUENCY[iband],
            GIIRS_F_NYQUIST, GIIRS_RESAMPLE_MAXX[iband], GIIRS_FILTER_WIDTH[iband],
            apodization_ori=iasi_apod, )
        spec_iasi2giirs = spec_iasi2giirs.reshape(1, -1)

        # 如果转换后的响应值中存在无效值，不进行输出
        condition = radiance <= 0
        idx = np.where(condition)[0]
        if len(idx) > 0:
            print('!!! Transformation data Has invalid data! continue.')
            continue

        if 'spectrum_radiance' not in result_out:
            result_out['spectrum_radiance'] = spec_iasi2giirs
        else:
            concatenate = (result_out['spectrum_radiance'], spec_iasi2giirs)
            result_out['spectrum_radiance'] = np.concatenate(concatenate, axis=0)

        if 'spectrum_wavenumber' not in result_out:
            result_out['spectrum_wavenumber'] = wavenumber_iasi2giirs.reshape(-1,)

    if 'spectrum_radiance' in result_out and 'spectrum_wavenumber' in result_out:
        print(radiances.shape)
        print(result_out['spectrum_radiance'].shape)
        write_hdf5_and_compress(out_file, result_out)


# ######################## 带参数的程序入口 ##############################
if __name__ == "__main__":
    # 获取程序参数接口
    ARGS = sys.argv[1:]
    HELP_INFO = \
        u"""
        [arg1]：dir_in
        [arg2]：dir_out
        [example]： python app.py arg1 arg2
        """
    if "-h" in ARGS:
        print(HELP_INFO)
        sys.exit(-1)

    if len(ARGS) != 2:
        print(HELP_INFO)
        sys.exit(-1)
    else:
        main(*ARGS)
# if __name__ == '__main__':
#     conv()
