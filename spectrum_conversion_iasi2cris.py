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
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

IBAND = [0, ]  # band 1、 2 or 3，光谱带

NIGHT = True

# #########  仪器参数 #############

IASI_F_NYQUIST = 6912.0  # 频带宽度  cm-1
IASI_RESAMPLE_MAXX = [2.0, ]  # cm OPD
IASI_D_FREQUENCY = [0.25, ]  # v cm-1  光谱分辨率
IASI_BAND_F1 = [645.25, ]  # 光谱带开始
IASI_BAND_F2 = [2760.25, ]  # 光谱带结束
IASI_FILTER_WIDTH = [20.0, ]  # cm-1  # COS过滤器过滤的宽度

CRIS_F_NYQUIST = 5875.0
CRIS_RESAMPLE_MAXX = [0.8, ]
CRIS_D_FREQUENCY = [0.625, ]
CRIS_BAND_F1 = [650.00, ]
CRIS_BAND_F2 = [2755.00, ]
CRIS_FILTER_WIDTH = [20.0, ]

# IASI_F_NYQUIST = 6912.0  # 频带宽度  cm-1
# IASI_RESAMPLE_MAXX = [2.0, 2.0, 2.0]  # cm OPD
# IASI_D_FREQUENCY = [0.25, 0.25, 0.25]  # v cm-1  光谱分辨率
# IASI_BAND_F1 = [645.00, 1210.00, 2000.0]  # 光谱带开始
# IASI_BAND_F2 = [1209.75, 1999.75, 2760.0]  # 光谱带结束
# IASI_FILTER_WIDTH = [20.0, 20.0, 20.0]

# CRIS_F_NYQUIST = 5875.0
# CRIS_RESAMPLE_MAXX = [0.8, 0.8, 0.8]
# CRIS_D_FREQUENCY = [0.625, 0.625, 0.625]
# CRIS_BAND_F1 = [650.00, 1210.00, 2155.0]
# CRIS_BAND_F2 = [1135.00, 1750.00, 2550.0]
# CRIS_FILTER_WIDTH = [20.0, 20.0, 20.0]
#
# HIRAS_F_NYQUIST = 5866.0
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

    spec_iasi2cris, wavenumber_iasi2cris, plot_data_iasi2cris = ori2other(
        radiance, IASI_BAND_F1[iband], IASI_BAND_F2[iband], IASI_D_FREQUENCY[iband],
        CRIS_BAND_F1[iband], CRIS_BAND_F2[iband], CRIS_D_FREQUENCY[iband],
        CRIS_F_NYQUIST, CRIS_RESAMPLE_MAXX[iband], CRIS_FILTER_WIDTH[iband],
        apodization_ori=iasi_apod, apodization_other=cris_apod,
    )

    spec1 = np.loadtxt(r'D:\nsmc\LBL\data\iasi_001.csv', delimiter=',')

    plt.plot(spec_iasi2cris - spec1)
    plt.tight_layout()
    plt.savefig('002.png')
    plt.show()

    print('sum: ', sum(spec_iasi2cris - spec1))

    np.savetxt(r'D:\nsmc\LBL\data\iasi_002.csv', spec_iasi2cris, delimiter=',')


def main(date):
    """
    :param date:
    # :param dir_in: 输入目录路径
    # :param dir_out:  输出目录路径
    :return:
    """
    dir_ins = ['/home/cali/data/GapFilling/IASI', ]
    dates = ['20160110', '20160406', '20160626', '20161101']

    dir_out1 = '/home/cali/data/GapFilling/CRISFull'
    dir_out2 = '/home/cali/data/GapFilling/CRISFull_validate'
    for dir_in in dir_ins:
        in_files = os.listdir(dir_in)
        in_files.sort()
        for in_file in in_files:
            if date not in in_file:
                continue
            dir_out = None
            for d in dates:
                if d in in_file:
                    dir_out = dir_out1
                    break
            if dir_out is None:
                dir_out = dir_out2
            in_file = os.path.join(dir_in, in_file)
            print('<<< {}'.format(in_file))
            out_filename = os.path.basename(in_file)
            out_file = os.path.join(dir_out, out_filename)
            if not os.path.isfile(out_file):
                iasi2cris(in_file, out_file)
            else:
                print("already exist: {}".format(out_file))


def iasi2cris(in_file, out_file):
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
            if sz <= 90:
                print('!!! Origin data is not night data! continue.')
                continue

        spec_iasi2cris, wavenumber_iasi2cris, plot_data_iasi2cris = ori2other(
            radiance, IASI_BAND_F1[iband], IASI_BAND_F2[iband], IASI_D_FREQUENCY[iband],
            CRIS_BAND_F1[iband], CRIS_BAND_F2[iband], CRIS_D_FREQUENCY[iband],
            CRIS_F_NYQUIST, CRIS_RESAMPLE_MAXX[iband], CRIS_FILTER_WIDTH[iband],
            apodization_ori=iasi_apod, apodization_other=cris_apod,
        )
        spec_iasi2cris = spec_iasi2cris.reshape(1, -1)

        # 如果转换后的响应值中存在无效值，不进行输出
        condition = radiance <= 0
        idx = np.where(condition)[0]
        if len(idx) > 0:
            print('!!! Transformation data Has invalid data! continue.')
            continue

        if 'spectrum_radiance' not in result_out:
            result_out['spectrum_radiance'] = spec_iasi2cris
        else:
            concatenate = (result_out['spectrum_radiance'], spec_iasi2cris)
            result_out['spectrum_radiance'] = np.concatenate(concatenate, axis=0)

        if 'spectrum_wavenumber' not in result_out:
            result_out['spectrum_wavenumber'] = wavenumber_iasi2cris.reshape(-1,)

    if 'spectrum_radiance' in result_out and 'spectrum_wavenumber' in result_out:
        print(radiances.shape)
        print(result_out['spectrum_radiance'].shape)
        write_hdf5_and_compress(out_file, result_out)


def get_noise(name='IASI'):
    """
    获取 IASI 的噪声数据
    mean 为 0，sigma 为 给定值的正态分布
    :return:
    """
    if name == 'IASI':
        current_dir = os.path.abspath(os.path.curdir)
        iasi_file = os.path.join(current_dir, 'data', 'iasi_instrument_noise.nc')
        iasi_nc = Dataset(iasi_file)
        noise = iasi_nc.variables['iasi_noise'][:]
        iasi_nc.close()
    else:
        return None
    noise_gauss = create_gauss_noise(noise)

    return noise_gauss


def create_gauss_noise(sigmas):
    """
    获取 mean 为 0， sigma 为给定值的正态分布
    :param sigmas:
    :return:
    """
    mean = 0
    number = 1
    noises = np.array([], dtype=np.float32)
    for sigma in sigmas:
        noise = np.random.normal(mean, sigma, number)
        noises = np.append(noises, noise)
    return noises


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

    if len(ARGS) != 1:
        print(HELP_INFO)
        sys.exit(-1)
    else:
        ARG1 = ARGS[0]
        main(ARG1)
# if __name__ == '__main__':
#     conv()
