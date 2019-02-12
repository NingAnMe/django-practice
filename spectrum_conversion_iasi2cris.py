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

IBAND = [0, ]  # band 1、 2 or 3，光谱带

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


def main(dir_in, dir_out):
    """
    :param dir_in: 输入目录路径
    :param dir_out:  输出目录路径
    :return:
    """
    in_files = os.listdir(dir_in)

    for in_file in in_files:
        in_file = os.path.join(dir_in, in_file)
        print('<<< {}'.format(in_file))
        out_filename = os.path.basename(in_file)
        out_file = os.path.join(dir_out, out_filename)
        iasi2cris(in_file, out_file)


def iasi2cris(in_file, out_file):
    """
    :param in_file: IASI L1原始数据绝对路径，全光谱分辨率
    :param out_file: CRIS 数据绝对路径
    :return:
    """
    loader_iasi = LoaderIasiL1(in_file)
    radiances = loader_iasi.get_spectrum_radiance()
    iband = 0

    result_out = dict()

    for radiance in radiances:
        radiance = radiance[:8461]  # 后面都是无效值
        # 如果响应值中存在无效值，不进行转换
        idx = np.where(radiance <= 0)[0]
        if len(idx) > 0:
            continue

        spec_iasi2cris, wavenumber_iasi2cris, plot_data_iasi2cris = ori2other(
            radiance, IASI_BAND_F1[iband], IASI_BAND_F2[iband], IASI_D_FREQUENCY[iband],
            CRIS_BAND_F1[iband], CRIS_BAND_F2[iband], CRIS_D_FREQUENCY[iband],
            CRIS_F_NYQUIST, CRIS_RESAMPLE_MAXX[iband], CRIS_FILTER_WIDTH[iband],
            apodization_ori=iasi_apod, apodization_other=cris_apod,
        )
        spec_iasi2cris = spec_iasi2cris.reshape(1, -1)
        # 如果转换后的响应值中存在无效值，不进行输出
        idx = np.where(spec_iasi2cris <= 0)[0]
        if len(idx) > 0:
            continue

        if 'spectrum_radiance' not in result_out:
            result_out['spectrum_radiance'] = spec_iasi2cris
        else:
            concatenate = (result_out['spectrum_radiance'], spec_iasi2cris)
            result_out['spectrum_radiance'] = np.concatenate(concatenate, axis=0)

        if 'spectrum_wavenumber' not in result_out:
            result_out['spectrum_wavenumber'] = wavenumber_iasi2cris.reshape(-1,)

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
        ARG1 = ARGS[0]
        ARG2 = ARGS[1]
        main(ARG1, ARG2)
