#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2018/12/20
"""
from conversion import *
from plot import *
from util import *

IBAND = [0, ]  # band 1、 2 or 3，光谱带
LBL_DIR = r'D:\nsmc\gap_filling_data'

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


def main():
    iasi_file = os.path.join(LBL_DIR, 'iasi_20180104.h5')
    with h5py.File(iasi_file, 'r') as h5r:
        responses = h5r.get('wavenumber_radiance').value * 10 ** 5

    spec_iasi = responses[0][0:8461]

    iband = 0

    plot = True

    # ################### Compute IASI to CrIS spectrum #############
    conversion_name = 'pic/IASI2CRIS_all'
    spec_iasi2cris, wavenumber_iasi2cris, plot_data_iasi2cris = ori2other(
        spec_iasi, IASI_BAND_F1[iband], IASI_BAND_F2[iband], IASI_D_FREQUENCY[iband],
        CRIS_BAND_F1[iband], CRIS_BAND_F2[iband], CRIS_D_FREQUENCY[iband],
        CRIS_F_NYQUIST, CRIS_RESAMPLE_MAXX[iband], CRIS_FILTER_WIDTH[iband],
        apodization_ori=iasi_apod, apodization_other=cris_apod,
        plot=plot,
    )
    print(wavenumber_iasi2cris[0], wavenumber_iasi2cris[-1])
    statistics_print(spec_iasi2cris)
    plot_conversion_picture(plot_data_iasi2cris, conversion_name)


# def iasi2cris(in_file, out_file):
#     conversion_name = 'pic/IASI2CRIS_all'
#     spec_iasi2cris, wavenumber_iasi2cris, plot_data_iasi2cris = ori2other(
#         spec_lbl2iasi, IASI_BAND_F1[iband], IASI_BAND_F2[iband], IASI_D_FREQUENCY[iband],
#         CRIS_BAND_F1[iband], CRIS_BAND_F2[iband], CRIS_D_FREQUENCY[iband],
#         CRIS_F_NYQUIST, CRIS_RESAMPLE_MAXX[iband], CRIS_FILTER_WIDTH[iband],
#         apodization_ori=iasi_apod, apodization_other=cris_apod,
#         plot=plot,
#     )
#     print(wavenumber_iasi2cris[0], wavenumber_iasi2cris[-1])
#     statistics_print(spec_iasi2cris)
#     plot_conversion_picture(plot_data_iasi2cris, conversion_name)


if __name__ == '__main__':
    main()
