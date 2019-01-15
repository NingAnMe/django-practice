#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2018/12/20
"""
from spectrum_conversion import *
from plot import *
from util import *

IBAND = [0, 1, 2]  # band 1、 2 or 3，光谱带
LBL_DIR = r'D:\nsmc\LBL\data'

# #########  仪器参数 #############
FILTER_WIDTH = 20.0  # cm-1  # COS过滤器过滤的宽度

IASI_F_NYQUIST = 6912.0  # 频带宽度  cm-1
IASI_RESAMPLE_MAXX = [2.0, ]  # cm OPD
IASI_D_FREQUENCY = [0.25, ]  # v cm-1  光谱分辨率
IASI_BAND_F1 = [645.25, ]  # 光谱带开始
IASI_BAND_F2 = [2760.25, ]  # 光谱带结束
IASI_FILTER_WIDTH = [20.0, ]

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
    lbl_file = os.path.join(LBL_DIR, 'lblrtm_res_001.h5')
    data = read_hdf5(lbl_file)
    # print data

    rad_lbl = data['SPECTRUM']
    rad_lbl = rad_lbl.reshape(-1)
    bf_lbl = data['BEGIN_FREQUENCY']
    ef_lbl = data['END_FREQUENCY']
    df_lbl = data['FREQUENCY_INTERVAL']

    rad_lbl = rad_lbl * 1.0e7

    iband = 0

    plot = True

    # ################### Compute IASI spectrum #############
    if not os.path.isdir('pic_lbl'):
        os.mkdir('pic_lbl')

    conversion_name = 'pic_lbl/LBL2IASI_all'
    spec_lbl2iasi, wavenumber_lbl2iasi, plot_data_lbl2iasi = lbl2other(
        rad_lbl, bf_lbl, ef_lbl, df_lbl,
        IASI_BAND_F1[iband], IASI_BAND_F2[iband], IASI_D_FREQUENCY[iband],
        IASI_F_NYQUIST, IASI_RESAMPLE_MAXX[iband], IASI_FILTER_WIDTH[iband],
        iasi_apod, plot=plot,
    )
    print(wavenumber_lbl2iasi[0], wavenumber_lbl2iasi[-1])
    statistics_print(spec_lbl2iasi)
    plot_conversion_picture(plot_data_lbl2iasi, conversion_name)

    # ################### Compute CrIS spectrum #############
    conversion_name = 'pic_lbl/LBL2CRIS_all'
    spec_lbl2cris, wavenumber_lbl2cris, plot_data_lbl2cris = lbl2other(
        rad_lbl, bf_lbl, ef_lbl, df_lbl,
        CRIS_BAND_F1[iband], CRIS_BAND_F2[iband], CRIS_D_FREQUENCY[iband],
        CRIS_F_NYQUIST, CRIS_RESAMPLE_MAXX[iband], CRIS_FILTER_WIDTH[iband],
        cris_apod, plot=plot,
    )
    print(wavenumber_lbl2cris[0], wavenumber_lbl2cris[-1])
    statistics_print(spec_lbl2cris)
    plot_conversion_picture(plot_data_lbl2cris, conversion_name)

    # ################### Compute IASI to CrIS spectrum #############
    conversion_name = 'pic_lbl/IASI2CRIS_all'
    spec_iasi2cris, wavenumber_iasi2cris, plot_data_iasi2cris = ori2other(
        spec_lbl2iasi, IASI_BAND_F1[iband], IASI_BAND_F2[iband], IASI_D_FREQUENCY[iband],
        CRIS_BAND_F1[iband], CRIS_BAND_F2[iband], CRIS_D_FREQUENCY[iband],
        CRIS_F_NYQUIST, CRIS_RESAMPLE_MAXX[iband], CRIS_FILTER_WIDTH[iband],
        apodization_ori=iasi_apod, apodization_other=cris_apod,
        plot=plot,
    )
    print(wavenumber_iasi2cris[0], wavenumber_iasi2cris[-1])
    statistics_print(spec_iasi2cris)
    plot_conversion_picture(plot_data_iasi2cris, conversion_name)

    spec_bias = spec_iasi2cris - spec_lbl2cris
    plot_kwargs = {'s': 0.1}
    format_kwargs = {
        'x_axis_min': 0,
        'x_axis_max': 3000,
        'x_interval': 500,
        'x_label': 'Wavenumber($cm^{-1}$)',

        'y_axis_min': -3,
        'y_axis_max': 3,
        'y_interval': 1,
        'y_label': 'Radiance Bias($mw/m^2/sr/cm^{-1}$)'

    }
    plot_scatter(wavenumber_iasi2cris, spec_bias, 'pic_lbl/IASI2CRIS_all_bias.png',
                 format_kwargs=format_kwargs, plot_kwargs=plot_kwargs)


if __name__ == '__main__':
    main()
