#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2018/12/20
"""
import os

import numpy as np
from netCDF4 import Dataset

from conversion import lbl2other, ori2other

IBAND = [0]  # band 1、 2 or 3，光谱带
LBL_DIR = 'D:/nsmc/LBL/data/'

# #########  仪器参数 #############
FILTER_WIDTH = 20.0  # cm-1  # COS过滤器过滤的宽度

IASI_F_NYQUIST = 6912.0  # 频带宽度  cm-1
IASI_RESAMPLE_MAXX = [2.0, 2.0, 2.0]  # cm OPD
IASI_D_FREQUENCY = [0.25, 0.25, 0.25]  # v cm-1  光谱分辨率
IASI_BAND_F1 = [645.00, 1210.00, 2000.0]  # 光谱带开始
IASI_BAND_F2 = [1209.75, 1999.75, 2760.0]  # 光谱带结束
IASI_FILTER_WIDTH = [20.0, 20.0, 20.0]

CRIS_F_NYQUIST = 5875.0
CRIS_RESAMPLE_MAXX = [0.8, 0.8, 0.8]
CRIS_D_FREQUENCY = [0.625, 0.625, 0.625]
CRIS_BAND_F1 = [650.00, 1210.00, 2155.0]
CRIS_BAND_F2 = [1135.00, 1750.00, 2550.0]
CRIS_FILTER_WIDTH = [20.0, 20.0, 20.0]

HIRAS_F_NYQUIST = 5866.0
HIRAS_RESAMPLE_MAXX = [0.8, 0.8, 0.8]
HIRAS_D_FREQUENCY = [0.625, 1.25, 2.5]
HIRAS_BAND_F1 = [650.00, 1210.00, 2155.0]
HIRAS_BAND_F2 = [1095.00, 1750.00, 2550.0]
HIRAS_FILTER_WIDTH = [20.0, 24.0, 20.0]


def main():
    for iband in IBAND:
        lbl_file = os.path.join(LBL_DIR,
                                'iasiB{:1d}_metop-a_lbl_radSpectrum.nc'.format(iband+1))
        data = read_nc(lbl_file)
        # print data

        rad_lbl = data['SPECTRUM']
        bf_lbl = data['BEGIN_FREQUENCY']
        ef_lbl = data['END_FREQUENCY']
        df_lbl = data['FREQUENCY_INTERVAL']

        rad_lbl = rad_lbl * 1.0e7

        # ################### Compute IASI spectrum #############
        conversion_name = 'LBL2IASI_{}'.format(iband)
        spec_lbl2iasi = lbl2other(
            rad_lbl, bf_lbl, ef_lbl, df_lbl,
            IASI_BAND_F1[iband], IASI_BAND_F2[iband], IASI_D_FREQUENCY[iband],
            IASI_F_NYQUIST, IASI_RESAMPLE_MAXX[iband], IASI_FILTER_WIDTH[iband],
            iasi_apod, conversion_name=conversion_name,
        )

        statistics_print(spec_lbl2iasi)
        #
        # ################### Compute CrIS spectrum #############
        conversion_name = 'LBL2CRIS_{}'.format(iband)
        spec_lbl2cris = lbl2other(
            rad_lbl, bf_lbl, ef_lbl, df_lbl,
            CRIS_BAND_F1[iband], CRIS_BAND_F2[iband], CRIS_D_FREQUENCY[iband],
            CRIS_F_NYQUIST, CRIS_RESAMPLE_MAXX[iband], CRIS_FILTER_WIDTH[iband],
            cris_apod, conversion_name=conversion_name
        )

        statistics_print(spec_lbl2cris)

        # ################### Compute IASI to CrIS spectrum #############
        conversion_name = 'IASI2CRIS_{}'.format(iband)
        spec_iasi2cris = ori2other(
            spec_lbl2iasi, IASI_BAND_F1[iband], IASI_BAND_F2[iband], IASI_D_FREQUENCY[iband],
            CRIS_BAND_F1[iband], CRIS_BAND_F2[iband], CRIS_D_FREQUENCY[iband],
            CRIS_F_NYQUIST, CRIS_RESAMPLE_MAXX[iband], CRIS_FILTER_WIDTH[iband],
            apodization_ori=iasi_apod, apodization_other=cris_apod,
            conversion_name=conversion_name,
        )

        statistics_print(spec_iasi2cris)


def main_origin():
    for iband in IBAND:
        lbl_file = os.path.join(LBL_DIR,
                                'iasiB{:1d}_metop-a_lbl_radSpectrum.nc'.format(iband+1))
        data = read_nc(lbl_file)
        # print data

        rad_lbl = data['SPECTRUM']
        bf = data['BEGIN_FREQUENCY']
        ef = data['END_FREQUENCY']
        df = data['FREQUENCY_INTERVAL']

        rad_lbl = rad_lbl * 1.0e7

        # p0 原数据
        # n_lbl = np.size(rad_lbl)  # rad 总数量
        # wn_lbl = bf + np.arange(0, n_lbl, dtype=np.float64) * df  # lbl 波数
        # plt.plot(wn_lbl, rad_lbl)
        # plt.show()

        # ################### Compute IASI spectrum #############
        # ################### Compute IASI spectrum #############
        # ################### Compute IASI spectrum #############

        # ########## 做一条和LBL相同分辨率的光谱，频带宽度和IASI相同
        # totalnum=width(cm-1)/interval  加1.5是因为要做切趾计算
        n_spectrum = int(np.floor(IASI_F_NYQUIST / df + 1.5))  # 要在LBL采样的点数 6912001

        spectrum = np.zeros(n_spectrum, dtype=np.float64)  # IASI光谱
        frequency = np.arange(0, n_spectrum, dtype=np.float64) * df  # IASI频率

        is1 = int(np.floor(bf / df + 0.5))  # rad_lbl 放到IASI光谱的开始位置
        is2 = int(np.floor(ef / df + 0.5)) + 1  # rad_lbl 放到IASI光谱的结束位置

        spectrum[is1: is2] = rad_lbl  # rad_lbl 放到光谱的对应位置
        # p1 原始光谱格栅到iasi的光谱网格上
        # plt.plot(frequency, spectrum)
        # plt.show()

        # ########## 使用 COS 滤波器 过滤两端的值  ####################
        n_filter = int(np.floor(FILTER_WIDTH / df + 1.5))  # 滤波的点数 20001

        if1 = is1
        if2 = is1 + n_filter
        ib1 = is2 - n_filter
        ib2 = is2

        frequency_filter = frequency[if1:if2]  # 600-620cm-1

        bfilter = 0.5 * (
                1.0 + np.cos((frequency_filter - frequency_filter[0]) * np.pi / FILTER_WIDTH))
        ffilter = bfilter[::-1]

        spectrum[if1:if2] = spectrum[if1:if2] * ffilter
        spectrum[ib1:ib2] = spectrum[ib1:ib2] * bfilter

        # p11 cos 滤波之后
        # plt.plot(frequency, spectrum)
        # plt.xlim(frequency[if1-100], frequency[ib2+100])
        # plt.show()

        # ########## 原光谱做成一个对称光谱
        # 前半部分同spc 后半部分去掉首末两点的reverse
        n_ifg = 2 * (n_spectrum - 1)  # 干涉图点数 13824000
        spectrum_fft = np.arange(0, n_ifg, dtype=np.float64)
        spectrum_fft[0:n_spectrum] = spectrum  # 前半部分和spectrum相同
        spectrum_fft[n_spectrum:] = spectrum[-2:0:-1]  # 后半部分是spectrum去掉首末两点的倒置

        # p2 对称的光谱图
        # frequency_fft = np.arange(0, n_ifg, dtype=np.float64) * df
        # plt.plot(frequency_fft, spectrum_fft)
        # plt.show()

        # ########## inverse fft 反傅里叶变换
        ifg = np.fft.fft(spectrum_fft) * df
        # 傅里叶反变换，转换为双边干涉图，光程差 500cm ，双边 1000cm，间隔 dx
        # 共n_ifg个点，13824000，是全部的干涉图，需要截取

        # p3 傅里叶转换后的干涉图
        # n_spectrum_ifg = np.arange(0, n_ifg, dtype=np.float64) * dx
        # plt.plot(n_spectrum_ifg, ifg)
        # plt.show()

        # ########## compute delta OPD  截断光谱
        max_x = 1. / (2.0 * df)  # 500cm
        dx = max_x / (n_spectrum - 1)  # cm 7.2337963e-005  723.3796e-7 cm 723.3796nm

        # compute index at which the interferogram is truncated
        idx_trunc = int(IASI_RESAMPLE_MAXX[iband] / dx)
        # 计算iasi最大光程差对应的截断点位置,也就是个数
        # iasi 对应的 光程差是 2cm 间隔 dx 不变 得到截取的点数是 27649 个

        # truncate interferogram
        ifg_iasi = ifg[0: idx_trunc + 1]
        n_ifg_iasi = idx_trunc + 1

        # p4 截取后的iasi干涉图
        # plt.plot(x_iasi, ifg_iasi)
        # plt.show()

        # apply iasi apodization
        x_iasi = np.arange(0, n_ifg_iasi, dtype=np.float64) * dx
        ifg_iasi_ap = ifg_iasi * iasi_apod(x_iasi)

        # p5 高斯apod
        # plt.plot(x_iasi, ifg_iasi_ap)  # 对iasi干涉图进行高斯apod乘法
        # plt.show()

        # convert ifg to spectrum，做一个对称的光谱
        n_ifg_fft = 2 * (n_ifg_iasi - 1)  # 干涉图点数拓展为  55296
        ifg_fft = np.arange(0, n_ifg_fft, dtype=ifg_iasi_ap.dtype)
        ifg_fft[0:n_ifg_iasi] = ifg_iasi_ap
        ifg_fft[n_ifg_iasi:] = ifg_iasi_ap[-2:0:-1]
        # 干涉图拓展 光程差2*2cm 为 4cm  点数n_ifg_fft 55296 间隔
        # 去掉最大值,和最末的值

        # p6 拓展干涉图的示意图
        # x_iasi_fft = np.arange(0, n_ifg_fft, dtype=np.float64) * dx
        # plt.plot(x_iasi_fft, ifg_fft)
        # plt.xlim(-1, 5)
        # plt.show()

        # ########## FFT 正变换
        spectrum_iasi_comp = np.fft.ifft(ifg_fft) * dx * n_ifg_fft  # FFT 正变换
        spectrum_iasi_comp = spectrum_iasi_comp.real  # 仅使用实数

        # take out iasi portion of the spectrum
        # 取得iasi的光谱  在干涉图中按2cm最大光程差取完之后
        # 做FFT反变换得到的光谱的分辨率就是0.25cm-1
        f1_iasi = IASI_BAND_F1[iband]
        f2_iasi = IASI_BAND_F2[iband]
        nt1 = int(np.floor(f1_iasi / IASI_D_FREQUENCY[iband] + 0.5))
        nt2 = int(np.floor(f2_iasi / IASI_D_FREQUENCY[iband] + 0.5))
        spectrum_iasi = spectrum_iasi_comp[nt1: nt2 + 1]

        statistics_print(spectrum_iasi)

        s_iasi_new = lbl2other(rad_lbl, bf, ef, 0.001,
                               IASI_BAND_F1[0], IASI_BAND_F2[0], IASI_D_FREQUENCY[0],
                               IASI_F_NYQUIST, IASI_RESAMPLE_MAXX[0], FILTER_WIDTH, iasi_apod)

        statistics_print(s_iasi_new)

        # p07 IASI 光谱
        # n_spectrum_iasi = nt2 - nt1 + 1
        # wn_iasi = np.arange(0, n_spectrum_iasi, dtype=np.float64)
        # wn_iasi = f1_iasi + wn_iasi * IASI_D_FREQUENCY[iband]
        # plt.plot(wn_iasi, spectrum_iasi)
        # plt.show()
        # ################### Compute CrIS spectrum ############
        # ################### Compute CrIS spectrum ############
        # ################### Compute CrIS spectrum ############

        # ########## 做一条和LBL相同分辨率的光谱，频带宽度和CrIS相同
        n_spectrum = int(np.floor(CRIS_F_NYQUIST / df + 1.5))  # 根据频带宽度计算在LBL采样的波长点数

        spectrum = np.zeros(n_spectrum, dtype=np.float64)  # CrIS光谱：
        frequency = np.arange(0, n_spectrum, dtype=np.float64) * df  # CrIS波长：间隔为 df

        is1 = int(np.floor(bf / df + 0.5))  # rad_lbl 放到CrIS光谱的开始位置
        is2 = int(np.floor(ef / df + 0.5)) + 1  # rad_lbl 放到CrIS光谱的结束位置

        spectrum[is1: is2] = rad_lbl  # rad_lbl 放到光谱的对应位置
        # p1 原始光谱格栅到iasi的光谱网格上
        # plt.plot(frequency, spectrum)
        # plt.show()

        # ########## 使用 COS 滤波器 过滤两端的值
        n_filter = int(np.floor(FILTER_WIDTH / df + 1.5))  # 滤波的点数：左右两边要使用cos函数滤波

        if1 = is1
        if2 = is1 + n_filter
        ib1 = is2 - n_filter
        ib2 = is2

        frequency_filter = frequency[if1:if2]  # 600-620cm-1

        bfilter = 0.5 * (
                    1.0 + np.cos((frequency_filter - frequency_filter[0]) * np.pi / FILTER_WIDTH))
        ffilter = bfilter[::-1]

        spectrum[if1:if2] = spectrum[if1:if2] * ffilter
        spectrum[ib1:ib2] = spectrum[ib1:ib2] * bfilter

        # p11 cos 滤波之后
        # plt.plot(frequency, spectrum)
        # plt.xlim(frequency[if1-100], frequency[ib2+100])
        # plt.show()

        # ########## 原光谱做成一个对称光谱
        # 前半部分同spc 后半部分去掉首末两点的reverse
        n_ifg = 2 * (n_spectrum - 1)
        spectrum_fft = np.arange(0, n_ifg, dtype=np.float64)
        spectrum_fft[0:n_spectrum] = spectrum  # 前半部分和spectrum相同
        spectrum_fft[n_spectrum:] = spectrum[-2:0:-1]  # 后半部分是spectrum去掉首末两点的倒置

        # p2 对称的光谱图
        # frequency_fft = np.arange(0, n_ifg, dtype=np.float64) * df
        # plt.plot(frequency_fft, spectrum_fft)
        # plt.show()

        # ########## inverse fft 反傅里叶变换
        ifg = np.fft.fft(spectrum_fft) * df
        # 傅里叶反变换，转换为双边干涉图，光程差 500cm ，双边 1000cm，间隔 dx
        # 共n_ifg个点，13824000，是全部的干涉图，需要截取

        # p3 傅里叶转换后的干涉图
        # n_spectrum_ifg = np.arange(0, n_ifg, dtype=np.float64) * dx
        # plt.plot(n_spectrum_ifg, ifg)
        # plt.show()

        # ########## compute delta OPD 截断光谱
        max_x = 1. / (2.0 * df)  # 500cm LBL 的最大光程差
        dx = max_x / (n_spectrum - 1)  # cm 7.2337963e-005  723.3796e-7 cm 723.3796nm

        # compute index at which the interferogram is truncated
        idx_trunc = int(CRIS_RESAMPLE_MAXX[iband] / dx)
        # 计算CrIS最大光程差对应的截断点位置,也就是个数
        # cris 对应的 光程差是 0.8cm 间隔 dx 不变 得到截取的点数是

        # truncate interferogram
        ifg_cris = ifg[0: idx_trunc + 1]
        n_ifg_cris = idx_trunc + 1

        # p4 截取后的cris干涉图
        # x_cris = np.arange(0, n_ifg_cris, dtype=np.float64) * dx
        # plt.plot(x_cris, ifg_cris)
        # plt.show()

        # convert ifg to spectrum，做一个对称的光谱
        n_ifg_fft = 2 * (n_ifg_cris - 1)  # 干涉图点数拓展
        ifg_fft = np.arange(0, n_ifg_fft, dtype=ifg_cris.dtype)
        ifg_fft[0:n_ifg_cris] = ifg_cris
        ifg_fft[n_ifg_cris:] = ifg_cris[-2:0:-1]
        # 干涉图拓展 光程差2*2cm 为 4cm  点数n_ifg_fft 55296 间隔
        # 去掉最大值,和最末的值

        # p6 拓展干涉图的示意图
        # x_iasi_fft = np.arange(0, n_ifg_fft, dtype=np.float64) * dx
        # plt.plot(x_iasi_fft, ifg_fft)
        # plt.xlim(-1, 5)
        # plt.show()

        # ########## FFT 正变换
        spectrum_cris_comp = np.fft.ifft(ifg_fft) * dx * n_ifg_fft  # FFT 正变换
        spectrum_cris_comp = spectrum_cris_comp.real  # 仅使用实数

        # take out iasi portion of the spectrum
        # 取得iasi的光谱  在干涉图中按2cm最大光程差取完之后
        # 做FFT反变换得到的光谱的分辨率就是0.25cm-1
        f1_cris = CRIS_BAND_F1[iband]
        f2_cris = CRIS_BAND_F2[iband]
        nt1 = int(np.floor(f1_cris / CRIS_D_FREQUENCY[iband] + 0.5))
        nt2 = int(np.floor(f2_cris / CRIS_D_FREQUENCY[iband] + 0.5))
        n_spectrum_cris = nt2 - nt1 + 1
        spectrum_cris = np.zeros(n_spectrum_cris, dtype=np.float64)

        # apply cris apodization
        a = 0.23
        coef = [a, (1.0 - 2.0 * a), a]
        for i in xrange(nt1, nt2 + 1):
            spectrum_cris[i - nt1] = sum(spectrum_cris_comp[i - 1: i + 2] * coef)

        # p5 hamming apod
        # plt.plot(x_iasi, ifg_iasi_ap)  # 对iasi干涉图进行高斯apod乘法
        # plt.show()

        statistics_print(spectrum_cris)

        s_cris_new = lbl2other(rad_lbl, bf, ef, 0.001,
                               CRIS_BAND_F1[0], CRIS_BAND_F2[0], CRIS_D_FREQUENCY[0],
                               CRIS_F_NYQUIST, CRIS_RESAMPLE_MAXX[0], FILTER_WIDTH, cris_apod)

        statistics_print(s_cris_new)

        # p07 CRIS 光谱
        # wn_cris = np.arange(0, n_spectrum_cris, dtype=np.float64)
        # wn_cris = f1_cris + wn_cris * CRIS_D_FREQUENCY[iband]
        # plt.plot(wn_cris, spectrum_iasi)
        # plt.show()

        # ################### IASI to CrIS ############
        # ################### IASI to CrIS ############
        # ################### IASI to CrIS ############
        # ########## 做一条和IASI相同分辨率的光谱，频带宽度和CrIS相同
        df = IASI_D_FREQUENCY[iband]
        n_spectrum = int(np.floor(CRIS_F_NYQUIST / df + 1.5))  # 根据频带宽度计算在LBL采样的波长点数

        spectrum = np.zeros(n_spectrum, dtype=np.float64)  # CrIS光谱：
        frequency = np.arange(0, n_spectrum, dtype=np.float64) * df  # CrIS波长：间隔为 df

        is1 = int(np.floor(f1_iasi / df + 0.5))  # rad_lbl 放到CrIS光谱的开始位置
        is2 = int(np.floor(f2_iasi / df + 0.5)) + 1  # rad_lbl 放到CrIS光谱的结束位置

        spectrum[is1: is2] = spectrum_iasi  # rad_lbl 放到光谱的对应位置
        # p1 原始光谱格栅到iasi的光谱网格上
        # plt.plot(frequency, spectrum)
        # plt.show()

        # ########## 使用 COS 滤波器 过滤两端的值  ####################
        n_filter = int(np.floor(FILTER_WIDTH / df + 1.5))  # 滤波的点数 20001

        if1 = is1 - n_filter + 1
        if2 = is1 + 1
        ib1 = is2 - 1
        ib2 = is2 + n_filter - 1

        frequency_filter = frequency[if1:if2]  # 600-620cm-1

        cos_filter = 0.5 * (
                1.0 + np.cos((frequency_filter - frequency_filter[0]) * np.pi / FILTER_WIDTH))
        ffilter = cos_filter[n_filter::-1] * spectrum[is1]
        bfilter = cos_filter * spectrum[is2]

        spectrum[if1:if2] = ffilter
        spectrum[ib1:ib2] = bfilter

        # ########## 原光谱做成一个对称光谱
        # 前半部分同spc 后半部分去掉首末两点的reverse
        n_ifg = 2 * (n_spectrum - 1)
        spectrum_fft = np.arange(0, n_ifg, dtype=np.float64)
        spectrum_fft[0:n_spectrum] = spectrum  # 前半部分和spectrum相同
        spectrum_fft[n_spectrum:] = spectrum[-2:0:-1]  # 后半部分是spectrum去掉首末两点的倒置

        # p2 对称的光谱图
        # frequency_fft = np.arange(0, n_ifg, dtype=np.float64) * df
        # plt.plot(frequency_fft, spectrum_fft)
        # plt.show()

        # ########## inverse fft 反傅里叶变换
        ifg = np.fft.fft(spectrum_fft) * df
        # 傅里叶反变换，转换为双边干涉图，光程差 500cm ，双边 1000cm，间隔 dx
        # 共n_ifg个点，13824000，是全部的干涉图，需要截取

        # p3 傅里叶转换后的干涉图
        # n_spectrum_ifg = np.arange(0, n_ifg, dtype=np.float64) * dx
        # plt.plot(n_spectrum_ifg, ifg)
        # plt.show()

        # ########## compute delta OPD 截断光谱
        max_x = 1. / (2.0 * df)  # 500cm LBL 的最大光程差
        dx = max_x / (n_spectrum - 1)  # cm 7.2337963e-005  723.3796e-7 cm 723.3796nm

        # compute index at which the interferogram is truncated
        idx_trunc = int(CRIS_RESAMPLE_MAXX[iband] / dx)
        # 计算CrIS最大光程差对应的截断点位置,也就是个数
        # cris 对应的 光程差是 0.8cm 间隔 dx 不变 得到截取的点数是

        # truncate interferogram
        ifg_cris = ifg[0: idx_trunc + 1]
        n_ifg_cris = idx_trunc + 1
        x_cris = np.arange(0, n_ifg_cris, dtype=np.float64) * dx

        # p4 截取后的cris干涉图
        # x_cris = np.arange(0, n_ifg_cris, dtype=np.float64) * dx
        # plt.plot(x_cris, ifg_cris)
        # plt.show()

        # 移除 IASI apod
        ifg_cris = ifg_cris / iasi_apod(x_cris)

        # convert ifg to spectrum，做一个对称的光谱
        n_ifg_fft = 2 * (n_ifg_cris - 1)  # 干涉图点数拓展
        ifg_fft = np.arange(0, n_ifg_fft, dtype=ifg_cris.dtype)
        ifg_fft[0:n_ifg_cris] = ifg_cris
        ifg_fft[n_ifg_cris:] = ifg_cris[-2:0:-1]
        # 干涉图拓展 光程差2*2cm 为 4cm  点数n_ifg_fft 55296 间隔
        # 去掉最大值,和最末的值

        # p6 拓展干涉图的示意图
        # x_iasi_fft = np.arange(0, n_ifg_fft, dtype=np.float64) * dx
        # plt.plot(x_iasi_fft, ifg_fft)
        # plt.xlim(-1, 5)
        # plt.show()

        # ########## FFT 正变换
        spectrum_cris_comp = np.fft.ifft(ifg_fft) * dx * n_ifg_fft  # FFT 正变换
        spectrum_cris_comp = spectrum_cris_comp.real  # 仅使用实数

        # take out iasi portion of the spectrum
        # 取得iasi的光谱  在干涉图中按2cm最大光程差取完之后
        # 做FFT反变换得到的光谱的分辨率就是0.25cm-1
        f1_cris = CRIS_BAND_F1[iband]
        f2_cris = CRIS_BAND_F2[iband]
        nt1 = int(np.floor(f1_cris / CRIS_D_FREQUENCY[iband] + 0.5))
        nt2 = int(np.floor(f2_cris / CRIS_D_FREQUENCY[iband] + 0.5))
        n_spectrum_cris = nt2 - nt1 + 1
        spectrum_cris = np.zeros(n_spectrum_cris, dtype=np.float64)

        # apply cris apodization
        a = 0.23
        coef = [a, (1.0 - 2.0 * a), a]
        for i in xrange(nt1, nt2 + 1):
            spectrum_cris[i - nt1] = sum(spectrum_cris_comp[i - 1: i + 2] * coef)

        statistics_print(spectrum_cris)

        iasi2cris_new = ori2other(spectrum_iasi, 645.00, 1209.75, 0.25,
                                  650.00, 1135.00, 0.625, 5875.0, 0.8, 20.0,
                                  apodization_ori=iasi_apod, apodization_other=cris_apod)

        statistics_print(iasi2cris_new)

        # p07 CRIS 光谱
        # wn_cris = np.arange(0, n_spectrum_cris, dtype=np.float64)
        # wn_cris = f1_cris + wn_cris * CRIS_D_FREQUENCY[iband]
        # plt.plot(wn_cris, spectrum_iasi)
        # plt.show()


def statistics_print(data):
    print data[0], data[-1], np.size(data), np.min(data), np.max(data), np.mean(data)


def iasi_apod(x, fwhm=0.5):
    """
    Gaussian function
    :param x: iasi 光谱
    :param fwhm: Gaussian 参数
    :return:
    """
    gft_fwhm = fwhm  # Gaussian function FWHM (cm^-1)
    gft_hwhm = gft_fwhm / 2.0
    ln2 = 0.693147180559945309417232

    sigma = ln2 / (np.pi * gft_hwhm)
    result = np.exp(-ln2 * (x / sigma) ** 2)
    return result


def cris_apod(x, opd=0.8):
    """
    Hamming Apod
    :param x:  cris 光谱
    :param opd: 光程差
    :return: cris_apod
    """
    a0 = 0.54
    a1 = 0.46
    opd = opd
    result = a0 + a1 * np.cos(np.pi * x / opd)
    return result


def read_nc(in_file):
    """
    读取 LBL 文件
    :param in_file: 文件绝对路径
    :return: 文件内容
    """
    f = Dataset(in_file)

    spectrum = f.variables['spectrum'][:]
    begin_wn = f.variables['begin_frequency'][:]
    end_wn = f.variables['end_frequency'][:]
    wn_interval = f.variables['frequency_interval'][:]

    data = {
        'SPECTRUM': spectrum,  # 光谱
        'BEGIN_FREQUENCY': begin_wn,  # 开始频率
        'END_FREQUENCY': end_wn,  # 结束频率
        'FREQUENCY_INTERVAL': wn_interval,  # 频率间隔
    }
    return data


def rad2tbb(rad, center_wave):
    """
    辐射率转亮温
    :param rad: 辐射率
    :param center_wave: 中心波数
    :return: 亮温
    """
    c1 = 1.1910427e-5
    c2 = 1.4387752

    tbb = (c2 * center_wave) / np.log(1 + ((c1 * center_wave ** 3) / rad))
    return tbb


if __name__ == '__main__':
    main()
    # main_origin()
