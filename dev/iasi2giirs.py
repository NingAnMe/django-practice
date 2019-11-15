#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/11/15 15:37
# @Author  : NingAnMe <ninganme@qq.com>
import numpy as np

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

iband = 0


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


def iasi2hiras(spectrum_ori, frequency_begin_ori, frequency_end_ori, frequency_interval_ori,
               frequency_begin_other, frequency_end_other, frequency_interval_other,
               nyquist_other, opd_other, cos_filter_width,
               apodization_ori=None, plot=False,
               ):
    plot_data = dict()
    spec_ori = spectrum_ori
    fb_ori = frequency_begin_ori
    fe_ori = frequency_end_ori
    fi_ori = frequency_interval_ori

    fb_other = frequency_begin_other
    fe_other = frequency_end_other
    fi_other = frequency_interval_other

    # p0 原始光谱
    if plot:
        plot_data['p0_x'] = np.arange(0, len(spec_ori), dtype=np.float64) * fi_ori + fb_ori
        plot_data['p0_y'] = spec_ori.copy()

    # ########## 做一条和LBL相同分辨率的光谱，频带宽度和other相同
    # totalnum=width(cm-1)/interval  加1.5是因为要做切趾计算
    n_spectrum = int(np.floor(nyquist_other / fi_ori + 1.5))  # 要在IASI采样的点数

    spectrum = np.zeros(n_spectrum, dtype=np.float64)  # other光谱
    frequency = np.arange(0, n_spectrum, dtype=np.float64) * fi_ori  # other频率

    is_b = int(np.floor(fb_ori / fi_ori + 0.5)) - 1  # 放到other光谱的开始位置,index_spec_begin
    is_e = int(np.floor(fe_ori / fi_ori + 0.5))  # 放到other光谱的结束位置 index_spec_end

    spectrum[is_b: is_e] = spec_ori  # spec_ori 放到光谱的对应位置
    # p1 原始光谱格栅到iasi的光谱网格上
    if plot:
        plot_data['p1_x'] = frequency.copy()
        plot_data['p1_y'] = spectrum.copy()

    # ########## 使用 COS 滤波器 过滤两端的值 ，这步和LBL转other不同####################
    n_filter = int(np.floor(cos_filter_width / fi_ori + 1.5))  # 滤波的点数

    if_b1 = is_b - n_filter + 1
    if_e1 = is_b + 1
    if_b2 = is_e - 1
    if_e2 = is_e + n_filter - 1

    frequency_filter = frequency[if_b1:if_e1]  # 600-620cm-1

    cos_filter = 0.5 * (1.0 + np.cos((frequency_filter - frequency_filter[0]) * np.pi / cos_filter_width))

    bfilter = cos_filter * spectrum[is_e - 1]
    ffilter = cos_filter[-1::-1] * spectrum[is_b]

    spectrum[if_b1:if_e1] = ffilter
    spectrum[if_b2:if_e2] = bfilter

    # p2 cos 滤波之后
    if plot:
        plot_data['p2_x'] = frequency.copy()
        plot_data['p2_y'] = spectrum.copy()

    # ########## 原光谱做成一个对称光谱
    # 前半部分同spc 后半部分去掉首末两点的reverse
    n_ifg = 2 * (n_spectrum - 1)  # 干涉图点数 13824000
    spectrum_fft = np.arange(0, n_ifg, dtype=np.float64)
    spectrum_fft[0:n_spectrum] = spectrum  # 前半部分和spectrum相同
    spectrum_fft[n_spectrum:] = spectrum[-2:0:-1]  # 后半部分是spectrum去掉首末两点的倒置

    # p3 对称的光谱图
    if plot:
        plot_data['p3_x'] = np.arange(0, n_ifg, dtype=np.float64) * fi_ori
        plot_data['p3_y'] = spectrum_fft.copy()

    # ########## inverse fft 反傅里叶变换
    ifg_ori = np.fft.fft(spectrum_fft) * fi_ori
    # 傅里叶反变换，转换为双边干涉图，光程差 500cm ，双边 1000cm，间隔 dx
    # 共n_ifg个点，13824000，是全部的干涉图，需要截取

    # p4 傅里叶转换后的干涉图
    if plot:
        plot_data['p4_x'] = np.arange(0, n_ifg, dtype=np.float64) * fi_ori
        plot_data['p4_y'] = ifg_ori.real.copy()

    # ########## compute delta OPD  截断光谱
    n_spectrum = int(np.floor(nyquist_other / fi_ori + 1.5))
    max_x = 1. / (2.0 * fi_ori)  # 500cm
    dx_other = max_x / (n_spectrum - 1)  # cm 7.2337963e-005  723.3796e-7 cm 723.3796nm

    # compute index at which the interferogram is truncated
    idx_trunc = int(opd_other / dx_other)  # 计算最大光程差对应的截断点位置,也就是个数

    # truncate interferogram
    ifg_other = ifg_ori[0: idx_trunc + 1]
    n_ifg_other = idx_trunc + 1
    x_other = np.arange(0, n_ifg_other, dtype=np.float64) * dx_other

    # p5 截取后的iasi干涉图
    if plot:
        plot_data['p5_x'] = np.arange(0, n_ifg_other, dtype=np.float64) * dx_other
        plot_data['p5_y'] = ifg_other.real.copy()

    # ########## 移除原来的 apod，应用新的 apod
    ifg_ap = ifg_other / apodization_ori(x_other)
    # ifg_other_ap = ifg_ap * apodization_other(x_other)

    # p6 apodization
    if plot:
        plot_data['p6_x'] = np.arange(0, n_ifg_other, dtype=np.float64) * dx_other
        plot_data['p6_y'] = ifg_other.real.copy()

    # ########## convert ifg to spectrum，做一个对称的光谱
    n_ifg_fft = 2 * (n_ifg_other - 1)  # 干涉图点数拓展
    ifg_fft = np.arange(0, n_ifg_fft, dtype=ifg_ap.dtype)
    ifg_fft[0:n_ifg_other] = ifg_ap
    ifg_fft[n_ifg_other:] = ifg_ap[-2:0:-1]
    # 干涉图拓展 光程差2*2cm 为 4cm  点数n_ifg_fft 55296 间隔
    # 去掉最大值,和最末的值

    # p7 拓展干涉图的示意图
    if plot:
        plot_data['p7_x'] = np.arange(0, n_ifg_fft, dtype=np.float64) * dx_other
        plot_data['p7_y'] = ifg_fft.real.copy()

    # ########## FFT 正变换
    spectrum_other_comp = np.fft.ifft(ifg_fft) * dx_other * n_ifg_fft  # FFT 正变换
    spectrum_other_comp = spectrum_other_comp.real  # 仅使用实数

    # ########## take out other portion of the spectrum
    # 取得iasi的光谱  在干涉图中按2cm最大光程差取完之后，做FFT反变换得到的光谱的分辨率就是0.25cm-1
    nt1 = int(np.floor(fb_other / fi_other + 0.5))
    nt2 = int(np.floor(fe_other / fi_other + 0.5))
    n_spectrum_hiras = nt2 - nt1 + 1
    spectrum_other = np.zeros(n_spectrum_hiras)

    # 20190416 将干涉图维度的平滑计算改到光谱维度
    a = 0.23
    coef = np.array([a, (1 - 2 * a), a])
    for i in range(nt1, nt2 + 1):
        spectrum_other[i - nt1] = sum(spectrum_other_comp[i - 1:i + 2] * coef)

    wavenumber_other = np.arange(0, n_spectrum_hiras, dtype=np.float64)
    wavenumber_other = fb_other + wavenumber_other * frequency_interval_other

    # p08 IASI 光谱
    if plot:
        plot_data['p8_x'] = wavenumber_other
        plot_data['p8_y'] = spectrum_other

    return spectrum_other, wavenumber_other, plot_data


def iasi2giirs(radiance):
    spec_iasi2giirs, wavenumber_iasi2giirs, plot_data_iasi2giirs = iasi2hiras(
        radiance, IASI_BAND_F1[iband], IASI_BAND_F2[iband], IASI_D_FREQUENCY[iband],
        GIIRS_BAND_F1[iband], GIIRS_BAND_F2[iband], GIIRS_D_FREQUENCY[iband],
        GIIRS_F_NYQUIST, GIIRS_RESAMPLE_MAXX[iband], GIIRS_FILTER_WIDTH[iband],
        apodization_ori=iasi_apod, )
    return spec_iasi2giirs, wavenumber_iasi2giirs
