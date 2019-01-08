#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/1/3
@Author  : AnNing
"""
import matplotlib.pyplot as plt
import numpy as np

LINEWIDTH = 0.5


def lbl2other(spectrum_lbl, frequency_begin_lbl, frequency_end_lbl, frequency_interval_lbl,
              frequency_begin_other, frequency_end_other, frequency_interval_other,
              nyquist_other, opd_other, cos_filter_width,
              apodization=None, conversion_name=None):
    spec_lbl = spectrum_lbl
    fb_lbl = frequency_begin_lbl
    fe_lbl = frequency_end_lbl
    fi_lbl = frequency_interval_lbl

    fb_other = frequency_begin_other
    fe_other = frequency_end_other
    fi_other = frequency_interval_other

    # p0 原始光谱格栅到iasi的光谱网格上
    if conversion_name is not None:
        frequency_ = np.arange(0, len(spec_lbl), dtype=np.float64) * fi_lbl
        plot_line(conversion_name + '_' + '00.png', frequency_,
                  spec_lbl, lw=LINEWIDTH)

    # ########## 做一条和LBL相同分辨率的光谱，频带宽度和other相同
    # totalnum=width(cm-1)/interval  加1.5是因为要做切趾计算
    n_spectrum = int(np.floor(nyquist_other / fi_lbl + 1.5))  # 要在LBL采样的点数 6912001

    spectrum = np.zeros(n_spectrum, dtype=np.float64)  # other光谱
    frequency = np.arange(0, n_spectrum, dtype=np.float64) * fi_lbl  # other频率

    is_b = int(np.floor(fb_lbl / fi_lbl + 0.5))  # 放到other光谱的开始位置,index_spec_begin
    is_e = int(np.floor(fe_lbl / fi_lbl + 0.5)) + 1  # 放到other光谱的结束位置 index_spec_end

    spectrum[is_b: is_e] = spec_lbl  # spec_lbl 放到光谱的对应位置
    # p1 原始光谱格栅到iasi的光谱网格上
    if conversion_name is not None:
        plot_line(conversion_name + '_' + '01.png', frequency[is_b - 100: is_e + 100],
                  spectrum[is_b-100: is_e + 100], lw=LINEWIDTH)

    # ########## 使用 COS 滤波器 过滤两端的值  ####################
    n_filter = int(np.floor(cos_filter_width / fi_lbl + 1.5))  # 滤波的点数

    if_b1 = is_b
    if_e1 = is_b + n_filter
    if_b2 = is_e - n_filter
    if_e2 = is_e

    frequency_filter = frequency[if_b1:if_e1]  # 600-620cm-1

    bfilter = 0.5 * (
            1.0 + np.cos((frequency_filter - frequency_filter[0]) * np.pi / cos_filter_width))
    ffilter = bfilter[::-1]

    spectrum[if_b1:if_e1] = spectrum[if_b1:if_e1] * ffilter
    spectrum[if_b2:if_e2] = spectrum[if_b2:if_e2] * bfilter

    # p2 cos 滤波之后
    if conversion_name is not None:
        plot_line(conversion_name + '_' + '02.png', frequency[is_b - 100: is_e + 100],
                  spectrum[is_b-100: is_e + 100], lw=LINEWIDTH)

    # ########## 原光谱做成一个对称光谱
    # 前半部分同spc 后半部分去掉首末两点的reverse
    n_ifg = 2 * (n_spectrum - 1)  # 干涉图点数 13824000
    spectrum_fft = np.arange(0, n_ifg, dtype=np.float64)
    spectrum_fft[0:n_spectrum] = spectrum  # 前半部分和spectrum相同
    spectrum_fft[n_spectrum:] = spectrum[-2:0:-1]  # 后半部分是spectrum去掉首末两点的倒置

    # p3 对称的光谱图
    if conversion_name is not None:
        frequency_fft = np.arange(0, n_ifg, dtype=np.float64) * fi_lbl
        plot_line(conversion_name + '_' + '03.png', frequency_fft, spectrum_fft, lw=LINEWIDTH)

    # ########## inverse fft 反傅里叶变换
    ifg_lbl = np.fft.fft(spectrum_fft) * fi_lbl
    # 傅里叶反变换，转换为双边干涉图，光程差 500cm ，双边 1000cm，间隔 dx
    # 共n_ifg个点，13824000，是全部的干涉图，需要截取

    # p4 傅里叶转换后的干涉图
    if conversion_name is not None:
        n_spectrum_ifg = np.arange(0, n_ifg, dtype=np.float64) * fi_lbl
        plot_line(conversion_name + '_' + '04.png', n_spectrum_ifg, ifg_lbl, lw=LINEWIDTH)

    # ########## compute delta OPD  截断光谱
    max_x = 1. / (2.0 * fi_lbl)  # 500cm
    dx_other = max_x / (n_spectrum - 1)  # cm 7.2337963e-005  723.3796e-7 cm 723.3796nm

    # compute index at which the interferogram is truncated
    idx_trunc = int(opd_other / dx_other)  # 计算最大光程差对应的截断点位置,也就是个数

    # truncate interferogram
    ifg_other = ifg_lbl[0: idx_trunc + 1]
    n_ifg_other = idx_trunc + 1
    x_other = np.arange(0, n_ifg_other, dtype=np.float64) * dx_other

    # p5 截取后的iasi干涉图
    if conversion_name is not None:
        plot_line(conversion_name + '_' + '05.png', x_other, ifg_other, lw=LINEWIDTH)

    # ########## apply apodization
    ifg_other_ap = ifg_other * apodization(x_other)

    # p6 apodization
    if conversion_name is not None:
        plot_line(conversion_name + '_' + '06.png', x_other, ifg_other, lw=LINEWIDTH)

    # ########## convert ifg to spectrum，做一个对称的光谱
    n_ifg_fft = 2 * (n_ifg_other - 1)  # 干涉图点数拓展
    ifg_fft = np.arange(0, n_ifg_fft, dtype=ifg_other_ap.dtype)
    ifg_fft[0:n_ifg_other] = ifg_other_ap
    ifg_fft[n_ifg_other:] = ifg_other_ap[-2:0:-1]
    # 干涉图拓展 光程差2*2cm 为 4cm  点数n_ifg_fft 55296 间隔
    # 去掉最大值,和最末的值

    # p7 拓展干涉图的示意图
    if conversion_name is not None:
        x_iasi_fft = np.arange(0, n_ifg_fft, dtype=np.float64) * dx_other
        plot_line(conversion_name + '_' + '07.png', x_iasi_fft, ifg_fft, lw=LINEWIDTH)

    # ########## FFT 正变换
    spectrum_other_comp = np.fft.ifft(ifg_fft) * dx_other * n_ifg_fft  # FFT 正变换
    spectrum_other_comp = spectrum_other_comp.real  # 仅使用实数

    # ########## take out other portion of the spectrum
    # 取得iasi的光谱  在干涉图中按2cm最大光程差取完之后，做FFT反变换得到的光谱的分辨率就是0.25cm-1
    nt1 = int(np.floor(fb_other / fi_other + 0.5))
    nt2 = int(np.floor(fe_other / fi_other + 0.5))
    spectrum_other = spectrum_other_comp[nt1: nt2 + 1]
    n_spectrum_other = nt2 - nt1 + 1
    wavenumber_other = np.arange(0, n_spectrum_other, dtype=np.float64)
    wavenumber_other = fb_other + wavenumber_other * frequency_interval_other

    # p08 IASI 光谱
    if conversion_name is not None:
        plot_line(conversion_name + '_' + '08.png', wavenumber_other, spectrum_other,
                  lw=LINEWIDTH)

    return spectrum_other, wavenumber_other


def ori2other(spectrum_ori, frequency_begin_ori, frequency_end_ori, frequency_interval_ori,
              frequency_begin_other, frequency_end_other, frequency_interval_other,
              nyquist_other, opd_other, cos_filter_width,
              apodization_ori=None, apodization_other=None, conversion_name=None,
              ):
    spec_ori = spectrum_ori
    fb_ori = frequency_begin_ori
    fe_ori = frequency_end_ori
    fi_ori = frequency_interval_ori

    fb_other = frequency_begin_other
    fe_other = frequency_end_other
    fi_other = frequency_interval_other

    # ########## 做一条和LBL相同分辨率的光谱，频带宽度和other相同
    # totalnum=width(cm-1)/interval  加1.5是因为要做切趾计算
    n_spectrum = int(np.floor(nyquist_other / fi_ori + 1.5))  # 要在LBL采样的点数 6912001

    spectrum = np.zeros(n_spectrum, dtype=np.float64)  # other光谱
    frequency = np.arange(0, n_spectrum, dtype=np.float64) * fi_ori  # other频率

    is_b = int(np.floor(fb_ori / fi_ori + 0.5))  # 放到other光谱的开始位置,index_spec_begin
    is_e = int(np.floor(fe_ori / fi_ori + 0.5)) + 1  # 放到other光谱的结束位置 index_spec_end

    spectrum[is_b: is_e] = spec_ori  # spec_ori 放到光谱的对应位置
    # p1 原始光谱格栅到iasi的光谱网格上
    if conversion_name is not None:
        plot_line(conversion_name + '_' + '01.png', frequency, spectrum, lw=LINEWIDTH)

    # ########## 使用 COS 滤波器 过滤两端的值 ，这步和LBL转other不同####################
    n_filter = int(np.floor(cos_filter_width / fi_ori + 1.5))  # 滤波的点数

    if_b1 = is_b - n_filter + 1
    if_e1 = is_b + 1
    if_b2 = is_e - 1
    if_e2 = is_e + n_filter - 1

    frequency_filter = frequency[if_b1:if_e1]  # 600-620cm-1

    bfilter = 0.5 * (
            1.0 + np.cos((frequency_filter - frequency_filter[0]) * np.pi / cos_filter_width))
    ffilter = bfilter[::-1]

    spectrum[if_b1:if_e1] = spectrum[is_b] * ffilter
    spectrum[if_b2:if_e2] = spectrum[is_e] * bfilter

    # p2 cos 滤波之后
    if conversion_name is not None:
        plot_line(conversion_name + '_' + '02.png', frequency, spectrum, lw=LINEWIDTH)

    # ########## 原光谱做成一个对称光谱
    # 前半部分同spc 后半部分去掉首末两点的reverse
    n_ifg = 2 * (n_spectrum - 1)  # 干涉图点数 13824000
    spectrum_fft = np.arange(0, n_ifg, dtype=np.float64)
    spectrum_fft[0:n_spectrum] = spectrum  # 前半部分和spectrum相同
    spectrum_fft[n_spectrum:] = spectrum[-2:0:-1]  # 后半部分是spectrum去掉首末两点的倒置

    # p3 对称的光谱图
    if conversion_name is not None:
        frequency_fft = np.arange(0, n_ifg, dtype=np.float64) * fi_ori
        plot_line(conversion_name + '_' + '03.png', frequency_fft, spectrum_fft, lw=LINEWIDTH)

    # ########## inverse fft 反傅里叶变换
    ifg_ori = np.fft.fft(spectrum_fft) * fi_ori
    # 傅里叶反变换，转换为双边干涉图，光程差 500cm ，双边 1000cm，间隔 dx
    # 共n_ifg个点，13824000，是全部的干涉图，需要截取

    # p4 傅里叶转换后的干涉图
    if conversion_name is not None:
        n_spectrum_ifg = np.arange(0, n_ifg, dtype=np.float64) * fi_ori
        plot_line(conversion_name + '_' + '04.png', n_spectrum_ifg, ifg_ori, lw=LINEWIDTH)

    # ########## compute delta OPD  截断光谱
    max_x = 1. / (2.0 * fi_ori)  # 500cm
    dx_other = max_x / (n_spectrum - 1)  # cm 7.2337963e-005  723.3796e-7 cm 723.3796nm

    # compute index at which the interferogram is truncated
    idx_trunc = int(opd_other / dx_other)  # 计算最大光程差对应的截断点位置,也就是个数

    # truncate interferogram
    ifg_other = ifg_ori[0: idx_trunc + 1]
    n_ifg_other = idx_trunc + 1
    x_other = np.arange(0, n_ifg_other, dtype=np.float64) * dx_other

    # p5 截取后的iasi干涉图
    if conversion_name is not None:
        plot_line(conversion_name + '_' + '05.png', x_other, ifg_other, lw=LINEWIDTH)

    # ########## 移除原来的 apod，应用新的 apod
    ifg_ap = ifg_other / apodization_ori(x_other)
    ifg_other_ap = ifg_ap * apodization_other(x_other)

    # p6 apodization
    if conversion_name is not None:
        plot_line(conversion_name + '_' + '06.png', x_other, ifg_other, lw=LINEWIDTH)

    # ########## convert ifg to spectrum，做一个对称的光谱
    n_ifg_fft = 2 * (n_ifg_other - 1)  # 干涉图点数拓展
    ifg_fft = np.arange(0, n_ifg_fft, dtype=ifg_other_ap.dtype)
    ifg_fft[0:n_ifg_other] = ifg_other_ap
    ifg_fft[n_ifg_other:] = ifg_other_ap[-2:0:-1]
    # 干涉图拓展 光程差2*2cm 为 4cm  点数n_ifg_fft 55296 间隔
    # 去掉最大值,和最末的值

    # p7 拓展干涉图的示意图
    if conversion_name is not None:
        x_iasi_fft = np.arange(0, n_ifg_fft, dtype=np.float64) * dx_other
        plot_line(conversion_name + '_' + '07.png', x_iasi_fft, ifg_fft, lw=LINEWIDTH)

    # ########## FFT 正变换
    spectrum_other_comp = np.fft.ifft(ifg_fft) * dx_other * n_ifg_fft  # FFT 正变换
    spectrum_other_comp = spectrum_other_comp.real  # 仅使用实数

    # ########## take out other portion of the spectrum
    # 取得iasi的光谱  在干涉图中按2cm最大光程差取完之后，做FFT反变换得到的光谱的分辨率就是0.25cm-1
    nt1 = int(np.floor(fb_other / fi_other + 0.5))
    nt2 = int(np.floor(fe_other / fi_other + 0.5))
    spectrum_other = spectrum_other_comp[nt1: nt2 + 1]
    n_spectrum_iasi = nt2 - nt1 + 1
    wavenumber_other = np.arange(0, n_spectrum_iasi, dtype=np.float64)
    wavenumber_other = fb_other + wavenumber_other * frequency_interval_other

    # p08 IASI 光谱
    if conversion_name is not None:
        plot_line(conversion_name + '_' + '08.png', wavenumber_other, spectrum_other,
                  lw=LINEWIDTH)

    return spectrum_other, wavenumber_other


def plot_line(out_file, data1, data2=None, **kwargs):
    fig_size = (6.4, 4.8)
    dpi = 100
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    if data2 is None:
        ax1.plot(data1, **kwargs)
    else:
        ax1.plot(data1, data2, **kwargs)
    fig.savefig(out_file, dpi=100)
    fig.clear()
    plt.close()
    print('>>> {}'.format(out_file))


def plot_scatter(out_file, data1, data2=None, **kwargs):
    fig_size = (6.4, 4.8)
    dpi = 100
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    if data2 is None:
        ax1.scatter(data1, **kwargs)
    else:
        ax1.scatter(data1, data2, **kwargs)
    ax1.set_ylim(-3, 3)
    fig.savefig(out_file, dpi=100)
    fig.clear()
    plt.close()
    print('>>> {}'.format(out_file))
