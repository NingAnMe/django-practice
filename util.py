#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/1/10
@Author  : AnNing
"""
import numpy as np
from netCDF4 import Dataset
import h5py


def statistics_print(data):
    print(data[0], data[-1], np.size(data), np.min(data), np.max(data), np.mean(data))


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
    f.close()
    return data


def read_hdf5(in_file):
    with h5py.File(in_file, 'r') as hdf5:
        spectrum = hdf5.get('spectrum').value
        wavenumber = hdf5.get('wavenumber').value
        begin_wn = wavenumber[0]
        end_wn = wavenumber[-1]
        wn_interval = wavenumber[1] - wavenumber[0]

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
