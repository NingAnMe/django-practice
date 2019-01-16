#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/1/10
@Author  : AnNing
"""
import numpy as np
import pandas as pd
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


def rad2tbb(radiance, center_wave):
    """
    辐射率转亮温
    :param radiance: 辐射率
    :param center_wave: 中心波数
    :return: 亮温
    """
    c1 = 1.1910427e-5
    c2 = 1.4387752

    tbb = (c2 * center_wave) / np.log(1 + ((c1 * center_wave ** 3) / radiance))
    return tbb


def read_lbl_nc(in_file):
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


def read_lbl_hdf5(in_file):
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


def get_cris_full_train_data(in_files, x_ranges=None, y_ranges=None, count=None):
    """
    返回训练数据
    :param in_files: 读取数据文件的列表
    :param x_ranges: X 的光谱范围
    :param y_ranges: Y 的光谱范围
    :param count: 读取光谱的数量
    :return:
    """
    data_all = None
    wavenumber = None
    for in_file in in_files:
        with h5py.File(in_file, 'r') as hdf5_r:
            data = hdf5_r.get('spectrum_radiance').value
            if data_all is None:
                data_all = data
            else:
                data_all = np.concatenate((data_all, data), axis=0)
            if count is not None:
                if len(data_all) > count:
                    data_all = data_all[:count+1]
                    break
            if wavenumber is None:
                wavenumber = hdf5_r.get('spectrum_wavenumber').value

    x = list()
    if x_ranges is None:
        x_ranges = [(650., 1095), (1210., 1750.), (2155., 2550.)]
    for start, end in x_ranges:
        index_start = int(np.where(wavenumber == start)[0])
        index_end = int(np.where(wavenumber == end)[0])
        if len(x) <= 0:
            x = data_all[:, index_start:index_end+1]
        else:
            x = np.concatenate((x, data_all[:, index_start:index_end+1]), axis=1)
    y = list()
    if y_ranges is None:
        y_ranges = [(1095., 1210), (1750., 2155.), (2550., 2755.)]
    for start, end in y_ranges:
        index_start = int(np.where(wavenumber == start)[0])
        index_end = int(np.where(wavenumber == end)[0])
        if len(y) <= 0:
            y = data_all[:, index_start+1:index_end]
        else:
            y = np.concatenate((y, data_all[:, index_start+1:index_end]), axis=1)
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    return x, y


if __name__ == '__main__':
    test_file = '/nas01/Data_anning/data/GapFilling/CRISFull/IASI_xxx_1C_M01_20180104003259Z_20180104003555Z_N_O_20180104011400Z__20180104011612'
    x_, y_ = get_cris_full_train_data([test_file, test_file])
    print(x_.shape)
    print(y_.shape)
    print(type(x_))
    print(type(y_))
