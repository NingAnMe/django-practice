#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/1/14
@Author  : AnNing
"""
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

try:
    import harp
    import coda
except ImportError:
    print('harp model is not existed! Cant load IASI data')

from plot_core import STYLE_PATH, PlotAx


class LoaderCrisL1:
    def __init__(self, in_file):
        self.in_file = in_file

    def get_radiance(self):
        pass

    def get_spectrum_radiance(self):
        with h5py.File(self.in_file, 'r') as h5r:
            sds_name = '/All_Data/CrIS-FS-SDR_All/ES_RealLW'
            sr_lw = h5r.get(sds_name).value

            sds_name = '/All_Data/CrIS-FS-SDR_All/ES_RealMW'
            sr_mw = h5r.get(sds_name).value

            sds_name = '/All_Data/CrIS-FS-SDR_All/ES_RealSW'
            sr_sw = h5r.get(sds_name).value

            # 切趾计算
            w0 = 0.23
            w1 = 0.54
            w2 = 0.23
            sr_lw[:, :, :, 1:-1] = w0 * sr_lw[:, :, :, :-2] + w1 * sr_lw[:, :, :, 1:-1] + w2 * sr_lw[:, :, :, 2:]
            sr_mw[:, :, :, 1:-1] = w0 * sr_mw[:, :, :, :-2] + w1 * sr_mw[:, :, :, 1:-1] + w2 * sr_mw[:, :, :, 2:]
            sr_sw[:, :, :, 1:-1] = w0 * sr_sw[:, :, :, :-2] + w1 * sr_sw[:, :, :, 1:-1] + w2 * sr_sw[:, :, :, 2:]

            sr_lw = sr_lw[:, :, :, 2:-2]
            sr_mw = sr_mw[:, :, :, 2:-2]
            sr_sw = sr_sw[:, :, :, 2:-2]

            # response = np.concatenate((sr_lw, sr_mw, sr_sw), axis=3)
            # response.reshape(-1, response[-1])

        return sr_lw, sr_mw, sr_sw

    @staticmethod
    def get_spectrum_wavenumber():
        wavenumber_lw = np.arange(650., 1095.0 + 0.625, 0.625)
        wavenumber_mw = np.arange(1210.0, 1750 + 0.625, 0.625)
        wavenumber_sw = np.arange(2155.0, 2550.0 + 0.625, 0.625)
        return wavenumber_lw, wavenumber_mw, wavenumber_sw

    def get_spectrum_response_wavenumber(self):
        pass


class LoadRealCris:
    def __init__(self, in_file):
        self.in_file = in_file

    def get_spectrum_radiance(self, coeff_file):
        """
        return 光谱波数和响应值，1维，2维
        """

        shape = (16200, 1)
        # 增加切趾计算
        w0 = 0.23
        w1 = 0.54
        w2 = 0.23
        data_file = self.in_file
        with h5py.File(data_file, 'r') as h5r:
            sds_name = '/All_Data/CrIS-FS-SDR_All/ES_RealLW'
            real_lw = h5r.get(sds_name)[:]

            sds_name = '/All_Data/CrIS-FS-SDR_All/ES_RealMW'
            real_mw = h5r.get(sds_name)[:]

            sds_name = '/All_Data/CrIS-FS-SDR_All/ES_RealSW'
            real_sw = h5r.get(sds_name)[:]

        # 切趾计算 w0*n-1 + w1*n + w2*n+1 当作n位置的修正值
        # 开头和结尾不参与计算
        real_lw[:, :, :, 1:-1] = w0 * real_lw[:, :, :, :-2] + \
            w1 * real_lw[:, :, :, 1:-1] + w2 * real_lw[:, :, :, 2:]
        real_mw[:, :, :, 1:-1] = w0 * real_mw[:, :, :, :-2] + \
            w1 * real_mw[:, :, :, 1:-1] + w2 * real_mw[:, :, :, 2:]
        real_sw[:, :, :, 1:-1] = w0 * real_sw[:, :, :, :-2] + \
            w1 * real_sw[:, :, :, 1:-1] + w2 * real_sw[:, :, :, 2:]
        print(real_lw.shape)
        real_lw = real_lw[:, :, :, 2:-2]
        real_mw = real_mw[:, :, :, 2:-2]
        real_sw = real_sw[:, :, :, 2:-2]
        print(real_lw.shape)

        # 波数范围和步长
        wave_number = np.arange(650., 2755.0 + 0.625, 0.625)

        response_old = np.concatenate((real_lw, real_mw, real_sw), axis=3)
        last_s = response_old.shape[-1]
        # 16200*最后一个光谱维度
        response_old = response_old.reshape(shape[0], last_s)

        if not os.path.isfile(coeff_file):
            raise ValueError('Data file is not exist. {}'.format(data_file))
        with h5py.File(coeff_file, 'r') as h5r:
            c0 = h5r.get('C0')[:]
            p0 = h5r.get('P0')[:]
            gap_num = h5r.get('GAP_NUM')[:]

        response_new = np.dot(response_old, p0)
        response_new = response_new + c0
        print('4', response_new.shape)
        ch_part1 = gap_num[0]
        ch_part2 = gap_num[0] + gap_num[1]
        ch_part3 = gap_num[0] + gap_num[1] + gap_num[2]
        real_lw_e = response_new[:, 0:ch_part1]
        real_mw_e = response_new[:, ch_part1:ch_part2]
        real_sw_e = response_new[:, ch_part2:ch_part3]

        # 把原响应值 维度转成2维
        real_lw = real_lw.reshape(shape[0], real_lw.shape[-1])
        real_mw = real_mw.reshape(shape[0], real_mw.shape[-1])
        real_sw = real_sw.reshape(shape[0], real_sw.shape[-1])
        response = np.concatenate(
            (real_lw, real_lw_e, real_mw, real_mw_e, real_sw, real_sw_e), axis=1)
        shape = response.shape
        response = response.reshape(shape[0], 1, shape[1])

        return wave_number, response


class LoaderIasiL1:
    def __init__(self, in_file):
        self.in_file = in_file

    def get_spectrum_radiance(self):
        """
        获取辐射率
        :return:
        """
        product = harp.import_product(self.in_file)
        data = product.wavenumber_radiance.data * 1e5
        del product
        return data

    def get_spectrum_wavenumber(self):
        """
        获取波数
        :return:
        """
        product = harp.import_product(self.in_file)
        data = product.wavenumber.data / 1e2
        del product
        return data

    def get_longitude(self):
        """
        获取经度
        :return:
        """
        product = harp.import_product(self.in_file)
        data = product.longitude.data
        del product
        return data

    def get_latitude(self):
        """
        获取纬度
        :return:
        """
        product = harp.import_product(self.in_file)
        data = product.latitude.data
        del product
        return data

    def get_timestamp_utc(self):
        """
        获取时间
        :return:
        """
        product = harp.import_product(self.in_file)
        # 1970-01-01 到 2000-01-01 的总秒数为  946684800
        data = product.datetime.data + 946684800
        del product
        return data

    def get_sun_zenith(self):
        """
        获取太阳天顶角
        :return:
        """
        fp = coda.open(self.in_file)
        angles = coda.fetch(fp, 'MDR', -1, 'MDR', 'GGeoSondAnglesSUN')
        zenith = np.array([])
        for i in angles:
            z = i.reshape(-1)[0::2]
            zenith = np.append(zenith, z)
        return zenith


class LoaderCrisFull:
    def __init__(self, in_file):
        self.in_file = in_file

    def get_spectrum_radiance(self):
        with h5py.File(self.in_file, 'r') as h5r:
            data = h5r.get('spectrum_radiance').value
        return data

    def get_spectrum_wavenumber(self):
        with h5py.File(self.in_file, 'r') as h5r:
            data = h5r.get('spectrum_wavenumber').value
        return data


def plot_cris(in_file, out_file, format_kwargs=None, plot_kwargs=None):
    loader = LoaderCrisL1(in_file)
    wavenumbers = loader.get_spectrum_wavenumber()
    responses = loader.get_spectrum_radiance()

    if format_kwargs is None:
        format_kwargs = {}
    style_path = STYLE_PATH
    style_file = os.path.join(style_path, 'plot_regression.mplstyle')
    plt.style.use(style_file)
    fig_size = (6.4, 4.8)
    dpi = 100
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    plot_ax = PlotAx()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    if plot_kwargs is None:
        plot_kwargs = dict()
    for x, y in zip(wavenumbers, responses):
        ax1.plot(x, y[0, 0, 0], **plot_kwargs)

    # ##### 格式化图片
    if format_kwargs is None:
        format_kwargs = dict()
    plot_ax.format_ax(ax1, **format_kwargs)
    plt.tight_layout()
    # ##### 保存图片
    fig.savefig(out_file, dpi=100)
    fig.clear()
    plt.close()
    print('>>> {}'.format(out_file))


def plot_iasi(in_file, out_file, format_kwargs=None, plot_kwargs=None):
    with h5py.File(in_file, 'r') as h5r:
        wavenumbers = h5r.get('wavenumber').value / 100.
        responses = h5r.get('wavenumber_radiance').value * 10 ** 5

    if format_kwargs is None:
        format_kwargs = {}
    style_path = STYLE_PATH
    style_file = os.path.join(style_path, 'plot_regression.mplstyle')
    plt.style.use(style_file)
    fig_size = (6.4, 4.8)
    dpi = 100
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    plot_ax = PlotAx()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    if plot_kwargs is None:
        plot_kwargs = dict()

    ax1.plot(wavenumbers[0, :8461], responses[0, :8461], **plot_kwargs)

    # ##### 格式化图片
    if format_kwargs is None:
        format_kwargs = dict()
    plot_ax.format_ax(ax1, **format_kwargs)
    plt.tight_layout()
    # ##### 保存图片
    fig.savefig(out_file, dpi=100)
    fig.clear()
    plt.close()
    print('>>> {}'.format(out_file))


if __name__ == '__main__':
    in_file_cris = r'D:\nsmc\gap_filling_data\cris_20180423.h5'

    format_kwargs_cris = {
        'x_axis_min': 0,
        'x_axis_max': 3000,
        'x_interval': 500,
        'x_label': 'Wavenumber($cm^{-1}$)',

        'y_axis_min': 0,
        'y_axis_max': 150,
        'y_interval': 30,
        'y_label': 'Radiance($mw/m^2/sr/cm^{-1}$)'

    }

    out_file_cris = 'pic\CRIS_p0.png'
    plot_cris(in_file_cris, out_file_cris, format_kwargs=format_kwargs_cris)

    in_file_iasi = r'D:\nsmc\gap_filling_data\iasi_20180104.h5'

    format_kwargs_iasi = {
        'x_axis_min': 0,
        'x_axis_max': 3000,
        'x_interval': 500,
        'x_label': 'Wavenumber($cm^{-1}$)',

        'y_axis_min': 0,
        'y_axis_max': 150,
        'y_interval': 30,
        'y_label': 'Radiance($mw/m^2/sr/cm^{-1}$)'

    }

    out_file_iasi = 'pic\IASI_p0.png'
    plot_iasi(in_file_iasi, out_file_iasi, format_kwargs=format_kwargs_iasi)
