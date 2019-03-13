#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/1/10
@Author  : AnNing
"""
import os
import matplotlib.pyplot as plt

from plot_core import PlotAx

STYLE_PATH = os.path.split(os.path.realpath(__file__))[0]


def plot_scatter(x, y, out_file, format_kwargs=None, plot_kwargs=None):
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
    ax1.scatter(x, y, **plot_kwargs)

    # ##### 格式化图片
    plot_ax.format_ax(ax1, **format_kwargs)
    plt.tight_layout()
    # ##### 保存图片
    fig.savefig(out_file, dpi=100)
    fig.clear()
    plt.close()
    print('>>> {}'.format(out_file))


def plot_line(x, y, out_file, format_kwargs=None, plot_kwargs=None):
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
    ax1.plot(x, y, **plot_kwargs)

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


def plot_model_validate(tbb_test, tbb_predict, wavenumber, index, out_file):
    """
    对模型结果进行验证
    """
    bias = tbb_predict - tbb_test

    style_path = STYLE_PATH
    style_file = os.path.join(style_path, 'plot_regression.mplstyle')
    plt.style.use(style_file)
    fig = plt.figure(figsize=(6.4, 3), dpi=120)
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

    for s, d in index:
        ax1.plot(wavenumber[s:d], bias.mean(axis=0)[s:d], lw=0.5)
        ax1.set_ylim(-0.2, 0.2)
        ax1.set_ylabel('Bias Mean $(K)$')
        ax2.plot(wavenumber[s:d], bias.std(axis=0)[s:d], lw=0.5)
        ax2.set_ylim(0.0, 1)
        ax2.set_xlabel('Wavenumber $(cm^{-1})$')
        ax2.set_ylabel('Bias Std $(K)$')
    # ##### 保存图片
    fig.savefig(out_file, dpi=100)
    fig.clear()
    plt.close()
    print('>>> {}'.format(out_file))


def plot_conversion_picture(plot_data, name):
    format_kwargs = {
        'x_axis_min': 0,
        'x_axis_max': 3000,
        'x_interval': 500,
        'x_label': 'Wavenumber($cm^{-1}$)',

        'y_axis_min': 0,
        'y_axis_max': 150,
        'y_interval': 30,
        'y_label': 'Radiance($mw/m^2/sr/cm^{-1}$)'

    }
    out_file = name + '_p0.png'
    plot_line(plot_data['p0_x'], plot_data['p0_y'], out_file, format_kwargs=format_kwargs)

    format_kwargs = {
        'x_axis_min': 0,
        'x_axis_max': 7000,
        'x_interval': 500,
        'x_label': 'Wavenumber($cm^{-1}$)',

        'y_axis_min': 0,
        'y_axis_max': 150,
        'y_interval': 30,
        'y_label': 'Radiance($mw/m^2/sr/cm^{-1}$)'

    }
    out_file = name + '_p1.png'
    plot_line(plot_data['p1_x'], plot_data['p1_y'], out_file, format_kwargs=format_kwargs)

    format_kwargs = {
        'x_axis_min': 600,
        'x_axis_max': 800,
        'x_interval': 50,
        'x_label': 'Wavenumber($cm^{-1}$)',

        'y_axis_min': 0,
        'y_axis_max': 150,
        'y_interval': 30,
        'y_label': 'Radiance($mw/m^2/sr/cm^{-1}$)'

    }
    out_file = name + '_p11.png'
    plot_line(plot_data['p1_x'], plot_data['p1_y'], out_file, format_kwargs=format_kwargs)

    format_kwargs = {
        'x_axis_min': 0,
        'x_axis_max': 7000,
        'x_interval': 500,
        'x_label': 'Wavenumber($cm^{-1}$)',

        'y_axis_min': 0,
        'y_axis_max': 150,
        'y_interval': 30,
        'y_label': 'Radiance($mw/m^2/sr/cm^{-1}$)'

    }
    out_file = name + '_p2.png'
    plot_line(plot_data['p2_x'], plot_data['p2_y'], out_file, format_kwargs=format_kwargs)

    format_kwargs = {
        'x_axis_min': 600,
        'x_axis_max': 800,
        'x_interval': 50,
        'x_label': 'Wavenumber($cm^{-1}$)',

        'y_axis_min': 0,
        'y_axis_max': 150,
        'y_interval': 30,
        'y_label': 'Radiance($mw/m^2/sr/cm^{-1}$)'

    }
    out_file = name + '_p22.png'
    plot_line(plot_data['p2_x'], plot_data['p2_y'], out_file, format_kwargs=format_kwargs)

    format_kwargs = {
        'x_axis_min': 0,
        'x_axis_max': 12000,
        'x_interval': 2000,
        'x_label': 'Wavenumber($cm^{-1}$)',

        'y_axis_min': 0,
        'y_axis_max': 150,
        'y_interval': 30,
        'y_label': 'Radiance($mw/m^2/sr/cm^{-1}$)'

    }
    out_file = name + '_p3.png'
    plot_line(plot_data['p3_x'], plot_data['p3_y'], out_file, format_kwargs=format_kwargs)

    format_kwargs = {
        'x_axis_min': -1000,
        'x_axis_max': 13000,
        'x_interval': 3000,
        'x_label': 'Mirror position($cm$)',

        'y_axis_min': -60000,
        'y_axis_max': 80000,
        'y_interval': 20000,
        'y_label': 'Light measured by detector',
    }
    out_file = name + '_p4.png'
    plot_line(plot_data['p4_x'], plot_data['p4_y'], out_file, format_kwargs=format_kwargs)

    format_kwargs = {
        'x_axis_min': -1000,
        'x_axis_max': 13000,
        'x_interval': 3000,
        'x_label': 'Mirror position($cm$)',

        'y_axis_min': -4000,
        'y_axis_max': 4000,
        'y_interval': 2000,
        'y_label': 'Light measured by detector',

    }
    out_file = name + '_p41.png'
    plot_line(plot_data['p4_x'], plot_data['p4_y'], out_file, format_kwargs=format_kwargs)

    format_kwargs = {
        'x_axis_min': -0.2,
        'x_axis_max': 1.0,
        'x_interval': 0.2,
        'x_label': 'Mirror position($cm$)',

        'y_axis_min': -60000,
        'y_axis_max': 80000,
        'y_interval': 20000,
        'y_label': 'Light measured by detector',
    }
    out_file = name + '_p5.png'
    plot_line(plot_data['p5_x'], plot_data['p5_y'], out_file, format_kwargs=format_kwargs)

    format_kwargs = {
        'x_axis_min': -0.2,
        'x_axis_max': 1.0,
        'x_interval': 0.2,
        'x_label': 'Mirror position($cm$)',

        'y_axis_min': -500,
        'y_axis_max': 500,
        'y_interval': 200,
        'y_label': 'Light measured by detector',
    }
    out_file = name + '_p51.png'
    plot_line(plot_data['p5_x'], plot_data['p5_y'], out_file, format_kwargs=format_kwargs)

    format_kwargs = {
        'x_axis_min': -0.2,
        'x_axis_max': 1.0,
        'x_interval': 0.2,
        'x_label': 'Mirror position($cm$)',

        'y_axis_min': -60000,
        'y_axis_max': 80000,
        'y_interval': 20000,
        'y_label': 'Light measured by detector',
    }
    out_file = name + '_p6.png'
    plot_line(plot_data['p6_x'], plot_data['p6_y'], out_file, format_kwargs=format_kwargs)

    format_kwargs = {
        'x_axis_min': -0.2,
        'x_axis_max': 1.0,
        'x_interval': 0.2,
        'x_label': 'Mirror position($cm$)',

        'y_axis_min': -500,
        'y_axis_max': 500,
        'y_interval': 200,
        'y_label': 'Light measured by detector',
    }
    out_file = name + '_p61.png'
    plot_line(plot_data['p6_x'], plot_data['p6_y'], out_file, format_kwargs=format_kwargs)

    format_kwargs = {
        'x_axis_min': -0.5,
        'x_axis_max': 2.0,
        'x_interval': 0.5,
        'x_label': 'Mirror position($cm$)',

        'y_label': 'Light measured by detector',
    }
    out_file = name + '_p7.png'
    plot_line(plot_data['p7_x'], plot_data['p7_y'], out_file, format_kwargs=format_kwargs)

    format_kwargs = {
        'x_axis_min': 0,
        'x_axis_max': 3000,
        'x_interval': 500,
        'x_label': 'Wavenumber($cm^{-1}$)',

        'y_axis_min': 0,
        'y_axis_max': 150,
        'y_interval': 30,
        'y_label': 'Radiance($mw/m^2/sr/cm^{-1}$)'

    }
    out_file = name + '_p8.png'
    plot_line(plot_data['p8_x'], plot_data['p8_y'], out_file, format_kwargs=format_kwargs)
