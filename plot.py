#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/1/10
@Author  : AnNing
"""
import matplotlib.pyplot as plt


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


def plot_p0(x, y, out_file):
    fig_size = (6.4, 4.8)
    dpi = 100
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.plot(x, y)
    fig.savefig(out_file, dpi=100)
    fig.clear()
    plt.close()
    print('>>> {}'.format(out_file))


def plot_picture(datas, name):
    out_file = name + '_p0.png'
    plot_p0(datas['p0_x'], datas['p0_y'], out_file)
