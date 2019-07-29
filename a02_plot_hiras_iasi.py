#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/7/24
@Author  : AnNing
"""
from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.pyplot.switch_backend('agg')
import matplotlib.pyplot as plt

from src.data_loader import LoaderMatch
from src.util import rad2tbb

STYLE_FILE = 'src/plot_regression.mplstyle'
plt.style.use(STYLE_FILE)


def main(version):
    in_dir = r'/nas01/Data_gapfilling/match_HIRAS+IASI_{}'.format(version)
    out_dir = '/nas01/Data_gapfilling/match_HIRAS+IASI/Picture'
    for i in (in_dir, out_dir):
        if not os.path.isdir(i):
            os.makedirs(i)
    file_names = os.listdir(in_dir)
    file_names.sort()
    for file_name in file_names:
        in_file = os.path.join(in_dir, file_name)
        if not os.path.isfile(in_file):
            continue
        try:
            loader = LoaderMatch(in_file)
            wn = loader.get_wn_full(sat_num=1)
            rad1_tem = loader.get_rad_full(sat_num=1)
            rad2_tem = loader.get_rad_full(sat_num=2)
            solz_tem = loader.get_solz(sat_num=1)
        except IOError as why:
            print(why)
            print(in_file)
            continue

        data_size = solz_tem.size
        print('总数据： {}'.format(data_size))
        if data_size <= 0:
            print('没有足够的有效数据')
            continue

        # 白天数据过滤,只保留晚上数据
        index = solz_tem > 120
        print('白天数据过滤：{}'.format(index.sum()))
        if not index.any():
            print('没有足够的白天数据')
            continue

        rad1_tem = rad1_tem[index]
        rad2_tem = rad2_tem[index]

        # 无效值过滤
        index = []
        for i in range(rad1_tem.shape[0]):
            if (rad1_tem[i] > 0).all() or (rad1_tem[i] > 0).all():
                index.append(i)
        print("无效值过滤： {}".format(len(index)))
        if not index:
            continue
        rad1_tem = rad1_tem[index]
        rad2_tem = rad2_tem[index]

        tbb1_tem = rad2tbb(rad1_tem, wn)
        tbb2_tem = rad2tbb(rad2_tem, wn)

        # 无效温度过滤
        index = []
        for i in range(tbb1_tem.shape[0]):
            if (tbb1_tem[i] > 350).any() or (tbb2_tem[i] > 350).any():
                continue
            elif (tbb1_tem[i] < 150).any() or (tbb2_tem[i] < 150).any():
                continue
            else:
                index.append(i)
        print(u'无效温度过滤： {}'.format(len(index)))
        if not index:
            continue

        tbb1_tem = tbb1_tem[index]
        tbb2_tem = tbb2_tem[index]

        # 绘图
        tbb1 = tbb1_tem
        tbb2 = tbb2_tem

        tbb1_mean = tbb1.mean(axis=0)
        tbb2_mean = tbb2.mean(axis=0)
        tbb_bias = tbb1 - tbb2
        tbb_bias_mean = tbb_bias.mean(axis=0)

        fig_file = os.path.join(out_dir, file_name + '_bias.jpg')
        fig_size = (12.8, 4.8)
        dpi = 100
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        ax1 = plt.subplot2grid((1, 1), (0, 0))
        ax1.plot(wn, tbb_bias_mean)
        ax1.set_xlim(600, 2800)
        fig.savefig(fig_file, dpi=200)
        fig.clear()
        plt.close()
        print('>>> :{}'.format(fig_file))

        fig_file = os.path.join(out_dir, file_name + '_tbb1.jpg')
        fig_size = (12.8, 4.8)
        dpi = 100
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        ax1 = plt.subplot2grid((1, 1), (0, 0))
        ax1.plot(wn, tbb1_mean)
        ax1.set_xlim(600, 2800)
        fig.savefig(fig_file, dpi=200)
        fig.clear()
        plt.close()
        print('>>> :{}'.format(fig_file))

        fig_file = os.path.join(out_dir, file_name + '_tbb2.jpg')
        fig_size = (12.8, 4.8)
        dpi = 100
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        ax1 = plt.subplot2grid((1, 1), (0, 0))
        ax1.plot(wn, tbb2_mean)
        ax1.set_xlim(600, 2800)
        fig.savefig(fig_file, dpi=200)
        fig.clear()
        plt.close()
        print('>>> :{}'.format(fig_file))

        fig_file = os.path.join(out_dir, file_name + '_tbb1_tbb2_bias.jpg')
        fig_size = (12.8, 4.8)
        dpi = 100

        fig = plt.figure(figsize=fig_size, dpi=dpi)
        ax1 = plt.subplot2grid((2, 1), (0, 0))
        ax2 = plt.subplot2grid((2, 1), (1, 0))
        ax1.plot(wn, tbb1_mean, color='red', alpha=0.5)
        ax1.plot(wn, tbb2_mean, color='blue', alpha=0.5)
        ax2.plot(wn, tbb_bias_mean)
        ax1.set_xlim(600, 2800)
        fig.savefig(fig_file, dpi=200)
        fig.clear()
        plt.close()
        print('>>> :{}'.format(fig_file))

        fig_file = os.path.join(out_dir, file_name + '_tbb1_tbb2.jpg')
        fig_size = (12.8, 4.8)
        dpi = 100
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        ax1 = plt.subplot2grid((1, 1), (0, 0))
        ax1.plot(wn, tbb1_mean, color='red', alpha=0.5)
        ax1.plot(wn, tbb2_mean, color='blue', alpha=0.5)
        ax1.set_xlim(600, 2800)

        fig.savefig(fig_file, dpi=200)
        fig.clear()
        plt.close()
        print('>>> :{}'.format(fig_file))


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args[0])
