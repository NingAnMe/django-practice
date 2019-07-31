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

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from src.data_loader import LoaderMatch
from src.util import rad2tbb

STYLE_FILE = 'src/plot_regression.mplstyle'
plt.style.use(STYLE_FILE)


def main(version):
    in_dir = r'/nas01/Data_gapfilling/match_HIRAS+IASI_{}'.format(version)
    out_dir = '/nas01/Data_gapfilling/day/year_picture_HIRAS+IASI_{}'.format(version)
    for i in (in_dir, out_dir):
        if not os.path.isdir(i):
            os.makedirs(i)
    file_names = os.listdir(in_dir)
    file_names.sort()

    tbb1_all = None
    tbb2_all = None
    solz_all = None
    wn = None
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
        if data_size <= 1:
            print('没有足够的有效数据')
            continue

        # 白天数据过滤,只保留晚上数据
        index = solz_tem < 70
        print('白天数据过滤：{}'.format(index.sum()))
        if not index.any():
            print('没有足够的白天数据')
            continue

        rad1_tem = rad1_tem[index]
        rad2_tem = rad2_tem[index]
        solz_tem = solz_tem[index]

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
        solz_tem = solz_tem[index]

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
        solz_tem = solz_tem[index]

        data_size = solz_tem.size
        print('过滤后的总数据： {}'.format(data_size))
        if data_size <= 1:
            print('过滤后没有足够的有效数据')
            continue
        elif tbb1_all is None:
            tbb1_all = tbb1_tem
            tbb2_all = tbb2_tem
            solz_all = solz_tem
            wn = wn
        else:
            tbb1_all = np.concatenate((tbb1_all, tbb1_tem), axis=0)
            tbb2_all = np.concatenate((tbb2_all, tbb2_tem), axis=0)
            solz_all = np.concatenate((solz_all, solz_tem), axis=0)

    # 绘图
    tbb1 = tbb1_all
    tbb2 = tbb2_all
    data_size = solz_all.size
    print('过滤后的总数据： {}'.format(data_size))
    if data_size is None or data_size <= 1:
        print('过滤后没有足够的有效数据')
        return

    file_name = file_names[-1]
    tbb1_ylim = (210, 280)

    tbb1_mean = np.nanmean(tbb1, axis=0)
    tbb2_mean = np.nanmean(tbb2, axis=0)
    tbb_bias = tbb1 - tbb2
    tbb_bias_mean = np.nanmean(tbb_bias, axis=0)

    # fig_file = os.path.join(out_dir, file_name + '_tbb_all.jpg')
    # fig_size = (12.8, 4.8)
    # dpi = 200
    # fig = plt.figure(figsize=fig_size, dpi=dpi)
    # ax1 = plt.subplot2grid((1, 1), (0, 0))
    # ax1.plot(wn, tbb1_mean, color='red', alpha=0.5, label='HIRAS')
    # ax1.plot(wn, tbb2_mean, color='blue', alpha=0.5, label='IASI')
    # ax1.set_xlim(600, 2800)
    # ax1.set_ylim(tbb1_ylim)
    # ax1.legend()
    #
    # plt.xlabel('Wavenumber($cm^{-1}$)')
    # plt.ylabel('TBB ($K$)')
    # fig.savefig(fig_file, dpi=dpi)
    # fig.clear()
    # plt.close()
    # print('>>> :{}'.format(fig_file))
    #
    # fig_file = os.path.join(out_dir, file_name + '_bias.jpg')
    # fig_size = (12.8, 4.8)
    # dpi = 100
    # fig = plt.figure(figsize=fig_size, dpi=dpi)
    # ax1 = plt.subplot2grid((1, 1), (0, 0))
    # ax1.plot(wn, tbb_bias_mean)
    # ax1.set_xlim(600, 2800)
    # ax1.set_ylim(-4, 4)
    #
    # plt.xlabel('Wavenumber($cm^{-1}$)')
    # plt.ylabel('TBB Bias($K$) HIRAS-IASI')
    # fig.savefig(fig_file, dpi=dpi)
    # fig.clear()
    # plt.close()
    # print('>>> :{}'.format(fig_file))
    #
    # fig_file = os.path.join(out_dir, file_name + '_tbb1.jpg')
    # fig_size = (12.8, 4.8)
    # dpi = 100
    # fig = plt.figure(figsize=fig_size, dpi=dpi)
    # ax1 = plt.subplot2grid((1, 1), (0, 0))
    # ax1.plot(wn, tbb1_mean)
    # ax1.set_xlim(600, 2800)
    # ax1.set_ylim(tbb1_ylim)
    #
    # plt.xlabel('Wavenumber($cm^{-1}$)')
    # plt.ylabel('TBB($K$) HIRAS')
    # fig.savefig(fig_file, dpi=dpi)
    # fig.clear()
    # plt.close()
    # print('>>> :{}'.format(fig_file))
    #
    # fig_file = os.path.join(out_dir, file_name + '_tbb2.jpg')
    # fig_size = (12.8, 4.8)
    # dpi = 100
    # fig = plt.figure(figsize=fig_size, dpi=dpi)
    # ax1 = plt.subplot2grid((1, 1), (0, 0))
    # ax1.plot(wn, tbb2_mean)
    # ax1.set_xlim(600, 2800)
    # ax1.set_ylim(tbb1_ylim)
    #
    # plt.xlabel('Wavenumber($cm^{-1}$)')
    # plt.ylabel('TBB($K$) IASI')
    # fig.savefig(fig_file, dpi=dpi)
    # fig.clear()
    # plt.close()
    # print('>>> :{}'.format(fig_file))
    #
    # fig_file = os.path.join(out_dir, file_name + '_bias_all.jpg')
    # fig_size = (12.8, 4.8)
    # dpi = 100
    #
    # fig = plt.figure(figsize=fig_size, dpi=dpi)
    # ax1 = plt.subplot2grid((2, 1), (0, 0))
    # ax2 = plt.subplot2grid((2, 1), (1, 0))
    # ax1.plot(wn, tbb1_mean, color='red', alpha=0.5)
    # ax1.plot(wn, tbb2_mean, color='blue', alpha=0.5)
    # ax2.plot(wn, tbb_bias_mean)
    # ax1.set_xlim(600, 2800)
    # ax1.set_ylim(tbb1_ylim)
    # ax1.set_xlabel('Wavenumber($cm^{-1}$)')
    # ax1.set_ylabel('TBB($K$)')
    #
    # ax2.set_xlim(600, 2800)
    # ax2.set_ylim(-4, 4)
    # ax2.set_xlabel('Wavenumber($cm^{-1}$)')
    # ax2.set_ylabel('TBB Bias($K$) HIRAS-IASI')
    #
    # fig.savefig(fig_file, dpi=dpi)
    # fig.clear()
    # plt.close()
    # print('>>> :{}'.format(fig_file))
    return tbb_bias_mean, wn


# if __name__ == '__main__':
#     args = sys.argv[1:]
#     for v in args:
#         main(v)

if __name__ == '__main__':
    tbb_bias_new, wn = main('new')
    tbb_bias_night, _ = main('night')
    tbb_bias_xuhui, _ = main('xuhui')

    out_dir = 'picture'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    file_name = 'dif'
    fig_file = os.path.join(out_dir, file_name + '_bias_day.jpg')
    fig_size = (12.8, 4.8)
    dpi = 100
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.plot(wn, tbb_bias_new, alpha=0.5, label='new')
    ax1.plot(wn, tbb_bias_night, alpha=0.5, label='night')
    ax1.plot(wn, tbb_bias_xuhui, alpha=0.5, label='xuhui')
    ax1.set_xlim(600, 2800)
    ax1.set_ylim(-4, 4)
    ax1.legend()

    plt.xlabel('Wavenumber($cm^{-1}$)')
    plt.ylabel('TBB Bias($K$) HIRAS-IASI')
    fig.savefig(fig_file, dpi=dpi)
    fig.clear()
    plt.close()
    print('>>> :{}'.format(fig_file))
