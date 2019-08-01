#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/3/13
@Author  : AnNing
"""
import os

import numpy as np

from plot_core import plt, STYLE_PATH

style_path = STYLE_PATH
style_file = os.path.join(style_path, 'plot_regression.mplstyle')
plt.style.use(style_file)


def plot_model_validate(tbb_test, tbb_predict, wavenumber, index, out_file):
    """
    对模型结果进行验证
    """
    bias = tbb_predict - tbb_test

    fig = plt.figure(figsize=(6.4, 3), dpi=120)
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)
    #     ax1 = plt.subplot2grid((4, 1), (0, 0))
    #     ax2 = plt.subplot2grid((4, 1), (1, 0), sharex=ax1)
    #     ax3 = plt.subplot2grid((4, 1), (2, 0), sharex=ax1)
    #     ax4 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)

    lw = 1

    for s, d in index:
        ax1.plot(wavenumber[s:d], bias.mean(axis=0)[s:d], lw=lw)
        #         ax1.set_ylim(-0.2, 0.2)
        ax1.set_ylabel('TBB Bias Mean $(K)$')
        ax2.plot(wavenumber[s:d], bias.std(axis=0)[s:d], lw=lw)
        #         ax2.set_ylim(0.0, 1)
        ax2.set_xlabel('Wavenumber $(cm^{-1})$')
        ax2.set_ylabel('TBB Bias Std $(K)$')
    #         ax3.plot(wavenumber[s:d], bias_abs.mean(axis=0)[s:d], lw=lw)
    #         ax3.set_ylim(-0.2, 0.2)
    #         ax3.set_ylabel('Abs Bias Mean $(K)$')
    #         ax4.plot(wavenumber[s:d], bias_abs.std(axis=0)[s:d], lw=lw)
    #         ax4.set_ylim(0.0, 1)
    #         ax4.set_xlabel('Wavenumber $(cm^{-1})$')
    #         ax4.set_ylabel('Abs Bias Std $(K)$')
    # ##### 保存图片
    fig.savefig(out_file, dpi=100)
    plt.show()
    fig.clear()
    plt.close()
    print('>>> {}'.format(out_file))
