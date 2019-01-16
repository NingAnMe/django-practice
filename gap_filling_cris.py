#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/1/15
@Author  : AnNing
"""
import os
import sys
from data_loader import LoaderCrisFull


def main(dir_in):
    in_files = os.listdir(dir_in)
    # 加载数据
    for filename in in_files:
        in_file = os.path.join(dir_in, filename)
        loader_cris_full = LoaderCrisFull(in_file)
        radiances = loader_cris_full.get_spectrum_radiance()


# ######################## 带参数的程序入口 ##############################
if __name__ == "__main__":
    # 获取程序参数接口
    ARGS = sys.argv[1:]
    HELP_INFO = \
        u"""
        [arg1]：dir_in
        [arg2]：dir_out
        [example]： python app.py arg1 arg2
        """
    if "-h" in ARGS:
        print(HELP_INFO)
        sys.exit(-1)

    if len(ARGS) != 1:
        print(HELP_INFO)
        sys.exit(-1)
    else:
        ARG1 = ARGS[0]
        ARG2 = ARGS[1]
        main(ARG1, ARG2)
