#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/11/15 14:51
# @Author  : NingAnMe <ninganme@qq.com>
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize('iasi2giirs.py')
)
