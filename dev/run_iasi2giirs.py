#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/11/15 15:46
# @Author  : NingAnMe <ninganme@qq.com>
"""
INPUT：
    radiance_iasi: shape为(8461,)

OUTPUT：
    radiance_giirs
    wavenumber_giirs

产生的GIIRS光谱范围为：645.625 - 2760.
使用的光谱范围为：650. - 2755.

"""
from iasi2giirs import iasi2giirs

radiance_giirs, wavenumber_giirs = iasi2giirs(radiance=radiance_iasi)
