#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/7/24
@Author  : AnNing
"""
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    in_dir = ''
    out_dir = ''
    for file_name in os.listdir(in_dir):
        in_file = os.path.join(in_dir, file_name)
