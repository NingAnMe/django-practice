#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/7/19
@Author  : AnNing
"""
import os
import yaml


def main():
    in_dir = ''
    out_dir = ''
    file_names = os.listdir(in_dir)
    in_files = [os.path.join(in_dir, filename) for filename in file_names]

    for in_file in in_files:
        with open(in_file, 'r') as f:
            file_data = yaml.load(f)
        