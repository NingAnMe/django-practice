#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/7/19
@Author  : AnNing
"""
from __future__ import print_function

import os
import yaml

import numpy as np
from pykdtree.kdtree import KDTree

from src.data_loader import LoaderHirasL1
from PB.DRC.pb_drc_IASI import ReadIasiL1
from hdf5 import write_hdf5_and_compress
from spectrum_conversion import iasi2hiras
from util import iasi_apod

DIST = 0.05  # 距离阈值
IBAND = [0, ]  # band 1、 2 or 3，光谱带
iband = 0

NIGHT = False

# #########  仪器参数 #############

IASI_F_NYQUIST = 6912.0  # 频带宽度  cm-1
IASI_RESAMPLE_MAXX = [2.0, ]  # cm OPD
IASI_D_FREQUENCY = [0.25, ]  # v cm-1  光谱分辨率
IASI_BAND_F1 = [645.25, ]  # 光谱带开始
IASI_BAND_F2 = [2760.25, ]  # 光谱带结束
IASI_FILTER_WIDTH = [20.0, ]  # cm-1  # COS过滤器过滤的宽度

GIIRS_F_NYQUIST = 5875.0
GIIRS_RESAMPLE_MAXX = [0.8, ]
GIIRS_D_FREQUENCY = [0.625, ]
GIIRS_BAND_F1 = [645.625, ]
GIIRS_BAND_F2 = [2760., ]
GIIRS_FILTER_WIDTH = [20.0, ]

# GIIRS的仪器参数和HIRAS相同，HIRAS切趾以后的通道范围为650.625~2755（9:3378）


def main():
    out_dir = '/nas01/Data_gapfilling/match_HIRAS+IASI'
    coeff_file = 'Model/linear_model_attribute_test_hiras.h5'
    in_dir = '/nas03/CMA_GSICS_TEST/CROSS/InterfaceFile/FY3D+MERSI_METOP-A+IASI/job_0110/'
    dir_names = os.listdir(in_dir)
    dir_names.sort()
    for dir_name in dir_names:
        file_names = os.listdir(os.path.join(in_dir, dir_name))

        results = {}
        out_file = os.path.join(out_dir, dir_name) + '.hdf'
        if os.path.isfile(out_file):
            print('Already >>> : {}'.format(out_file))
            continue

        for file_name in file_names:
            in_file = os.path.join(in_dir, dir_name, file_name)

            with open(in_file, 'r') as f:
                file_data = yaml.load(f)
            in_files1 = file_data["PATH"]["ipath1"]
            in_files2 = file_data["PATH"]["ipath2"]

            for in_file1 in in_files1:
                in_file1 = in_file1.replace('MERSI', 'HIRAS')
                in_file1 = in_file1.replace('1000M', '016KM')
                for in_file2 in in_files2:
                    in_file1 = in_file1.replace('/home/gsics', '')
                    in_file2 = in_file2.replace('/home/gsics', '')
                    print(in_file1)
                    print(in_file2)
                    loader_hiras = LoaderHirasL1(in_file1)
                    loader_iasi = ReadIasiL1(in_file2)

                    try:
                        lons1 = loader_hiras.get_longitude().reshape(-1,)
                        lats1 = loader_hiras.get_latitude().reshape(-1,)

                        lons2 = loader_iasi.get_longitude().reshape(-1,)
                        lats2 = loader_iasi.get_latitude().reshape(-1,)

                    except IOError as why:
                        print(why)
                        continue

                    # for data in (lons1, lats1, rad1, wn1, sza1, rad_full1, wn_full1,
                    #              lons2, lats2, rad2, wn2, sza2):
                    #     print(data.shape)
                    #     print('=' * 50)

                    combined_x_y_arrays = np.dstack([lons1.ravel(), lats1.ravel()])[0]
                    points = np.dstack([lons2.ravel(), lats2.ravel()])[0]

                    mytree = KDTree(combined_x_y_arrays)
                    dist, index = mytree.query(points)
                    # a = np.histogram(dist, bins=20)
                    # print(a)

                    index_dist = dist < DIST
                    print(index_dist.sum())
                    if index_dist.sum() < 2:
                        continue

                    wn1, rad1 = loader_hiras.get_radiance()
                    wn1 = wn1.reshape(-1,)
                    rad1 = rad1.reshape(-1, 2275)
                    sza1 = loader_hiras.get_solar_zenith().reshape(-1,)
                    wn_full1, rad_full1 = loader_hiras.get_radiance(coeff_file=coeff_file)

                    wn2, rad2 = loader_iasi.get_spectrum_radiance()
                    wn2 = wn2[:8461]
                    rad2 = rad2[:, :8461]
                    sza2 = loader_iasi.get_solar_zenith().reshape(-1, )
                    wn_full2 = wn_full1

                    try:
                        result = {
                            'S1_Lat': lats1[index][index_dist],
                            'S1_Lon': lons1[index][index_dist],
                            'S1_SolZ': sza1[index][index_dist],
                            'S1_Rad': rad1[index][index_dist],
                            'S1_Wn': wn1,
                            'S1_Rad_full': rad_full1[index][index_dist],
                            'S1_Wn_full': wn_full1,

                            'S2_Lat': lats2[index_dist],
                            'S2_Lon': lons2[index_dist],
                            'S2_SolZ': sza2[index_dist],
                            'S2_Rad': rad2[index_dist],
                            'S2_Wn': wn2,
                            'S2_Wn_full': wn_full2,
                        }
                    except IndexError as why:
                        print(why)
                        continue
                    for key in result.keys():
                        # print(result[key].shape)
                        if key not in results:
                            results[key] = result[key]
                        else:
                            if 'Wn' in key:
                                continue
                            else:
                                results[key] = np.concatenate((results[key], result[key]))

        rad_full2 = np.zeros((results['S2_Rad'].shape[0], 3369), dtype=np.float32)
        count = 0
        for radiance in results['S2_Rad']:
            spec_iasi2hiras, wavenumber_iasi2hiras, plot_data_iasi2hiras = iasi2hiras(
                radiance, IASI_BAND_F1[iband], IASI_BAND_F2[iband], IASI_D_FREQUENCY[iband],
                GIIRS_BAND_F1[iband], GIIRS_BAND_F2[iband], GIIRS_D_FREQUENCY[iband],
                GIIRS_F_NYQUIST, GIIRS_RESAMPLE_MAXX[iband], GIIRS_FILTER_WIDTH[iband],
                apodization_ori=iasi_apod, )
            rad_full2[count, :] = spec_iasi2hiras[9:3378]
            count += 1
        results['S2_Rad_full'] = rad_full2
                        # print(results[key].shape)
        write_hdf5_and_compress(out_file, results)


if __name__ == '__main__':
    main()
