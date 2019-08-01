#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/2/13
@Author  : AnNing
由 IASI 网站的订单号下载 IASI 数据
"""

from __future__ import print_function
import os
import requests
import re
import sys
from datetime import datetime


OUT_PATH = 'download'  # 下载路径
IASI_URI_FILE = '20160405.txt'  # 由订单号获取的数据下载链接
DOWNLOAD_THREADS = 3
# IASI 数据订单号
ORDER_NUMBER_LIST = [u'3914687424', u'3914687724', u'3914688174', u'3914688684', u'3914689104',
                     u'3914689574', u'3914690414', u'3914691384', u'3914692124', u'3914693094',
                     u'3914693844', u'3914694074', u'3914694084', u'3914694094', u'3914694104',
                     u'3914694114', u'3914694124', u'3914694134', u'3914694144', u'3914694154',
                     u'3914694164', u'3914694174', u'3914694224', u'3914695724', u'3914696754',
                     u'3914697434', u'3914699054', u'3914701014', u'3914702274', u'3914703694',
                     u'3914705364', u'3914705374', u'3914705384', u'3914705394', u'3914705404',
                     u'3914705414', u'3914705424', u'3914705434', u'3914705444', u'3914705454',
                     u'3914705464', u'3914705474', u'3914705494', u'3914705504', u'3914705514',
                     u'3914705524', u'3914705534', u'3914705544', u'3914705554', u'3914705564',
                     u'3914705874', u'3914706424', u'3914707194', u'3914708444', u'3914709484',
                     u'3914710024', u'3914710734', u'3914711374', u'3914711914', u'3914712294',
                     u'3914713134', u'3914714004', u'3914715254', u'3914716334', u'3914716604',
                     u'3914716614', u'3914716624', u'3914716634', u'3914716644', u'3914716654',
                     u'3914716664', u'3914716674', u'3914716684', u'3914716694', u'3914716704',
                     u'3914716714', u'3914716724', u'3914716734', u'3914717784', u'3914720924',
                     u'3914723034', u'3914724934', u'3914726024', u'3914727194', u'3914728384',
                     u'3914729484', u'3914729494', u'3914729504', u'3914729514', u'3914729524',
                     u'3914729534', u'3914729544', u'3914729554', u'3914729564', u'3914729574',
                     u'3914729584']


def get_url_list(order):
    url_list = list()
    http_uri = 'https://download.bou.class.noaa.gov/download/{}/001'.format(order)
    ftp_uri = 'ftp://ftp.bou.class.noaa.gov/{}/001'.format(order)
    session = requests.session()
    response = session.get(http_uri)
    if response.status_code == 200:
        text = response.text
        str_list = text.split()
        pattern = r'.*"\d{3}/(.*__\d{14})">'
        _host = ftp_uri + '/{}'
        for s in str_list:
            result = re.match(pattern, s)
            if result:
                host = _host.format(result.groups()[0])
                url_list.append(host)
    return url_list


def get_download_cmd(out_path, in_uri):
    cmd_t = "wget --tries=3 --no-check-certificate \
            --waitretry=3 -c --no-parent -nd -nH \
            -P {out_path} -i {in_uri}"
    return cmd_t.format(**{'out_path': out_path, 'in_uri': in_uri})


def main(file_name):
    uri_list = list()
    if not file_name:
        if not os.path.isfile(IASI_URI_FILE):
            for order in ORDER_NUMBER_LIST:
                uri_list.extend(get_url_list(order))
                print('URI count: {}'.format(len(uri_list)))
            with open(IASI_URI_FILE, 'w') as f:
                f.writelines([i + '\n' for i in uri_list])
        else:
            with open(IASI_URI_FILE, 'r') as f:
                uri_list = f.readlines()
                print('URI count: {}'.format(len(uri_list)))
        uri_list.sort()
    else:
        with open(IASI_URI_FILE, 'r') as f:
            uri_list = f.readlines()
            print('URI count: {}'.format(len(uri_list)))

    count = 0
    time_tmp = datetime.now()
    for uri in uri_list:
        count += 1
        print('Start downloading {} / {}'.format(count, len(uri_list)))
        cmd = get_download_cmd(OUT_PATH, uri)
        print(cmd)
        os.system(cmd)
        print(datetime.now() - time_tmp)
        time_tmp = datetime.now()

    # count = 0
    # pool = Pool(3)
    # for uri in uri_list:
    #     count += 1
    #     print('Start downloading {} / {}'.format(count, len(uri_list)))
    #     cmd = get_download_cmd(OUT_PATH, uri)
    #     pool.apply_async(os.system, (cmd,))
    # pool.close()
    # pool.join()


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        FILE_NAME = None
    else:
        FILE_NAME = args[0]
    main(FILE_NAME)
