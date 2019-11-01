import harp
import os
from datetime import datetime
import argparse

import h5py


def main(dir_in, dir_out):
    filenames = os.listdir(dir_in)
    filenames.sort(reverse=True)
    for filename in filenames:
        infile = os.path.join(dir_in, filename)
        outfile = os.path.join(dir_out, filename) + '.hdf'
        if not os.path.isfile(outfile):
            iasi2hdf(infile, outfile)


def iasi2hdf(infile, outfile):
    s = datetime.now()
    print('<<< {}'.format(infile))
    product = harp.import_product(infile)
    harp.export_product(product, outfile, file_format='hdf5', hdf5_compression=5)
    print('>>> {}'.format(outfile))
    print(datetime.now() - s)


def wirte_iasi_hdf5(iasi, ofile):
    opath = os.path.dirname(ofile)
    if not os.path.isdir(opath):
        os.makedirs(opath)
    h5_file_w = h5py.File(ofile, 'w')
    h5_file_w.create_dataset('Lons', dtype='f4', data=iasi.Lons, compression='gzip', compression_opts=5, shuffle=True)
    h5_file_w.create_dataset('Lats', dtype='f4', data=iasi.Lats, compression='gzip', compression_opts=5, shuffle=True)
    h5_file_w.create_dataset('Times', dtype='f4', data=iasi.Time, compression='gzip', compression_opts=5, shuffle=True)
    h5_file_w.create_dataset('satAzimuth', dtype='f4', data=iasi.satAzimuth, compression='gzip', compression_opts=5,
                             shuffle=True)
    h5_file_w.create_dataset('satZenith', dtype='f4', data=iasi.satZenith, compression='gzip', compression_opts=5,
                             shuffle=True)
    h5_file_w.create_dataset('sunAzimuth', dtype='f4', data=iasi.sunAzimuth, compression='gzip', compression_opts=5,
                             shuffle=True)
    h5_file_w.create_dataset('sunZenith', dtype='f4', data=iasi.sunZenith, compression='gzip', compression_opts=5,
                             shuffle=True)
    h5_file_w.create_dataset('wavenumber', dtype='f4', data=iasi.wavenumber, compression='gzip', compression_opts=5,
                             shuffle=True)
    h5_file_w.create_dataset('radiance', dtype='f4', data=iasi.radiance, compression='gzip', compression_opts=5,
                             shuffle=True)
    h5_file_w.close()

    # def wirte_iasi_hdf5(iasi, ofile):
    #     opath = os.path.dirname(ofile)
    #     if not os.path.isdir(opath):
    #         os.makedirs(opath)
    #     h5_file_w = h5py.File(ofile, 'w')
    #     h5_file_w.create_dataset('Lons', dtype = 'f4', data = iasi.Lons, compression = 'gzip', compression_opts = 5, shuffle = True)
    #     h5_file_w.create_dataset('Lats', dtype = 'f4', data = iasi.Lats, compression = 'gzip', compression_opts = 5, shuffle = True)
    #     h5_file_w.create_dataset('Times', dtype = 'f4', data = iasi.Time, compression = 'gzip', compression_opts = 5, shuffle = True)
    #     h5_file_w.create_dataset('satAzimuth', dtype = 'f4', data = iasi.satAzimuth, compression = 'gzip', compression_opts = 5, shuffle = True)
    #     h5_file_w.create_dataset('satZenith', dtype = 'f4', data = iasi.satZenith, compression = 'gzip', compression_opts = 5, shuffle = True)
    #     h5_file_w.create_dataset('sunAzimuth', dtype = 'f4', data = iasi.sunAzimuth, compression = 'gzip', compression_opts = 5, shuffle = True)
    #     h5_file_w.create_dataset('sunZenith', dtype = 'f4', data = iasi.sunZenith, compression = 'gzip', compression_opts = 5, shuffle = True)
    #     h5_file_w.create_dataset('wavenumber', dtype = 'f4', data = iasi.wavenumber, compression = 'gzip', compression_opts = 5, shuffle = True)
    #     h5_file_w.create_dataset('radiance', dtype = 'f4', data = iasi.radiance, compression = 'gzip', compression_opts = 5, shuffle = True)
    #     h5_file_w.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in',  required=True)
    parser.add_argument('--dir_out', required=True)
    args = parser.parse_args()
    main(args.dir_in, args.dir_out)
