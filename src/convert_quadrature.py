#!/usr/bin/python
# -*- coding: utf-8 -*-

import h5py
from matplotlib import pyplot
import numpy
import sys


def load_data_from_text_file(filename):
    interlaced_data = numpy.loadtxt(filename)
    phi = interlaced_data[:, 0::2]
    x = interlaced_data[:, 1::2]
    quadratures = numpy.dstack([phi.T, x.T])
    return quadratures


def save_data_to_hdf5_file(output_filename, data):
    h5 = h5py.File(output_filename)
    ds = h5.create_dataset("Quadratures", data=data)
    h5.close()


def main():
    input_filename = sys.argv[1]
    if len(sys.argv)>2:
        output_filename = sys.argv[2]
    else:
        output_filename = input_filename[:-3]+"h5"
    data = load_data_from_text_file(input_filename)
    save_data_to_hdf5_file(output_filename, data)

    
if __name__ == "__main__":
    main()
