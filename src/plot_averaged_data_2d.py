#!/usr/bin/python
# -*- coding: utf-8 -*-


import argparse
import h5py
import scipy
from scipy import average, linspace, pi, var
from scipy.interpolate import interp1d
from tools import parse_range


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Name of the HDF5 file")
    parser.add_argument("-s", "--scans", help="Select scans to treat", type=parse_range, default="all")
    return parser.parse_args()


def main():
    args = parse_args()
    dt = 1.32e-14
    with h5py.File(args.filename) as h5:
        no_scans, no_steps, no_angles, no_pulses = h5["raw_quadratures"].shape
        if args.scans == "all":
            scans = range(no_scans)
        else:
            scans = args.scans
        t = linspace(0, no_steps*dt, no_steps)
        dphi = h5["corrected_angles"][0,0,1]-h5["corrected_angles"][0,0,0]
        phi = linspace(0.5*pi, 2.5*pi, 2.*pi/dphi)
        av_mean = scipy.zeros((len(phi), len(t)), dtype=scipy.float32)
        for i_scan in scans:
            for i_step in xrange(no_steps):
                print i_scan, i_step
                ip = interp1d(h5["corrected_angles"][i_scan,i_step],
                              var(h5["standardized_quadratures"][i_scan,i_step],
                                  axis=1))
                av_mean[:,i_step] += ip(phi)
        from matplotlib import pyplot
        pyplot.imshow(av_mean)
        pyplot.show()


if __name__=="__main__":
    main()
