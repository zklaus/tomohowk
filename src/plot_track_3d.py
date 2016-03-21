#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import h5py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
import numpy
from tools import parse_range

import warnings
# This filters bugs in matplotlib.
warnings.filterwarnings('ignore', category=FutureWarning,
                        message="elementwise comparison failed; "
                                "returning scalar instead, but in the future "
                                "will perform elementwise comparison")
warnings.filterwarnings('ignore', category=FutureWarning,
                        message="comparison to `None` will result in an "
                                "elementwise object comparison in the future.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename",
                        help="HDF5 file containing the reconstructions")
    parser.add_argument("-s", "--scans", help="Select scans to treat. "
                        "All data will be averaged over all specified scans.",
                        type=parse_range, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with h5py.File(args.filename) as h5:
        if args.scans == "all":
            scans = range(h5["reconstructions/q_mean"].shape[0])
        else:
            scans = args.scans
        qe = numpy.average(h5["reconstructions/q_mean"][scans], axis=0)
        pe = numpy.average(h5["reconstructions/p_mean"][scans], axis=0)
    t = numpy.linspace(0, 1, qe.shape[0])
    fig = pyplot.figure()
    ax = fig.gca(projection="3d")
    ax.plot(qe, pe, t, '.-')
    ax.set_xlim3d(13, 15)
    ax.set_ylim3d(-2, 2)
    pyplot.show()


if __name__ == "__main__":
    main()
