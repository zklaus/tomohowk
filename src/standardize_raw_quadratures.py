#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import glob
import h5py
from lmfit import Model
import logging
import scipy
from scipy import average, cos, pi, sqrt, std
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Name of the HDF5 file")
    parser.add_argument("-f", "--force",
                        help="Overwrite previous reconstructions", action="store_true")
    parser.add_argument("-e", "--eta", help="Detection efficiency eta", type=float, default=.8)
    return parser.parse_args()


def setup_dataset(args, h5):
    shape = h5["raw_quadratures"].shape
    if "standardized_quadratures" in h5.keys():
        if args.force:
            print ("Old standardized quadratures found. "
                   "Force active, deleting old data.")
            del h5["standardized_quadratures"]
        else:
            print ("Old standardized quadratures found. "
                   "If you want to overwrite them, use --force. Aborting.")
            sys.exit(1)
    ds = h5.create_dataset("standardized_quadratures", shape, compression="gzip", dtype="float32")
    return ds


def cosmod(step, V0, A, omega, phi0):
    return V0 + A*cos(omega*step + phi0)


def center_on_cos(raw_quadratures):
    mean = scipy.average(raw_quadratures, axis=1)
    N_angles, N_pulses = raw_quadratures.shape
    model = Model(cosmod)
    model.set_param_hint("V0", value=scipy.average(mean))
    model.set_param_hint("A", value=(mean.max()-mean.min())/2.)
    model.set_param_hint("omega", value=2.*pi/(N_angles*.7))
    model.set_param_hint("phi0", value=0.)
    pars = model.make_params()
    step = scipy.arange(N_angles)
    res = model.fit(mean, step=step)
    l = scipy.arange(N_angles)
    mean_fit = res.eval(step=l)
    offset = mean-mean_fit
    quadratures = raw_quadratures - scipy.tile(offset, (N_pulses, 1)).T
    return quadratures, float(res.params["omega"]), float(res.params["phi0"])


def vacuum_correct(quadratures, gamma_prime):
    return quadratures/gamma_prime


def standardize_quadratures(raw_quadratures, gamma_prime):
    centered_quadratures, omega, phi0 = center_on_cos(raw_quadratures)
    quadratures = vacuum_correct(centered_quadratures, gamma_prime)
    return quadratures


def standardize_all_quadratures(args, h5, ds):
    vac_ds = h5["vacuum_quadratures"]
    gamma_prime = sqrt(2.)*average(std(vac_ds, axis=1))
    raw_ds = h5["raw_quadratures"]
    no_scans, no_steps = raw_ds.shape[:2]
    for i_scan in xrange(no_scans):
        sys.stderr.write("Starting scan {} of {}:\n".format(i_scan, no_scans))
        for i_step in xrange(no_steps):
            raw_quadratures = raw_ds[i_scan, i_step, :, :]
            ds[i_scan, i_step, :, :] = standardize_quadratures(raw_quadratures, gamma_prime)
            sys.stderr.write("\r{0:3.2%}".format(float(i_step)/no_steps))
        sys.stderr.write("\r100.00%\n")


def main():
    args = parse_args()
    h5 = h5py.File(args.filename, "r+")
    ds = setup_dataset(args, h5)
    standardize_all_quadratures(args, h5, ds)


if __name__ == "__main__":
    main()
