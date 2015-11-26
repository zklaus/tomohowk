#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import h5py
from lmfit import Model
import logging
import scipy
from scipy import arange, average, cos, float32, pi, sqrt, std
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
    no_scans, no_steps, no_angles, no_pulses = shape
    if "standardized_quadratures" in h5.keys():
        if args.force:
            print ("Old standardized quadratures found. "
                   "Force active, deleting old data.")
            del h5["standardized_quadratures"]
        else:
            print ("Old standardized quadratures found. "
                   "If you want to overwrite them, use --force. Aborting.")
            sys.exit(1)
    ds_q = h5.create_dataset("standardized_quadratures", shape, compression="gzip", dtype="float32")
    if "angles" in h5.keys():
        if args.force:
            print ("Old angles found. "
                   "Force active, deleting old data.")
            del h5["angles"]
        else:
            print ("Old angles found. "
                   "If you want to overwrite them, use --force. Aborting.")
            sys.exit(1)
    ds_a = h5.create_dataset("angles", (no_scans, no_angles), compression="gzip", dtype="float32")
    return ds_a, ds_q


def cosmod(x, V0, A, omega, phi0):
    return V0 + A*cos(omega*x + phi0)


def center_on_cos(raw_quadratures):
    mean = scipy.average(raw_quadratures, axis=1)
    no_angles, no_pulses = raw_quadratures.shape
    model = Model(cosmod)
    model.set_param_hint("V0", value=scipy.average(mean))
    model.set_param_hint("A", value=(mean.max()-mean.min())/2.)
    model.set_param_hint("omega", value=2.*pi/(no_angles*.7))
    model.set_param_hint("phi0", value=0.)
    model.make_params()
    steps = scipy.arange(no_angles)
    res = model.fit(mean, x=steps)
    mean_fit = res.eval(x=steps)
    offset = mean-mean_fit
    aligned_quadratures = raw_quadratures - scipy.tile(offset, (no_pulses, 1)).T
    centered_quadratures = aligned_quadratures - float(res.params["V0"])
    return centered_quadratures, float(res.params["omega"]), float(res.params["phi0"])


def vacuum_correct(quadratures, vacuum_quadratures):
    gamma_prime = sqrt(2.)*average(std(vacuum_quadratures, axis=1))
    return quadratures/gamma_prime


def standardize_quadratures(raw_quadratures, vacuum_quadratures):
    centered_quadratures, omega, phi_0 = center_on_cos(raw_quadratures)
    quadratures = vacuum_correct(centered_quadratures, vacuum_quadratures)
    return omega, phi_0, quadratures


def standardize_all_quadratures(args, h5, ds_a, ds_q):
    vacuum_quadratures = h5["vacuum_quadratures"][:]
    raw_ds = h5["raw_quadratures"]
    no_scans, no_steps, no_angles, no_pulses = raw_ds.shape
    omegas = scipy.empty((no_steps,), dtype=float32)
    phi_0s = scipy.empty((no_steps,), dtype=float32)
    for i_scan in xrange(no_scans):
        sys.stderr.write("Starting scan {} of {}:\n".format(i_scan, no_scans))
        for i_step in xrange(no_steps):
            raw_quadratures = raw_ds[i_scan, i_step, :, :]
            omega, phi_0, quadratures = standardize_quadratures(raw_quadratures, vacuum_quadratures)
            omegas[i_step] = omega
            phi_0s[i_step] = phi_0
            ds_q[i_scan, i_step, :, :] = quadratures
            sys.stderr.write("\r{0:3.2%}".format(float(i_step)/no_steps))
        sys.stderr.write("\r100.00%\n")
        omega = average(omegas)
        angles = (arange(no_angles)*omega).astype(float32)
        ds_a[i_scan] = angles


def main():
    args = parse_args()
    h5 = h5py.File(args.filename, "r+")
    ds_a, ds_q = setup_dataset(args, h5)
    standardize_all_quadratures(args, h5, ds_a, ds_q)


if __name__ == "__main__":
    main()
