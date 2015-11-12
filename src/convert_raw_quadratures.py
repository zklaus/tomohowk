#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import glob
import h5py
import logging
import scipy
import sys


def read_tabular_data(filename, dtype=scipy.float32):
    table = []
    with open(filename, "U") as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line=="":
                continue
            data = stripped_line.replace(",", ".").replace(";", "").replace("x","\t").split("\t")
            table.append(map(dtype, data))
    return scipy.array(table, dtype=dtype)


def read_information(basename):
    base = read_tabular_data(basename)
    no_scans = read_tabular_data(basename+"-nscans", int)[0,0]
    no_pulses_per_angle, no_angles = read_tabular_data(basename+"-pulses", int)[0]
    try:
        timestep_size, no_timesteps = read_tabular_data(basename+"-step", int)[0]
    except IOError:
        timestep_size = scipy.nan
        no_timesteps = 1
    fns = glob.glob(basename+"-scan-*-step-*")
    scans = set()
    steps = {}
    for fn in fns:
        data = fn[len(basename):].split("-")
        scan = int(data[2])
        step = int(data[4])
        scans.add(scan)
        try:
            step_list = steps[scan]
        except KeyError:
            step_list = []
            steps[scan] = step_list
        step_list.append(step)
    for scan in scans:
        steps[scan] = sorted(steps[scan])
    all_steps = steps.values()
    reference_steps = all_steps[0]
    for s in all_steps[1:]:
        if s!=reference_steps:
            raise RuntimeError("Not all scans use the same number of steps. "
                               "This does not fit into a table.")
    if reference_steps!=range(no_timesteps):
        logging.warn("The actually used steps are not those described in the -step file.")
        no_timesteps = len(reference_steps)
        if reference_steps != scipy.arange(no_timesteps):
            raise RuntimeError("Timesteps are not consecutive. Aborting.")
    if scans != set(range(no_scans)):
        logging.warn("The actually used scans are not those described in the -nscans file.")
        no_scans = len(scans)
        if scans != set(range(no_scans)):
            raise RuntimeError("Scans are not consecutive. Aborting.")
    shape = (no_scans, no_timesteps, no_angles, no_pulses_per_angle)
    return shape


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("basename", help="Basename of quadrature data")
    parser.add_argument("-f", "--force",
                        help="Overwrite previous reconstructions", action="store_true")
    parser.add_argument("-e", "--eta", help="Detection efficiency eta", type=float, default=.8)
    return parser.parse_args()


def setup_h5_file(args, shape):
    if args.force:
        mode = "w"
    else:
        mode = "w-"
    h5 = h5py.File(args.basename+".h5", mode)
    ds = h5.create_dataset("raw_quadratures", shape, compression="gzip", dtype="float32")
    return h5, ds


def import_data(args, ds):
    (no_scans, no_timesteps, no_angles, no_pulses_per_angle) = ds.shape
    for scan in xrange(no_scans):
        print "Starting scan {} of {}:".format(scan, no_scans)
        for step in xrange(no_timesteps):
            fn = "{}-scan-{}-step-{}".format(args.basename, scan, step)
            data = read_tabular_data(fn)
            if data.shape == ds.shape[2:]:
                ds[scan, step, :, :] = data
            else:
                logging.error("Shape mismatch. Either number of angles or "
                              "pulses per angle are inconsistent. "
                              "Trying to pad with nans.")
                N_a, N_p = data.shape
                ds[scan, step, :N_a, :N_p] = data
                ds[scan, step, N_a:, :N_p] = scipy.nan
                ds[scan, step, :, N_p:] = scipy.nan
            sys.stderr.write("\r{0:3.2%}".format(float(step)/no_timesteps))
        sys.stderr.write("\r100.00%\n")


def main():
    args = parse_args()
    shape = read_information(args.basename)
    h5, ds = setup_h5_file(args, shape)
    import_data(args, ds)


if __name__ == "__main__":
    main()
