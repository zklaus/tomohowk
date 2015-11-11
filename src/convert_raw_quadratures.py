#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import glob
import h5py
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
    nscans = read_tabular_data(basename+"-nscans", int)[0,0]
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
        assert(s==reference_steps)
    assert(reference_steps==range(no_timesteps))
    no_scans = len(scans)
    assert(scans == set(range(no_scans)))
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
            ds[scan, step, :, :] = read_tabular_data(fn)
            sys.stderr.write("\r{0:3.2%}".format(float(step)/no_timesteps))
        sys.stderr.write("\r100.00%\n")


def main():
    args = parse_args()
    shape = read_information(args.basename)
    h5, ds = setup_h5_file(args, shape)
    import_data(args, ds)


if __name__ == "__main__":
    main()
