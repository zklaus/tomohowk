#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import h5py
from numpy import float32
import scipy
from scipy import arange, pi, polyfit, polyval, tile
import sys
from tools import parse_range, create_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Name of the HDF5 file")
    parser.add_argument("-f", "--force",
                        help="Overwrite previous corrected angles (default: %(default)s)",
                        action="store_true")
    parser.add_argument("-s", "--scans", help="Select scans to treat", type=parse_range, default="all")
    return parser.parse_args()


def load_phi_0(args):
    with h5py.File(args.filename) as h5:
        phi_0_ds = h5["raw_phase_offsets"]
        if args.scans == "all":
            scans = range(phi_0_ds.shape[0])
        else:
            scans = args.scans
        phi_0 = phi_0_ds[scans]
    return phi_0


def unfold(phi_0):
    pr = phi_0.ravel()/pi
    for k in range(4):
        for i in range(len(pr)-1):
            if pr[i+1]-pr[i]>.5:
                pr[i+1] -= 1.
            elif pr[i+1]-pr[i]<-.5:
                pr[i+1] += 1.
    return pr*pi


def calculate_slow_phi_0s(phi_0s):
    slow_phi_0s = scipy.empty_like(phi_0s)
    for i, phi_0 in enumerate(phi_0s):
        phi_0_unfolded = unfold(phi_0)
        x = arange(len(phi_0_unfolded))
        model = polyfit(x, phi_0_unfolded, 3)
        slow_phi_0s[i] = polyval(model, x)
    return slow_phi_0s


def correct_angles(args, phi_0s, slow_phi_0s):
    with h5py.File(args.filename) as h5:
        angles_ds = h5["angles"]
        no_scans, no_angles = angles_ds.shape
        no_scans, no_steps = slow_phi_0s.shape
        if args.scans == "all":
            scans = list(range(angles_ds.shape[0]))
        else:
            scans = args.scans
        standardized_phases_ds = create_dataset(args, h5, "standardized_phases",
                                               (no_scans, no_steps, no_angles))
        phase_offsets_ds = create_dataset(args, h5, "phase_offsets",
                                             (no_scans, no_steps))
        for i_scan in scans:
            angles = tile(angles_ds[i_scan,:], (no_steps,1))
            standardized_phases_ds[i_scan,:,:] = angles+slow_phi_0s[i_scan,:,None]
            phase_offsets_ds[i_scan,:] = phi_0s[i_scan,:]-slow_phi_0s[i_scan,:]


def main():
    args = parse_args()
    phi_0s = load_phi_0(args)
    slow_phi_0s = calculate_slow_phi_0s(phi_0s)
    correct_angles(args, phi_0s, slow_phi_0s)


if __name__=="__main__":
    main()
