#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from argparse_tools import parse_range
from datetime import timedelta
from functools import partial
import h5py
from importlib import import_module
import itertools
import time
from tomography_tools import setup_reconstructions_group
from scipy import floor, pi
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="HDF5 file containing the quadrature data")
    parser.add_argument("-f", "--force",
                        help="Overwrite previous reconstructions", action="store_true")
    parser.add_argument("--Nq", help="Number of grid points in q direction", type=int, default=11)
    parser.add_argument("--Np", help="Number of grid points in p direction", type=int, default=11)
    parser.add_argument("-b", "--beta", help="Beta reconstruction parameter", type=float, default=.2)
    parser.add_argument("-e", "--eta", help="Detection efficiency eta", type=float, default=.8)
    parser.add_argument("-a", "--approximation-order", help="Order of the approximation in the "
                        "series expansion for the error function", type=int, default=6)
    parser.add_argument("-m", "--method", help="Select implementation",
                        choices=["cuda", "multiprocessing", "serial"], default="multiprocessing")
    parser.add_argument("-s", "--scans", help="Select scans to treat", type=parse_range, default="all")
    return parser.parse_args()


def reconstruct_all_wigners(args, Calculator):
    with h5py.File(args.filename, "r+") as h5:
        q_ds, p_ds, Q_ds, P_ds, W_ds = setup_reconstructions_group(h5, args.Nq, args.Np, args.force)
        quadrature_ds = h5["standardized_quadratures"]
        no_scans, no_steps, no_angles, no_pulses = quadrature_ds.shape
        if args.scans=="all":
            scans = xrange(no_scans)
        else:
            scans = args.scans
            no_scans = len(scans)
        for scan_no, i_scan in enumerate(scans, 1):
            sys.stderr.write("Starting scan {}, {} of {}:\n".format(i_scan, scan_no, no_scans))
            angles = h5["angles"][i_scan]
            max_angle = floor(angles.max()/pi)*pi
            angles = angles[angles<max_angle]
            no_angles = angles.shape[0]
            L = no_angles*no_pulses
            calculator = Calculator(args.eta, args.beta, L, angles, no_pulses,
                                    order=args.approximation_order)
            R = partial(calculator.reconstruct_wigner, Nq=args.Nq, Np=args.Np)
            start = time.time()
            for i, (q, p, Q, P, W) in enumerate(itertools.imap(R, quadrature_ds[i_scan,:,:no_angles,:])):
                q_ds[i_scan, i] = q
                p_ds[i_scan, i] = p
                Q_ds[i_scan, i,:,:] = Q
                P_ds[i_scan, i,:,:] = P
                W_ds[i_scan, i,:,:] = W
                elapsed = time.time()-start
                part = float(i)/no_steps
                if part>0:
                    eta = int(elapsed/part)
                else:
                    eta = 0
                sys.stderr.write("\r{0:7.2%} (Elapsed: {1}, ETA: {2})".format(part,
                                                                              timedelta(seconds=int(elapsed)),
                                                                              timedelta(seconds=eta)))
            sys.stderr.write("\n")


def main():
    args = parse_args()
    mod = import_module("tomography_"+args.method)
    calculator_name = args.method.capitalize()+"Calculator"
    Calculator = getattr(mod, calculator_name)
    print "Method chosen:", args.method, Calculator
    reconstruct_all_wigners(args, Calculator)


if __name__ == "__main__":
    main()
