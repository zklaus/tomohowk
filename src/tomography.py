#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from datetime import timedelta
from functools import partial
import h5py
from importlib import import_module
import itertools
import logging
import time
from scipy import float32, floor, pi
import sys
from tools import parse_range


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="HDF5 file containing the quadrature data")
    parser.add_argument("-f", "--force",
                        help="Overwrite previous reconstructions (default: %(default)s)",
                        action="store_true")
    parser.add_argument("--Nq",
                        help="Number of grid points in q direction (default: %(default)u)",
                        type=int, default=11)
    parser.add_argument("--Np", help="Number of grid points in p direction (default: %(default)u)",
                        type=int, default=11)
    parser.add_argument("-b", "--beta",
                        help="Beta reconstruction parameter (default: %(default).2f)",
                        type=float, default=.2)
    parser.add_argument("-a", "--approximation-order",
                        help="Order of the approximation in the "
                        "series expansion for the error function (default: %(default)u)",
                        type=int, default=6)
    parser.add_argument("-m", "--method",
                        help="Select implementation (default: %(default)s)",
                        choices=["cuda", "multiprocessing", "serial"], default="multiprocessing")
    parser.add_argument("-s", "--scans",
                        help="Select scans to treat (default: %(default)s)",
                        type=parse_range, default="all")
    return parser.parse_args()


def setup_reconstructions_group(h5, args):
    Nq, Np = args.Nq, args.Np
    beta = float32(args.beta)
    order = args.approximation_order
    if "reconstructions" in h5.keys():
        if args.force:
            logging.info("Old reconstructions found. "
                         "Checking for dimensional compatibility.")
            old_group = h5["reconstructions"]
            if (Nq, Np) == old_group["Q"].shape[2:]:
                logging.info("Old reconstructions are dimensionally "
                             "compatible. Checking parameters.")
                old_beta = h5["reconstructions"].attrs["beta"]
                old_order = h5["reconstructions"].attrs["approximation_order"]
                if (beta != old_beta or order != old_order):
                    logging.warning("Old reconstructions with different "
                                    "parameters found (old: beta={:.2f}, order={}; "
                                    "new: beta={:.2f}, order={}). Deleting."
                                    "".format(old_beta, old_order, beta, order))
                    del h5["reconstructions"]
                else:
                    return (old_group["q_mean"], old_group["p_mean"],
                            old_group["Q"], old_group["P"], old_group["W"])
            else:
                logging.warning("Old reconstructions are dimensionally "
                                "incompatible. Deleting.")
                del h5["reconstructions"]
        else:
            logging.critical("Old reconstructions found. If you want to "
                             "overwrite them, use --force. Aborting.")
            sys.exit(1)
    reconstruction_group = h5.create_group("reconstructions")
    reconstruction_group.attrs.create("beta", beta, dtype=float32)
    reconstruction_group.attrs.create("approximation_order", order, dtype=int)
    no_scans = h5["standardized_quadratures"].shape[0]
    no_steps = h5["standardized_quadratures"].shape[1]
    q_ds = reconstruction_group.create_dataset("q_mean", (no_scans, no_steps,))
    p_ds = reconstruction_group.create_dataset("p_mean", (no_scans, no_steps,))
    Q_ds = reconstruction_group.create_dataset("Q", (no_scans, no_steps, Nq, Np),
                                               chunks=(1, no_steps, Nq, Np))
    P_ds = reconstruction_group.create_dataset("P", (no_scans, no_steps, Nq, Np),
                                               chunks=(1, no_steps, Nq, Np))
    W_ds = reconstruction_group.create_dataset("W", (no_scans, no_steps, Nq, Np),
                                               chunks=(1, no_steps, Nq, Np))
    return q_ds, p_ds, Q_ds, P_ds, W_ds


def reconstruct_all_wigners(args, Calculator):
    with h5py.File(args.filename, "r+") as h5:
        try:
            eta = h5["raw_quadratures"].attrs["eta"]
        except KeyError:
            logging.critical("The detection efficiency eta could not be found in the data file. "
                             "Please add it before continuing.")
            raise
        q_ds, p_ds, Q_ds, P_ds, W_ds = setup_reconstructions_group(h5, args)
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
            calculator = Calculator(eta, args.beta, L, angles, no_pulses,
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
                    estimate = int(elapsed/part)
                else:
                    estimate = 0
                sys.stderr.write("\r{0:7.2%} (Elapsed: {1}, ETA: {2})".format(part,
                                                                              timedelta(seconds=int(elapsed)),
                                                                              timedelta(seconds=estimate)))
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
