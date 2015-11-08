#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from importlib import import_module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="HDF5 file containing the quadrature data")
    parser.add_argument("-f", "--force",
                        help="Overwrite previous reconstructions", action="store_true")
    parser.add_argument("--Nq", help="Number of grid points in q direction", type=int, default=11)
    parser.add_argument("--Np", help="Number of grid points in p direction", type=int, default=11)
    parser.add_argument("-b", "--beta", help="Beta reconstruction parameter", type=float, default=.2)
    parser.add_argument("-e", "--eta", help="Detection efficiency eta", type=float, default=.8)
    parser.add_argument("-m", "--method", help="Select implementation",
                        choices=["cuda", "multiprocessing", "serial"], default="multiprocessing")
    return parser.parse_args()


def main():
    args = parse_args()
    mod = import_module("tomography_"+args.method)
    reconstruct = mod.reconstruct_all_wigners
    print "Method chosen:", args.method, reconstruct
    reconstruct(args)


if __name__ == "__main__":
    main()
