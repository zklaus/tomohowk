# -*- coding: utf-8 -*-

import argparse
import logging
from numpy import float32
import sys


def parse_range(string):
    if string=="all":
        return string
    range_strings = string.split(",")
    scans = []
    for range_string in range_strings:
        limits = range_string.split("-")
        if len(limits)>2:
            msg = "Invalid range specification: {}".format(range_string)
            raise argparse.ArgumentTypeError(msg)
        if len(limits)==1:
            try:
                scans.append(int(limits[0]))
            except ValueError:
                msg = "Invalid range specification: {}".format(range_string)
                raise argparse.ArgumentTypeError(msg)
        if len(limits)==2:
            try:
                lower = int(limits[0])
                upper = int(limits[1])
            except ValueError:
                msg = "Invalid range specification (not integers): {}".format(range_string)
                raise argparse.ArgumentTypeError(msg)
            if lower>=upper:
                msg = "Invalid range specification (not ascending): {}".format(range_string)
                raise argparse.ArgumentTypeError(msg)
            scans.extend(list(range(lower, upper+1)))
    return scans


def git_describe():
    try:
        from git import Repo
    except:
        logging.warning("Could not import GitPython. "
                        "Version information will not "
                        "be available in the data file")
        return "unknown".encode("ascii")
    repo = Repo(__file__, search_parent_directories=True)
    description = repo.git.describe(always=True, dirty=True)
    return description.encode("ascii")


def tag_hdf5_object_with_git_version(obj):
    obj.attrs.create("git_version", git_describe())


def create_dataset(args, h5, name, shape=None, data=None):
    if shape is None and data is None:
        raise ValueError("Neither shape nor data is given. One of them is necessary.")
    if name in h5.keys():
        if args.force:
            logging.warning("Old %s found. Force active, overwriting old data.", name)
            return h5[name]
        else:
            logging.warning("Old %s found. "
                            "If you want to overwrite them, use --force. Aborting.", name)
            sys.exit(1)
    if data is None:
        ds = h5.create_dataset(name, shape, compression="gzip", dtype=float32)
    else:
        ds = h5.create_dataset(name, compression="gzip", dtype=float32, data=data)
    tag_hdf5_object_with_git_version(ds)
    return ds
