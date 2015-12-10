# -*- coding: utf-8 -*-

import argparse
import logging

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
            scans.extend(range(lower, upper+1))
    return scans

def git_describe():
    try:
        from git import Repo
    except:
        logging.warning("Could not import GitPython. "
                        "Version information will not "
                        "be available in the data file")
        return "unknown"
    repo = Repo(__file__)
    return repo.git.describe(always=True, dirty=True)

def tag_hdf5_object_with_git_version(obj):
    obj.attrs.create("git_version", git_describe())
