#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import h5py
from lmfit import Model
import scipy
from numpy import float32
from scipy import arange, arccos, average, cos, pi, polyfit, polyval, sqrt, std
from scipy.signal import savgol_filter
from scipy.stats import kstest
import sys
from tools import parse_range, create_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Name of the HDF5 file")
    parser.add_argument("-f", "--force",
                        help="Overwrite previous reconstructions "
                        "(default: %(default)s)",
                        action="store_true")
    parser.add_argument("-s", "--scans",
                        help="Select scans to treat (default: %(default)s)",
                        type=parse_range, default="all")
    return parser.parse_args()


def setup_datasets(args, h5):
    shape = h5["raw_quadratures"].shape
    no_scans, no_steps, no_angles, no_pulses = shape
    ds_q = create_dataset(args, h5, "standardized_quadratures", shape)
    ds_a = create_dataset(args, h5, "angles", (no_scans, no_steps, no_angles))
    ds_phi_0 = create_dataset(args, h5,
                              "raw_phase_offsets", (no_scans, no_steps))
    ds_p_values = create_dataset(args, h5, "p_values", (no_scans, no_steps))
    return ds_phi_0, ds_a, ds_q, ds_p_values


def cos_model(x, offset, amplitude, omega, phi0):
    return offset + amplitude * cos(omega * x + phi0)


def clip(x, l, u):
    return l if x < l else u if x > u else x


def guess_initial_parameters(mean, phi0, omega):
    mean_max = mean.max()
    mean_min = mean.min()
    offset = (mean_max + mean_min)/2.
    amplitude = (mean_max - mean_min)/2.
    if phi0 is None or omega is None:
        y = mean-offset
        y2 = savgol_filter(y, 11, 1, mode="nearest")
        if phi0 is None:
            cos_phi0 = clip(y2[0]/amplitude, -1., 1.)
            if y2[1] > y2[0]:
                phi0 = -arccos(cos_phi0)
            else:
                phi0 = arccos(cos_phi0)
        if omega is None:
            zero_crossings = scipy.where(scipy.diff(scipy.sign(y2)))[0]
            omega = pi/scipy.average(scipy.diff(zero_crossings))
    return offset, amplitude, phi0, omega


def center_on_cos(raw_quadratures, phi0=None, omega=None, snap_omega=False):
    mean = scipy.average(raw_quadratures, axis=1)
    no_angles, no_pulses = raw_quadratures.shape
    model = Model(cos_model)
    offset, amplitude, phi0, omega = guess_initial_parameters(mean, phi0, omega)
    model.set_param_hint("offset", value=offset)
    model.set_param_hint("amplitude", min=0., value=amplitude)
    model.set_param_hint("phi0", value=phi0)
    model.set_param_hint("omega", min=0., value=omega)
    model.make_params(verbose=False)
    steps = scipy.arange(no_angles)
    res = model.fit(mean, x=steps, verbose=False)
    omega_param = res.params["omega"]
    if snap_omega:
        appx_omega = float(omega_param)
        no_pi_intervals = int(round(pi/appx_omega))
        omega = pi/no_pi_intervals
        omega_param.set(omega, vary=False)
        res.fit(mean, x=steps, verbose=False)
    d_value, p_value_ks = kstest(res.residual, 'norm')
    mean_fit = res.eval(x=steps)
    offset = mean-mean_fit
    aligned_quadratures = raw_quadratures - offset[:,None]
    centered_quadratures = aligned_quadratures - float(res.params["offset"])
    return (centered_quadratures,
            float(omega_param), float(res.params["phi0"]), p_value_ks)


def vacuum_correct(quadratures, vacuum_quadratures):
    gamma_prime = sqrt(2.)*average(std(vacuum_quadratures, axis=1))
    return quadratures/gamma_prime


def correct_intrastep_drift(quadratures):
    no_steps, no_pulses = quadratures.shape
    pulses = scipy.arange(no_pulses)
    for i in range(quadratures.shape[0]):
        quads = quadratures[i]
        mean = average(quads)
        model = polyfit(pulses, quads, 5)
        quadratures[i] = quads - polyval(model, pulses) + mean
    return quadratures


def standardize_quadratures(raw_quadratures, gamma_prime,
                            phi_0=None, omega=None):
    corrected_quadratures = correct_intrastep_drift(raw_quadratures)
    res = center_on_cos(corrected_quadratures, phi_0, omega)
    centered_quadratures, omega, phi_0, p_value = res
    return p_value, omega, phi_0, centered_quadratures/gamma_prime


def standardize_vacuum_quadratures(args, h5):
    vacuum_quadratures = h5["vacuum_quadratures"][:]
    corrected_vacuum = correct_intrastep_drift(vacuum_quadratures)
    create_dataset(args, h5,
                   "corrected_vacuum_quadratures", data=corrected_vacuum)
    mean = average(corrected_vacuum, axis=1)
    centered_vacuum = corrected_vacuum - mean[:, None]
    create_dataset(args, h5,
                   "centered_vacuum_quadratures", data=centered_vacuum)
    return average(std(centered_vacuum, axis=1))


def standardize_all_quadratures(args, gamma_prime, h5):
    ds_phi_0, ds_a, ds_q, ds_p_values = setup_datasets(args, h5)
    raw_ds = h5["raw_quadratures"]
    no_scans, no_steps, no_angles, no_pulses = raw_ds.shape
    if args.scans == "all":
        scans = range(no_scans)
    else:
        scans = args.scans
        no_scans = len(scans)
    omegas = scipy.empty((no_steps,), dtype=float32)
    for scan_no, i_scan in enumerate(scans, 1):
        sys.stderr.write("Starting scan {}, {} of {}:\n"
                         "".format(i_scan, scan_no, no_scans))
        for i_step in range(no_steps):
            raw_quadratures = raw_ds[i_scan, i_step, :, :]
            res = standardize_quadratures(raw_quadratures, gamma_prime)
            p_value, omega, phi_0, quadratures = res
            ds_p_values[i_scan, i_step] = p_value
            omegas[i_step] = omega
            ds_phi_0[i_scan, i_step] = phi_0
            ds_a[i_scan, i_step, :] = (arange(no_angles)*omega).astype(float32)
            ds_q[i_scan, i_step, :, :] = quadratures
            sys.stderr.write("\r{0:3.2%}".format(float(i_step)/no_steps))
        sys.stderr.write("\r100.00%\n")


def main():
    args = parse_args()
    with h5py.File(args.filename, "r+") as h5:
        gamma_prime = standardize_vacuum_quadratures(args, h5)
        standardize_all_quadratures(args, gamma_prime, h5)


if __name__ == "__main__":
    main()
