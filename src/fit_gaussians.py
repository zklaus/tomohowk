#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import h5py
from itertools import imap
import scipy
from scipy import cos, exp, pi, sin, sqrt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import sys


def find_mean(W):
    Wn = W/W.sum()
    m, n = Wn.shape
    i_q = 0.
    i_p = 0.
    for i in xrange(m):
        for j in xrange(n):
            i_q += j*Wn[i,j]
            i_p += i*Wn[i,j]
    return (i_q, i_p)


def convert_params(sigma_q, sigma_p, theta):
    a = (cos(theta)**2)/(sigma_q**2) + (sin(theta)**2)/(sigma_p**2)
    b = -(sin(2.*theta))/(2.*sigma_q**2) + (sin(2.*theta))/(2.*sigma_p**2)
    c = (sin(theta)**2)/(sigma_q**2) + (cos(theta)**2)/(sigma_p**2)
    return a, b, c


def fit_gaussian_state(Q, P, W):
    q = Q[0,:]
    p = P[:,0]
    m, n = W.shape
    idx_to_q = interp1d(scipy.arange(n), q)
    idx_to_p = interp1d(scipy.arange(m), p)
    i_mean = find_mean(W)
    try:
        q0, p0 = idx_to_q(i_mean[0]), idx_to_p(i_mean[1])
        s0 = 1./(W.max()*sqrt(2.*pi))
        theta0 = 0.
        def twoD_Gaussian(qp, a, b, c):
            q, p = qp
            det = a*c-b**2
            if det<0:
                raise RuntimeError
            normalization = sqrt(det)/(2.*pi)
            g = normalization*exp( -1./2.* (a*((q-q0)**2) + 2*b*(q-q0)*(p-p0) + c*((p-p0)**2)))
            return g.ravel()
        initial_guess = convert_params(s0, s0, theta0)
        (a, b, c), pcov = curve_fit(twoD_Gaussian, (Q, P), W.ravel(), p0=initial_guess)
        cov = scipy.array([[c, -b], [-b, a]])/(a*c-b**2)
        dq = cov[0,0]
        cqp = cov[0,1]
        dp = cov[1,1]
    except:
        q0 = scipy.nan
        p0 = scipy.nan
        dq = scipy.nan
        cqp = scipy.nan
        dp = scipy.nan
    return scipy.array([q0, p0, dq, cqp, dp])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infilename", help="HDF5 file containing reconstruction data")
    parser.add_argument("-f", "--force",
                        help="Overwrite previous gaussian fits", action="store_true")
    args = parser.parse_args()
    return args


def load_reconstructions(h5):
    rg = h5["reconstructions"]
    Q_ds = rg["Q"]
    P_ds = rg["P"]
    W_ds = rg["W"]
    return Q_ds, P_ds, W_ds


def setup_gaussian_state_ds(h5, Nsteps, force):
    if "gaussians" in h5.keys():
        if force:
            print "Old gaussian fits found. Force active, deleting old fits."
            del h5["gaussians"]
        else:
            print "Old gaussian fits found. If you want to overwrite them, use --force. Aborting."
            sys.exit(1)
    G_ds = h5.create_dataset("gaussians", (Nsteps, 5))
    return G_ds


def main():
    args = parse_args()
    h5 = h5py.File(args.infilename, "r+")
    Q_ds, P_ds, W_ds = load_reconstructions(h5)
    Nsteps = W_ds.shape[0]
    G_ds = setup_gaussian_state_ds(h5, Nsteps, args.force)
    for i, state in enumerate(imap(fit_gaussian_state, Q_ds, P_ds, W_ds)):
        G_ds[i] = state
        sys.stderr.write('\rdone {0:.2%}'.format(float(i)/Nsteps))
    print


if __name__ == "__main__":
    main()
