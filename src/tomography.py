#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from functools import partial
import h5py
import itertools
from multiprocessing import Pool
import scipy
from scipy import exp, sin, cos, sinh, cosh, sqrt, log, pi
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import sys


def gamma(eta):
    return (1.-eta)/(4.*eta)


def calc_h(ndata, beta, eta):
    g = gamma(eta)
    return 1.0/(sqrt(log(ndata)/(2.*beta+2.*g))) #puntual estimation class r=2


def estimate_position_from_quadratures(phix, N_phi=30, N_x=101):
    phi = phix[:,0]
    x = phix[:,1]
    phi_edges = scipy.linspace(0, 2.*scipy.pi, N_phi)
    phi_centers = (phi_edges[:-1]+phi_edges[1:])/2.
    phi_idx = scipy.digitize(phi, phi_edges)
    xs = [x[phi_idx==n+1] for n in range(len(phi_centers))]
    means = scipy.array([scipy.mean(x) for x in xs])
    stds = scipy.array([scipy.std(x) for x in xs])
    m = interp1d(phi_centers, means)
    return -m(pi), m(pi/2.), stds.max()


def K_ana(eta, h, Q, P, phix, order=5):
    g = gamma(eta)
    y = sqrt(g)/h
    phi = phix[:,0]
    X = phix[:,1]/sqrt(eta)
    Z = (scipy.einsum('ij,l->lij', Q, cos(phi))
         + scipy.einsum('ij,l->lij', P, sin(phi))
         - X[:,None,None])/h
    Zy = Z/y
    Zy2 = Zy**2
    n = scipy.arange(1, order+1)
    n2 = n**2
    f_denom = 1./(n2 + Zy2[:,:,:,None])
    v = Zy*sin(Z)*scipy.dot(f_denom, exp(-n2/4.)*n*sinh(n*y))
    cosZ = cos(Z)
    v += Zy2*(scipy.dot(f_denom, exp(-n2/4.))
              - cosZ*scipy.dot(f_denom, exp(-n2/4.)*cosh(n*y)))
    del f_denom
    v /= sqrt(pi)
    v += cosZ*(exp(y**2)-1./(2.*sqrt(pi)))+(1./(2.*sqrt(pi))-1.)
    v /= 4.*pi*g
    res = v.sum(axis=0)/phix.shape[0]
    return res


def refine_position_estimate(eta, h, phix, q_mean, p_mean, s_max):
    q_min = q_mean - s_max
    q_max = q_mean + s_max
    p_min = p_mean - s_max
    p_max = p_mean + s_max
    def mini(qp):
        return -K_ana(eta, h, scipy.array([[qp[0]]]), scipy.array([[qp[1]]]), phix, 3)
    res = minimize(mini, scipy.array([q_mean, p_mean]), bounds=[(q_min, q_max),(p_min, p_max)])
    return res.x


def build_mesh(q_mean, p_mean, s_max, Nq, Np):
    s_max *= 2.
    q_min = q_mean - s_max
    q_max = q_mean + s_max
    p_min = p_mean - s_max
    p_max = p_mean + s_max
    q = scipy.linspace(q_min, q_max, Nq)
    p = scipy.linspace(p_min, p_max, Np)
    Q, P = scipy.meshgrid(q, p)
    return q, p, Q, P


def reconstruct_wigner(eta, beta, phix, Nq, Np):
    L = len(phix)
    h = calc_h(L, beta, eta)
    q_mean, p_mean, s_max = estimate_position_from_quadratures(phix)
    q_mean, p_mean = refine_position_estimate(eta, h, phix, q_mean, p_mean, s_max)
    q, p, Q, P = build_mesh(q_mean, p_mean, s_max, Nq, Np)
    W = K_ana(eta, h, Q, P, phix)
    return Q, P, W


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
                        choices=["serial", "multiprocessing"], default="multiprocessing")
    return parser.parse_args()


def setup_reconstructions_group(h5, Nq, Np, force):
    if "Reconstructions" in h5.keys():
        if force:
            print "Old reconstructions found. Force active, deleting old reconstructions."
            del h5["Reconstructions"]
        else:
            print "Old reconstructions found. If you want to overwrite them, use --force. Aborting."
            sys.exit(1)
    reconstruction_group = h5.create_group("Reconstructions")
    Nsteps = h5["Quadratures"].shape[0]
    Q_ds = reconstruction_group.create_dataset("Q", (Nsteps, Nq, Np), chunks=(1, Nq, Np))
    P_ds = reconstruction_group.create_dataset("P", (Nsteps, Nq, Np), chunks=(1, Nq, Np))
    W_ds = reconstruction_group.create_dataset("W", (Nsteps, Nq, Np), chunks=(1, Nq, Np))
    return Q_ds, P_ds, W_ds


def reconstruct_all_wigners_serial(args):
    with h5py.File(args.filename, "r+") as h5:
        Q_ds, P_ds, W_ds = setup_reconstructions_group(h5, args.Nq, args.Np, args.force)
        R = partial(reconstruct_wigner, args.eta, args.beta, Nq=args.Nq, Np=args.Np)
        Nsteps = h5["Quadratures"].shape[0]
        for i, (Q, P, W) in enumerate(itertools.imap(R, h5["Quadratures"][:])):
            Q_ds[i,:,:] = Q
            P_ds[i,:,:] = P
            W_ds[i,:,:] = W
            sys.stderr.write("\r{0:%}".format(float(i)/Nsteps))
        sys.stderr.write("\n")


def reconstruct_all_wigners_multiprocessing(args):
    with h5py.File(args.filename, "r+") as h5:
        Q_ds, P_ds, W_ds = setup_reconstructions_group(h5, args.Nq, args.Np, args.force)
        R = partial(reconstruct_wigner, args.eta, args.beta, Nq=args.Nq, Np=args.Np)
        Nsteps = h5["Quadratures"].shape[0]
        pool = Pool(4)
        for i, (Q, P, W) in enumerate(pool.imap(R, h5["Quadratures"][:])):
            Q_ds[i,:,:] = Q
            P_ds[i,:,:] = P
            W_ds[i,:,:] = W
            sys.stderr.write("\r{0:%}".format(float(i)/Nsteps))
        sys.stderr.write("\n")


def main():
    args = parse_args()
    reconstruct = eval("reconstruct_all_wigners_"+args.method)
    print "Method chosen:", args.method, reconstruct
    reconstruct(args)


if __name__ == "__main__":
    main()
