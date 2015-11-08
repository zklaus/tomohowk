#!/usr/bin/python
# -*- coding: utf-8 -*-

from functools import partial
import h5py
from multiprocessing import Pool
import sys
from tomography_tools import *
from tomography_K_scipy import K


def reconstruct_wigner(eta, beta, phix, Nq, Np):
    L = len(phix)
    h = calc_h(L, beta, eta)
    q_mean, p_mean, s_max = estimate_position_from_quadratures(eta, phix)
    q, p, Q, P = build_mesh(q_mean, p_mean, s_max, Nq, Np)
    W = K(eta, h, Q, P, phix).sum(axis=0)/phix.shape[0]
    return Q, P, W.reshape(Q.shape)


def reconstruct_all_wigners(args):
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
