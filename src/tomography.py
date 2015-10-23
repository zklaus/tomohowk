#!/usr/bin/python
# -*- coding: utf-8 -*-

from functools import partial
import h5py
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


def estimate_position(phix, N_phi=30, N_x=101):
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


def appx(x, y, order=5):
    two_x = 2.*x
    two_x2 = two_x**2
    s = sin(two_x*y)
    c = cos(two_x*y)
    n = scipy.arange(1, order+1)
    n2 = n**2
    nsh = n*sinh(n*y)
    ch2x = cosh(n*y)*two_x[:,None]
    f = two_x[:,None] - ch2x*c[:,None] + nsh*s[:,None]
    t = exp(-n2/4.)/(n2+two_x2[:,None])*f
    a = t.sum(axis=1)
    value = (1.-c)/two_x
    value += 2.*a
    value /= pi
    return value


def error_factor(h, gamma, X, order=5):
    gr = sqrt(gamma)
    Xp = X/(2.*gr)
    grp = gr/h
    value = appx(Xp, grp, order)
    return value


def K_ana(eta, h, qp, phi_l, x_l, order=5):
    L = len(phi_l)
    q, p = qp
    X = q*cos(phi_l) + p*sin(phi_l) - x_l/sqrt(eta)
    g = gamma(eta)
    term_1 = X/(8.*sqrt(pi)*sqrt(g)**3)*error_factor(h, g, X, order)
    term_2 = (exp(g/h**2)*cos(X/h)-1.)/(4.*pi*g)
    value = term_1 + term_2
    return value.sum()/L


def reconstruct_wigner(eta, beta, phix, N_q=11, N_p=11, pool=None):
    q_mean, p_mean, s_max = estimate_position(phix)
    q_min = q_mean - s_max
    q_max = q_mean + s_max
    p_min = p_mean - s_max
    p_max = p_mean + s_max
    L = len(phix)
    h = calc_h(L, beta, eta)
    def mini(qp):
        return -K_ana(eta, h, qp, phix[:,0], phix[:,1], 3)
    res = minimize(mini, scipy.array([q_mean, p_mean]), bounds=[(q_min, q_max),(p_min, p_max)])
    q_mean, p_mean = res.x
    s_max *= 5.
    q_min = q_mean - s_max
    q_max = q_mean + s_max
    p_min = p_mean - s_max
    p_max = p_mean + s_max
    q = scipy.linspace(q_min, q_max, N_q)
    p = scipy.linspace(p_min, p_max, N_p)
    Q, P = scipy.meshgrid(q, p)
    k = partial(K_ana, eta, h, phi_l=phix[:,0], x_l=phix[:,1], order=5)
    if pool != None:
        W = scipy.array(pool.map(k, zip(Q.flat,P.flat))).reshape(N_p, N_q)
    else:
        W = scipy.array(map(k, zip(Q.flat,P.flat))).reshape(N_p, N_q)
    return Q, P, W


def main():
    Nq = 20
    Np = 20
    filename = sys.argv[1]
    h5 = h5py.File(filename, "r+")
    if "Reconstructions" in h5.keys():
        print "Old reconstructions found. Aborting."
        sys.exit(1)
    quadrature_ds = h5["Quadratures"]
    Nsteps = quadrature_ds.shape[0]
    eta = 0.8
    beta = 0.2
    reconstruction_group = h5.create_group("Reconstructions")
    Q_ds = reconstruction_group.create_dataset("Q", (Nsteps, Nq, Np), chunks=(1, Nq, Np))
    P_ds = reconstruction_group.create_dataset("P", (Nsteps, Nq, Np), chunks=(1, Nq, Np))
    W_ds = reconstruction_group.create_dataset("W", (Nsteps, Nq, Np), chunks=(1, Nq, Np))
    pool = Pool(4)
    for i in range(Nsteps):
        print "Starting {}/{}".format(i, Nsteps)
        Q, P, W = reconstruct_wigner(eta, beta, quadrature_ds[i,:], Nq, Np, pool)
        Q_ds[i,:,:] = Q
        P_ds[i,:,:] = P
        W_ds[i,:,:] = W
        del Q
        del P
        del W
        h5.flush()
    h5.close()


if __name__ == "__main__":
    main()
