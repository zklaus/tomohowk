#!/usr/bin/python
# -*- coding: utf-8 -*-

import scipy
from scipy import sqrt, log, pi
from scipy.interpolate import interp1d
import sys


def gamma(eta):
    return (1.-eta)/(4.*eta)


def calc_h(ndata, beta, eta):
    g = gamma(eta)
    return 1.0/(sqrt(log(ndata)/(2.*beta+2.*g))) #puntual estimation class r=2


def estimate_position_from_quadratures(eta, phix, N_phi=30, N_x=101):
    phi = phix[:,0]
    x = phix[:,1]/sqrt(eta)
    phi_edges = scipy.linspace(0, 2.*scipy.pi, N_phi)
    phi_centers = (phi_edges[:-1]+phi_edges[1:])/2.
    phi_idx = scipy.digitize(phi, phi_edges)
    xs = [x[phi_idx==n+1] for n in range(len(phi_centers))]
    means = scipy.array([scipy.mean(x) for x in xs])
    stds = scipy.array([scipy.std(x) for x in xs])
    m = interp1d(phi_centers, means)
    return -m(pi), m(pi/2.), stds.max()


def build_mesh(q_mean, p_mean, s_max, Nq, Np):
    s_max *= 3.
    q_min = q_mean - s_max
    q_max = q_mean + s_max
    p_min = p_mean - s_max
    p_max = p_mean + s_max
    q = scipy.linspace(q_min, q_max, Nq).astype(scipy.float32)
    p = scipy.linspace(p_min, p_max, Np).astype(scipy.float32)
    Q, P = [f.astype(scipy.float32) for f in scipy.meshgrid(q, p)]
    return q, p, Q, P

