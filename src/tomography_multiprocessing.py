#!/usr/bin/python
# -*- coding: utf-8 -*-

from functools import partial
import multiprocessing
import scipy
from scipy import arange, cos, cosh, dot, exp, outer, pi, sin, sinh, sqrt
from scipy.interpolate import interp1d
from tomography_tools import build_mesh, calc_h, gamma


def estimate_position_from_quadratures(eta, angles, quadratures):
    X = quadratures/sqrt(eta)
    mean = scipy.average(X, axis=1)
    avg = interp1d(angles, mean)
    q_mean = -avg(pi)
    p_mean = avg(pi/2.)
    s_max = scipy.std(X, axis=1).max()
    return q_mean, p_mean, s_max


def K(eta, h, QP, phix, order=5):
    g = gamma(eta)
    y = sqrt(g)/h
    phi = phix[:,0]
    X = phix[:,1]/sqrt(eta)
    Z = (outer(cos(phi), QP[:,0]) + outer(sin(phi), QP[:,1]) - X[:,None])/h
    Zy = Z/y
    Zy2 = Zy**2
    n = arange(1, order+1)
    n2 = n**2
    f_denom = 1./(n2 + Zy2[:,:,None])
    v = Zy*sin(Z)*dot(f_denom, exp(-n2/4.)*n*sinh(n*y))
    cosZ = cos(Z)
    v += Zy2*(dot(f_denom, exp(-n2/4.))
              - cosZ*dot(f_denom, exp(-n2/4.)*cosh(n*y)))
    del f_denom
    v /= sqrt(pi)
    v += cosZ*(exp(y**2)-1./(2.*sqrt(pi)))+(1./(2.*sqrt(pi))-1.)
    v /= 4.*pi*g
    return v


class MultiprocessingCalculator(object):
    def __init__(self, eta, beta, L, no_angles, no_pulses, order=5, no_procs=4):
        self.order = order
        self.eta = eta
        self.gamma = gamma(eta)
        self.beta = beta
        self.L = L
        self.h = calc_h(L, beta, eta)
        self.rsq4pi = 1./sqrt(4.*pi)
        self.sqeta = sqrt(self.eta)
        self.four_pi_gamma = 4.*pi*self.gamma
        self.y = sqrt(self.gamma)/self.h
        n = scipy.arange(1, order+1, dtype=scipy.float32)
        self.n2 = n**2
        self.pre_s1 = exp(-self.n2/4.)
        self.pre_s2 = self.pre_s1*cosh(n*self.y)
        self.pre_s3 = self.pre_s1*n*sinh(n*self.y)
        self.no_procs = no_procs
        self.pool = multiprocessing.Pool(no_procs)

    def set_angles(self, angles):
        self.angles = angles
        self.cos_phi = cos(angles)
        self.sin_phi = sin(angles)

    def K(self, Q, P, quadratures):
        no_angles, no_pulses = quadratures.shape
        phix = scipy.vstack([scipy.tile(self.angles, (no_pulses,1)).T.ravel(), quadratures.ravel()]).T
        QP = scipy.vstack([Q, P]).T
        QP.shape[0]
        Kp = partial(K, self.eta, self.h, phix=phix)
        Wp = self.pool.map(Kp, scipy.array_split(QP, 2*self.no_procs, axis=0))
        W = scipy.hstack(Wp)
        return scipy.sum(W, axis=0)/self.L

    def reconstruct_wigner(self, angles_quadratures, Nq, Np):
        angles, quadratures = angles_quadratures
        self.set_angles(angles)
        q_mean, p_mean, s_max = estimate_position_from_quadratures(self.eta, angles, quadratures)
        q, p, Q, P = build_mesh(q_mean, p_mean, s_max, Nq, Np)
        W = self.K(Q.ravel(), P.ravel(), quadratures)
        return q_mean, p_mean, Q, P, W.reshape(Q.shape)
