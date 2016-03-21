#!/usr/bin/python
# -*- coding: utf-8 -*-

from functools import partial
from math import ceil
import multiprocessing
from numpy import float32
import scipy
from scipy import arange, cos, cosh, dot, exp, outer, pi, sin, sinh, sqrt
from scipy.interpolate import interp1d
from tomography_tools import build_mesh, calc_h, gamma


def estimate_position_from_quadratures(eta, angles, quadratures):
    scaled_quadratures = quadratures/sqrt(eta)
    mean = scipy.average(scaled_quadratures, axis=1)
    avg = interp1d(angles, mean)
    # find the first included multiple of pi
    q_angle_i = ceil(angles[0]/pi)
    q_angle = q_angle_i*pi
    q_sign = 1 if q_angle_i % 2 == 0 else -1
    if q_angle-angles[0] < pi/2.:
        p_angle = q_angle + pi/2.
        p_sign = q_sign
    else:
        p_angle = q_angle - pi/2.
        p_sign = -q_sign
    q_mean = q_sign*avg(q_angle)
    p_mean = p_sign*avg(p_angle)
    s_max = scipy.std(scaled_quadratures, axis=1).max()
    return q_mean, p_mean, s_max


def K(eta, g, h, y, n2, pre_s1, pre_s2, pre_s3, qp, phix):
    phi = phix[:, 0]
    scaled_quadratures = phix[:, 1]/sqrt(eta)
    z = (outer(cos(phi), qp[:, 0]) + outer(sin(phi), qp[:, 1]) - scaled_quadratures[:, None]) / h
    zy = z/y
    zy2 = zy**2
    f_denom = 1./(n2 + zy2[:, :, None])
    v = zy*sin(z)*dot(f_denom, pre_s3)
    cos_z = cos(z)
    v += zy2*(dot(f_denom, pre_s1) - cos_z*dot(f_denom, pre_s2))
    del f_denom
    v /= sqrt(pi)
    v += cos_z*(exp(y**2)-1./(2.*sqrt(pi)))+(1./(2.*sqrt(pi))-1.)
    v /= 4.*pi*g
    return v


class MultiprocessingCalculator(object):
    def __init__(self, eta, beta, order=5, no_procs=4):
        self.order = order
        self.eta = eta
        self.gamma = gamma(eta)
        self.beta = beta
        self.rsq4pi = 1./sqrt(4.*pi)
        self.sqeta = sqrt(self.eta)
        self.four_pi_gamma = 4.*pi*self.gamma
        self.no_procs = no_procs
        self.pool = multiprocessing.Pool(no_procs)

    def K(self, Q, P, angles, quadratures):
        no_angles, no_pulses = quadratures.shape
        L = no_angles*no_pulses
        h = calc_h(L, self.beta, self.eta)
        y = sqrt(self.gamma)/h
        n = arange(1, self.order+1, dtype=float32)
        n2 = n**2
        pre_s1 = exp(-n2/4.)
        pre_s2 = pre_s1*cosh(n*y)
        pre_s3 = pre_s1*n*sinh(n*y)
        phix = scipy.vstack([scipy.tile(angles, (no_pulses, 1)).T.ravel(), quadratures.ravel()]).T
        QP = scipy.vstack([Q, P]).T
        Kp = partial(K, self.eta, self.gamma, h, y,
                     n2, pre_s1, pre_s2, pre_s3, QP)
        W = scipy.zeros_like(Q, dtype=float32)
        for Wp in self.pool.imap(Kp, scipy.array_split(phix, 2*self.no_procs, axis=0), chunksize=1):
            W += scipy.sum(Wp, axis=0)
        return W/L

    def reconstruct_wigner(self, angles_quadratures, Nq, Np):
        angles, quadratures = angles_quadratures
        q_mean, p_mean, s_max = estimate_position_from_quadratures(self.eta, angles, quadratures)
        q, p, Q, P = build_mesh(q_mean, p_mean, s_max, Nq, Np)
        W = self.K(Q.ravel(), P.ravel(), angles, quadratures)
        return q_mean, p_mean, Q, P, W.reshape(Q.shape)
