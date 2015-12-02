#!/usr/bin/python
# -*- coding: utf-8 -*-

import scipy
from scipy import exp, sinh, cosh, cos, sin, sqrt, pi
from scipy.interpolate import interp1d
from tomography_tools import build_mesh, calc_h, gamma
from tomography_K_scipy import K

def estimate_position_from_quadratures(eta, angles, quadratures):
    X = quadratures/sqrt(eta)
    mean = scipy.average(X, axis=1)
    avg = interp1d(angles, mean)
    q_mean = avg(0.)
    p_mean = avg(pi/2.)
    s_max = scipy.std(X, axis=1).max()
    return q_mean, p_mean, s_max


class SerialCalculator(object):
    def __init__(self, eta, beta, L, angles, no_pulses, order=5):
        self.order = order
        self.angles = angles
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
        self.cos_phi = cos(angles)
        self.sin_phi = sin(angles)

    def K(self, Q, P, quadratures):
        no_angles, no_pulses = quadratures.shape
        phix = scipy.vstack([scipy.tile(self.angles, (no_pulses,1)).T.ravel(), quadratures.ravel()]).T
        W = K(self.eta, self.h, Q, P, phix)
        return scipy.sum(W, axis=0)/self.L

    def reconstruct_wigner(self, quadratures, Nq, Np):
        q_mean, p_mean, s_max = estimate_position_from_quadratures(self.eta, self.angles, quadratures)
        q, p, Q, P = build_mesh(q_mean, p_mean, s_max, Nq, Np)
        W = self.K(Q.ravel(), P.ravel(), quadratures)
        return q_mean, p_mean, Q, P, W.reshape(Q.shape)
