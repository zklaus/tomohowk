# -*- coding: utf-8 -*-

from tomography_tools import build_mesh
from tools import estimate_position_from_quadratures
from scipy import array, cos, einsum, exp, float64, linspace, pi, sin, sqrt
from scipy.integrate import trapz
from scipy.special import wofz, dawsn
from scipy.stats import norm, kstest

class NormalCalculator(object):
    def __init__(self, eta, beta, order):
        self.eta = eta
        self.beta = beta
        self.order = order

    def W(self, Q, P, phases, quadratures, unbias=True):
        scaled_quadratures = quadratures / sqrt(self.eta)
        mus = scaled_quadratures.mean(axis=1)
        if unbias:
            no_phases, no_pulses = quadratures.shape
            squared = (scaled_quadratures-mus[:, None])**2
            sigmas = sqrt(squared.sum(axis=1)/(no_pulses-1.5))
        else:
            sigmas = scaled_quadratures.std(axis=1)
        a = einsum("ij,k", Q, cos(phases))
        a += einsum("ij,k", P, sin(phases))
        a -= mus[None, None, :]
        a /= sigmas[None, None, :] * sqrt(2.)
        w = trapz((1. - 2.*a*dawsn(a))/sigmas[None, None, :]**2,
                  x=phases, axis=2)/(2.*pi*(phases[-1]-phases[0]))
        return w

    def reconstruct_wigner(self, angles_quadratures, Nq, Np):
        angles, quadratures = angles_quadratures
        q_mean, p_mean, s_max = estimate_position_from_quadratures(self.eta,
                                                                   angles,
                                                                   quadratures)
        q, p, Q, P = build_mesh(q_mean, p_mean, s_max, Nq, Np)
        W = self.W(Q, P, angles, quadratures)
        print("{:.4f}".format(trapz(trapz(W, x=Q[0,:], axis=0), x=P[:,0])))
        return q_mean, p_mean, Q, P, W
