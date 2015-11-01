#!/usr/bin/python
# -*- coding: utf-8 -*-

from scipy import (arange, dot, outer,
                   exp, sin, cos, sinh, cosh, sqrt, log, pi)
from tomography_tools import gamma


def K(eta, h, Q, P, phix, order=5):
    g = gamma(eta)
    y = sqrt(g)/h
    phi = phix[:,0]
    X = phix[:,1]/sqrt(eta)
    Z = (outer(cos(phi), Q) + outer(sin(phi), P) - X[:,None])/h
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
