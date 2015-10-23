#!/usr/bin/python
# -*- coding: utf-8 -*-

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
    q0, p0 = idx_to_q(i_mean[0]), idx_to_p(i_mean[1])
    s0 = 1./(W.max()*sqrt(2.*pi))
    theta0 = 0.
    def twoD_Gaussian(qp, a, b, c):
        q, p = qp
        normalization = sqrt(a*c-b**2)/(2.*pi)
        g = normalization*exp( -1./2.* (a*((q-q0)**2) + 2*b*(q-q0)*(p-p0) + c*((p-p0)**2)))
        return g.ravel()
    initial_guess = convert_params(s0, s0, theta0)
    (a, b, c), pcov = curve_fit(twoD_Gaussian, (Q, P), W.ravel(), p0=initial_guess)
    d = twoD_Gaussian((Q, P), a, b, c).reshape(W.shape)
    pyplot.subplot(2, 2, 1)
    pyplot.imshow(W, origin="lower")
    pyplot.colorbar()
    pyplot.subplot(2, 2, 2)
    pyplot.imshow(d, origin="lower")
    pyplot.colorbar()
    pyplot.subplot(2, 2, 3)
    diff = W-d
    m = abs(diff).max()
    pyplot.imshow(diff, cmap=cm.coolwarm, vmin=-m, vmax=m, origin="lower")
    pyplot.colorbar()
    pyplot.show()
    cov = scipy.array([[c, -b], [-b, a]])/(a*c-b**2)
    return scipy.array([q0, p0]), cov


def main()
    try:
        mean, cov = fit_gaussian_state(Q, P, W)
        rv = multivariate_normal(mean, cov)
        E = rv.pdf(scipy.dstack([Q, P]))
    except:
        pass
    #CS = pyplot.contour(Q, P, rv.pdf(scipy.dstack([Q,P])))
    #pyplot.clabel(CS)
    #pyplot.colorbar()
    #pyplot.show()
    #print W.shape, E.shape
    #mlab.surf(Q.T, P.T, W, warp_scale="auto", colormap="coolwarm", opacity=.3)
    #mlab.surf(Q.T, P.T, E, colormap="copper")
    #mlab.axes()
    #mlab.show()
    pyplot.pcolormesh(Q, P, W)
    pyplot.show()
    # pyplot.xlim(Q.min(), Q.max())
    # pyplot.ylim(P.min(), P.max())
    # #pyplot.imshow(W, interpolation="nearest", origin="lower")
    # pyplot.colorbar()
    # pyplot.show()
