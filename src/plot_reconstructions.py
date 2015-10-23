#!/usr/bin/python
# -*- coding: utf-8 -*-

import h5py
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
import sys


class QuadContainer(object):
    def __init__(self, quad):
        self.quad = quad

    def update_quad(self, quad):
        self.quad.remove()
        self.quad = quad


def main():
    filename = sys.argv[1]
    h5 = h5py.File(filename, "r")
    rg = h5["Reconstructions"]
    Qds = rg["Q"]
    Pds = rg["P"]
    Wds = rg["W"]
    Nsteps = Qds.shape[0]
    q_min = Qds[:,0,0].min()
    q_max = Qds[:,0,-1].max()
    p_min = Pds[:,0,0].min()
    p_max = Pds[:,-1,0].max()
    W_abs_max = abs(Wds[:]).max()
    W_min = -W_abs_max
    W_max = W_abs_max
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim(q_min, q_max)
    ax.set_ylim(p_min, p_max)
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    quad = QuadContainer(ax.pcolormesh(Qds[0], Pds[0], Wds[0],
                                       vmin=W_min, vmax=W_max,
                                       cmap="coolwarm",
                                       shading="flat"))
    cb = fig.colorbar(quad.quad)
    cb.set_label("Quasiprobability Density")
    title_string = "Wigner Function at step {}"
    title = ax.set_title(title_string.format(0))
    #ax.set_axis_bgcolor(cb.to_rgba(0.))
    def animate(i):
        title.set_text(title_string.format(i))
        quad.update_quad(ax.pcolormesh(Qds[i], Pds[i], Wds[i],
                                       vmin=W_min, vmax=W_max,
                                       cmap="coolwarm",
                                       shading="flat"))
        return quad.quad,
    ani = FuncAnimation(fig, animate, Nsteps, interval=100, repeat_delay=1000)
    ani.save('flat.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
    pyplot.show()


if __name__ == "__main__":
    main()
