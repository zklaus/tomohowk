#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from argparse_tools import parse_range
import h5py
from math import log10, ceil
from matplotlib import cm, colors, pyplot
from matplotlib.animation import FuncAnimation
import numpy

import warnings
#This filters a bug in matplotlib. Will be fixed in version 1.5.0.
warnings.filterwarnings('ignore', category=FutureWarning,
                        message="elementwise comparison failed")


def shifted_color_map(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }
    # regular index to compute the colors
    reg_index = numpy.linspace(start, stop, 257)
    # shifted index to match the data
    shift_index = numpy.hstack([
        numpy.linspace(0.0, midpoint, 128, endpoint=False),
        numpy.linspace(midpoint, 1.0, 129, endpoint=True)
    ])
    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    newcmap = colors.LinearSegmentedColormap(name, cdict)
    pyplot.register_cmap(cmap=newcmap)
    return newcmap


class QuadContainer(object):
    def __init__(self, quad):
        self.quad = quad
        self.quad.set_rasterized(True)

    def update_quad(self, quad):
        self.quad.remove()
        self.quad = quad
        self.quad.set_rasterized(True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infilename", help="HDF5 file containing the reconstructions")
    parser.add_argument("-o", "--output",
                        help="Output basename for movie. The file will be <name>.mp4")
    parser.add_argument("-v", "--visualization",
                        help="Visualization style. The raw option shows precisely the data,"
                        "polished gives a more pleasant rendering.", choices=["raw", "polished"],
                        default="raw")
    parser.add_argument("-t", "--track", help="Plot track of central position in phase space.",
                        action="store_true", default=False)
    parser.add_argument("-s", "--scans", help="Select scans to treat. "
                        "All data, including coordinates, will be averaged over all specified scans.",
                        type=parse_range, required=True)
    args = parser.parse_args()
    return args


def prepare_data(args):
    with h5py.File(args.infilename, "r") as h5:
        rg = h5["reconstructions"]
        no_scans = rg["Q"].shape[0]
        if args.scans=="all":
            scans = range(no_scans)
        else:
            scans = args.scans
        if len(scans)==1:
            q_mean = rg["q_mean"][scans[0]]
            p_mean = rg["p_mean"][scans[0]]
            Q_ds = rg["Q"][scans[0]]
            P_ds = rg["P"][scans[0]]
            W_ds = rg["W"][scans[0]]
        else:
            q_mean = numpy.average(rg["q_mean"][scans], axis=0)
            p_mean = numpy.average(rg["p_mean"][scans], axis=0)
            Q_ds = numpy.average(rg["Q"][scans], axis=0)
            P_ds = numpy.average(rg["P"][scans], axis=0)
            W_ds = numpy.average(rg["W"][scans], axis=0)
    return q_mean, p_mean, Q_ds, P_ds, W_ds


def main():
    args = parse_args()
    q_mean, p_mean, Q, P, W = prepare_data(args)
    no_steps = Q.shape[0]
    q_min = Q[:,0,0].min()
    q_max = Q[:,0,-1].max()
    p_min = P[:,0,0].min()
    p_max = P[:,-1,0].max()
    W_min = numpy.percentile(W[:], 0.1)
    W_max = numpy.percentile(W[:], 99.9)
    midpoint = 1 - W_max/(W_max + abs(W_min))
    cmap = shifted_color_map(cm.coolwarm, midpoint=midpoint, name="shifted")
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim(q_min, q_max)
    ax.set_ylim(p_min, p_max)
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    if args.visualization=="polished":
        shading="gouraud"
    else:
        shading="flat"
    quad = QuadContainer(ax.pcolormesh(Q[0], P[0], W[0],
                                       vmin=W_min, vmax=W_max,
                                       cmap=cmap,
                                       shading=shading))
    ax.set_aspect("equal")
    if args.track:
        ax.set_color_cycle([cm.copper(1.*i/(no_steps-1)) for i in range(no_steps-1)])
        for i in range(no_steps-1):
            ax.plot(q_mean[i:i+2],p_mean[i:i+2], alpha=.3)
    cb = fig.colorbar(quad.quad)
    cb.set_label("Quasiprobability Density")
    cb.solids.set_rasterized(True)
    if args.visualization=="polished":
        ax.set_axis_bgcolor(cb.to_rgba(0.))
    no_digits = int(ceil(log10(no_steps))+1)
    title_string = "Wigner Function at step {:{width}}/{:{width}}"
    title = ax.set_title(title_string.format(0, no_steps, width=no_digits))
    if args.output:
        fig.savefig(args.output+"_thumb.pdf")
    def animate(i):
        title.set_text(title_string.format(i, no_steps, width=no_digits))
        quad.update_quad(ax.pcolormesh(Q[i], P[i], W[i],
                                       vmin=W_min, vmax=W_max,
                                       cmap=cmap,
                                       shading=shading))
        return quad.quad,
    ani = FuncAnimation(fig, animate, no_steps, interval=100, repeat_delay=1000)
    if args.output:
        print "Saving movie to {}. This may take a couple of minutes.".format(args.output)
        ani.save(args.output+".mp4", fps=10, extra_args=['-vcodec', 'libx264'])
    pyplot.show()


if __name__ == "__main__":
    main()
