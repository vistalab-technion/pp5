import itertools as it
from typing import Union, List, Tuple

import numpy as np
from numpy import ndarray

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.style
from matplotlib.pyplot import Axes, Figure, cm

import pp5

DEFAULT_STYLE = str(pp5.CFG_DIR.joinpath('pp5_plotstyle.rc.ini'))


def ramachandran(
        pdist: Union[ndarray, List[ndarray]],
        legend_label: Union[str, List[str]],
        title: str = None,
        ax: Axes = None,
        style: str = DEFAULT_STYLE,
        **colormesh_kw
) -> Tuple[Figure, Axes]:
    """
    Creates a Ramachandran plot from dihedral angle probabilities.
    :param pdist: A matrix of shape (M, M) containing the estimated
    probability of dihedral angles on a discrete grid from -pi to pi.
    Entry i,j of this matrix is assumed to correspond to the probability
    P(phi=phi_i, psi=psi_j). Can also be a sequence of such matrices, in which
    case they will be plotted one over the other.
    :param legend_label: Label for legend. Can be a list with the same
    number of elements as pdist.
    :param title: Optional title for axes.
    :param ax: Axes to plot onto. If None, a new figure will be created.
    :param style: Matplotlib style to apply (name or filename).
    :param colormesh_kw: Extra keyword args for matplotlib's pcolormesh()
    function.
    :return: (fig, ax) the figure and axes containing the plot.
    """
    if isinstance(pdist, ndarray):
        pdist = [pdist]
        assert isinstance(legend_label, str)
        legend_label = [legend_label]
    elif isinstance(pdist, (list, tuple)):
        assert isinstance(legend_label, (list, tuple))
        assert len(pdist) == len(legend_label)
    else:
        raise ValueError("Invalid pdist type")
    for p in pdist:
        assert p.ndim == 2
        assert p.shape[0] == p.shape[1]

    with mpl.style.context(style, after_reset=False):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig, ax = ax.figure, ax

        n_bins = p.shape[0]
        grid = np.linspace(-180., 180, endpoint=False, num=n_bins)
        colormesh_args = dict(shading='gouraud')
        colormesh_args.update(colormesh_kw)

        if 'cmap' not in colormesh_args:
            cmaps = ['Reds', 'Blues', 'Greens', 'Greys']
        else:
            cmaps = [colormesh_args.pop('cmap')]

        if 'alpha' not in colormesh_args:
            alphas = np.linspace(0.8, 0.1, num=len(pdist), endpoint=False)
        else:
            alphas = [colormesh_args.pop('alpha')]

        for i, p in enumerate(pdist):
            cmap = cmaps[i % len(cmaps)]
            alpha = alphas[i % len(alphas)]

            # Transpose because in a Ramachandram plot phi is the x-axis
            ax.pcolormesh(grid, grid, p.T, cmap=cmap, alpha=alpha,
                          **colormesh_kw)

        legend_colors = ['darkred', 'darkblue', 'darkgreen', 'grey']
        legend_handles = []
        for i in range(len(pdist)):
            color = legend_colors[i % len(legend_colors)]
            label = legend_label[i]
            legend_handles.append(mpl.patches.Patch(color=color, label=label))

        ax.set_xlabel(r'$\varphi$')
        ax.set_ylabel(r'$\psi$')
        ticks = np.linspace(-180, 180, endpoint=True, num=7)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.grid()
        ax.legend(handles=legend_handles)
        if title is not None:
            ax.set_title(title)

    return fig, ax
