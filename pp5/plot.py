import itertools as it
from typing import Union, List, Tuple, Callable

import numpy as np
from numpy import ndarray

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.style
import matplotlib.colors
from matplotlib.pyplot import Axes, Figure, cm

import pp5

PP5_MPL_STYLE = str(pp5.CFG_DIR.joinpath('pp5_plotstyle.rc.ini'))


def ramachandran(
        pdist: Union[ndarray, List[ndarray]],
        legend_label: Union[str, List[str]],
        title: str = None,
        ax: Axes = None,
        style: str = PP5_MPL_STYLE,
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


def multi_heatmap(
        datas: Union[np.ndarray, List[np.ndarray]],
        row_labels: List[str], col_labels: List[str],
        figsize=None, style=PP5_MPL_STYLE,
        data_annotation_fn: Callable[[int, int, int], str] = None,
):
    """
    Plots multiple 2D heatmaps horizontally next to each other while
    normalizing them to the same scale.
    :param datas: List of 2D arrays, or a single 3D array (the first
    dimension will be treated as a list).
    :param row_labels: Labels for the heatmap rows.
    :param col_labels: Labels for the heatmap columns.
    :param figsize: Size of figure. If scalar it will be the height, and the
    width will be N*figsize where N is the number of heatmaps. Otherwise it
    should be a tuple of (width, height).
    :param style: Style name or stylefile path.
    :param data_annotation_fn: An optional callable accepting three indices
    (i,j, k) into the given datas. It should return a string which will be
    used as an annotation (drawn inside the corresponding location in the
    heatmap).
    :return: Tuple of figure, axes and  colorbar objects.
    """

    for d in datas:
        assert d.ndim == 2, "Invalid data shape"
        assert d.shape[0] == len(row_labels), "Inconsistent number of labels"
        assert d.shape[1] == len(col_labels), "Inconsistent number of labels"

    vmin = min(np.min(d) for d in datas)
    vmax = max(np.max(d) for d in datas)
    norm = mpl.colors.Normalize(vmin, vmax)
    n = len(datas)

    if figsize is None:
        figsize = 10
    if isinstance(figsize, (int, float)):
        figsize = (n * figsize, figsize)
    elif isinstance(figsize, (list, tuple)):
        assert len(figsize) == 2, 'Invalid figsize'
        assert all(isinstance(x, (int, float)) for x in figsize)
    else:
        raise ValueError(f'Invalid figsize: {figsize}')

    with mpl.style.context(style, after_reset=False):
        fig, ax = plt.subplots(1, len(datas), figsize=figsize)
        ax: List[plt.Axes]

        for i in range(n):
            data = datas[i]
            im = ax[i].imshow(data, interpolation=None, norm=norm)

            for edge, spine in ax[i].spines.items():
                spine.set_visible(False)

            ax[i].set_xticks(np.arange(data.shape[1]))
            ax[i].set_yticks(np.arange(data.shape[0]))
            ax[i].set_xticklabels(col_labels)
            ax[i].set_yticklabels(row_labels)
            ax[i].tick_params(top=True, bottom=False, labeltop=True,
                              labelbottom=False)
            plt.setp(ax[i].get_xticklabels(), rotation=45)

            ax[i].set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
            ax[i].set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
            ax[i].grid(which="minor", color="w", linestyle='-', linewidth=1.)
            ax[i].tick_params(which="minor", bottom=False, left=False)

            if data_annotation_fn is not None:
                rc_ind = it.product(range(data.shape[0]), range(data.shape[1]))
                for r, c in rc_ind:
                    annotation = str(data_annotation_fn(i, r, c))
                    ax[i].text(c, r, annotation, ha="center", va="center",
                               color="w")

        cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.05,
                            shrink=0.7)

    return fig, ax, cbar
