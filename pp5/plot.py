import itertools as it
import os
from typing import Union, List, Tuple, Callable, Optional, Iterable
from pathlib import Path
import logging

import numpy as np
from numpy import ndarray

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.style
import matplotlib.colors
from matplotlib.pyplot import Axes, Figure

import pp5

LOGGER = logging.getLogger(__name__)

PP5_MPL_STYLE = str(pp5.CFG_DIR.joinpath('pp5_plotstyle.rc.ini'))


def ramachandran(
        pdist: Union[ndarray, List[ndarray]],
        legend_label: Union[str, List[str]],
        title: str = None,
        ax: Axes = None,
        style: str = PP5_MPL_STYLE,
        outfile: Union[Path, str] = None,
        **colormesh_kw
) -> Optional[Tuple[Figure, Axes]]:
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
    :param outfile: Optional path to write output figure to.
    :return: Nothing if outfile was specified; otherwise (fig, ax) the figure
    and axes containing the plot.
    """
    if isinstance(pdist, ndarray):
        pdist = [pdist]
        assert isinstance(legend_label, str)
        legend_label = [legend_label]
    elif isinstance(pdist, (list, tuple)):
        assert isinstance(legend_label, (list, tuple))
        assert len(pdist) == len(legend_label)
    else:
        raise ValueError(f"Invalid pdist type: {type(pdist)}")
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

    if outfile is not None:
        savefig(fig, outfile, style=style)
        return None

    return fig, ax


def multi_heatmap(
        datas: Union[np.ndarray, List[np.ndarray]],
        row_labels: List[str] = None, col_labels: List[str] = None,
        titles: List[str] = None,
        fig_size=None, fig_rows=1,
        data_annotation_fn: Callable[[int, int, int], str] = None,
        style=PP5_MPL_STYLE, outfile: Union[Path, str] = None,
) -> Optional[Tuple[Figure, Iterable[Axes]]]:
    """
    Plots multiple 2D heatmaps horizontally next to each other while
    normalizing them to the same scale.
    :param datas: List of 2D arrays, or a single 3D array (the first
    dimension will be treated as a list).
    :param row_labels: Labels for the heatmap rows.
    :param col_labels: Labels for the heatmap columns.
    :param titles: Title for each axes.
    :param fig_size: Size of figure. If scalar it will be used as a base
    size and scaled by the number of rows and columns in the figure.
    Otherwise it should be a tuple of (width, height).
    :param fig_rows: How many rows of heatmaps to create in the figure.
    :param data_annotation_fn: An optional callable accepting three indices
    (i,j, k) into the given datas. It should return a string which will be
    used as an annotation (drawn inside the corresponding location in the
    heatmap).
    :param style: Style name or stylefile path.
    :param outfile: Optional path to write output figure to.
    :return: If figure was written to file, return nothing. Otherwise
    returns Tuple of figure, axes objects.
    """

    for d in datas:
        assert d.ndim == 2, "Invalid data shape"
        if row_labels:
            assert d.shape[0] == len(row_labels), "Inconsistent label number"
        if col_labels:
            assert d.shape[1] == len(col_labels), "Inconsistent label number"
    if titles:
        assert len(datas) == len(titles), "Inconsistent number of titles"
    assert fig_rows >= 1, "Invalid number of rows"

    vmin = min(np.nanmin(d) for d in datas)
    vmax = max(np.nanmax(d) for d in datas)
    norm = mpl.colors.Normalize(vmin, vmax)

    n = len(datas)
    fig_cols = int(np.ceil(n / fig_rows))

    if fig_size is None:
        fig_size = 10
    if isinstance(fig_size, (int, float)):
        fig_size = (fig_cols * fig_size, fig_rows * fig_size)
    elif isinstance(fig_size, (list, tuple)):
        assert len(fig_size) == 2, 'Invalid figsize'
        assert all(isinstance(x, (int, float)) for x in fig_size)
    else:
        raise ValueError(f'Invalid fig_size: {fig_size}')

    with mpl.style.context(style, after_reset=False):
        fig, ax = plt.subplots(fig_rows, fig_cols, figsize=fig_size)
        ax: np.ndarray[plt.Axes] = np.reshape(ax, -1)

        for i in range(n):
            data = datas[i]
            im = ax[i].imshow(data, interpolation=None, norm=norm)

            for edge, spine in ax[i].spines.items():
                spine.set_visible(False)

            ax[i].set_xticks(np.arange(data.shape[1]))
            ax[i].set_yticks(np.arange(data.shape[0]))
            if col_labels:
                ax[i].set_xticklabels(col_labels)
            if row_labels:
                ax[i].set_yticklabels(row_labels)
            if titles:
                ax[i].set_title(titles[i])

            ax[i].tick_params(top=True, bottom=False, labeltop=True,
                              labelbottom=False)
            plt.setp(ax[i].get_xticklabels(), rotation=45)
            ax[i].set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
            ax[i].set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
            ax[i].grid(which="minor", color="w", linestyle='-', linewidth=.5)
            ax[i].tick_params(which="minor", bottom=False, left=False)

            if data_annotation_fn is not None:
                rc_ind = it.product(range(data.shape[0]), range(data.shape[1]))
                for r, c in rc_ind:
                    annotation = str(data_annotation_fn(i, r, c))
                    ax[i].text(c, r, annotation, ha="center", va="center",
                               color="w")

        fig.colorbar(im, ax=ax, orientation='vertical', pad=0.05, shrink=0.7)

    if outfile is not None:
        savefig(fig, outfile, style=style)
        return None

    return fig, ax


def savefig(fig: plt.Figure, outfile: Union[Path, str],
            close=True, style=PP5_MPL_STYLE) -> Path:
    """
    Saves a figure to file.
    :param fig: Figure to save
    :param outfile: Path to save to. Suffix, if present, determines format.
    :param close: Whether to close the figure. When generating many plots
    and writing them it's crucial to close, otherwise matplotlib keeps
    referencing them and they consume memory.
    :param style: Style name or filename.
    :returns: Path of written file.
    """

    outfile = Path(outfile)
    fmt = outfile.suffix[1:]
    if not fmt:
        # Will be take from our default style
        fmt = None

    os.makedirs(str(outfile.parent), exist_ok=True)

    with mpl.style.context(style, after_reset=False):
        fig.savefig(str(outfile), format=fmt)

    LOGGER.info(f'Wrote {outfile}')
    if close:
        plt.close(fig)

    return outfile
