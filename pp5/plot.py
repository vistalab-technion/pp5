import os
import re
import logging
import itertools as it
from typing import Dict, List, Tuple, Union, Callable, Iterable, Optional, Sequence
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.style
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections
from numpy import ndarray
from matplotlib.pyplot import Axes, Figure
from pandas import DataFrame
from itertools import count

import pp5

LOGGER = logging.getLogger(__name__)

PP5_MPL_STYLE = str(pp5.CFG_DIR.joinpath("pp5_plotstyle.rc.ini"))


def ramachandran(
    pdist: Union[ndarray, List[ndarray]],
    legend_label: Union[str, List[str]],
    title: str = None,
    grid_2pi: bool = False,
    ax: Axes = None,
    style: str = PP5_MPL_STYLE,
    figsize: float = None,
    outfile: Union[Path, str] = None,
    **colormesh_kw,
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
    :param grid_2pi: Whether the data grid is [0, 2pi) (True) or [-pi, pi) (False, default).
    :param ax: Axes to plot onto. If None, a new figure will be created.
    :param style: Matplotlib style to apply (name or filename).
    :param figsize: Size in inches of figure in both directions. None means use default.
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
            figsize = (figsize,) * 2 if figsize is not None else None
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig, ax = ax.figure, ax

        n_bins = p.shape[0]
        grid_range = (0.0, 360.0) if grid_2pi else (-180.0, 180.0)
        grid = np.linspace(*grid_range, endpoint=False, num=n_bins)
        colormesh_args = dict(shading="gouraud")
        colormesh_args.update(colormesh_kw)

        if "cmap" not in colormesh_args:
            cmaps = ["Reds", "Blues", "Greens", "Greys"]
        else:
            cmaps = [colormesh_args.pop("cmap")]

        if "alpha" not in colormesh_args:
            alphas = np.linspace(0.8, 0.1, num=len(pdist), endpoint=False)
        else:
            alphas = [colormesh_args.pop("alpha")]

        for i, p in enumerate(pdist):
            cmap = cmaps[i % len(cmaps)]
            alpha = alphas[i % len(alphas)]

            # Transpose because in a Ramachandram plot phi is the x-axis
            ax.pcolormesh(
                grid, grid, p.T, cmap=cmap, alpha=alpha, **colormesh_args,
            )

        legend_colors = ["darkred", "darkblue", "darkgreen", "grey"]
        legend_handles = []
        for i in range(len(pdist)):
            color = legend_colors[i % len(legend_colors)]
            label = legend_label[i]
            legend_handles.append(patches.Patch(color=color, label=label))

        ax.set_xlabel(r"$\varphi$")
        ax.set_ylabel(r"$\psi$")
        ticks = np.linspace(*grid_range, endpoint=True, num=7)
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
    row_labels: List[str] = None,
    col_labels: List[str] = None,
    titles: List[str] = None,
    fig_size=None,
    fig_rows=1,
    vmin: float = None,
    vmax: float = None,
    data_annotation_fn: Callable[[int, int, int], str] = None,
    style=PP5_MPL_STYLE,
    outfile: Union[Path, str] = None,
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
    :param vmin: Minimum value for scaling.
    :param vmax: Maximum value for scaling.
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

    vmin = vmin or min(np.nanmin(d) for d in datas)
    vmax = vmax or max(np.nanmax(d) for d in datas)
    norm = mpl.colors.Normalize(vmin, vmax)

    n = len(datas)
    fig_cols = int(np.ceil(n / fig_rows))

    if isinstance(fig_size, (int, float)):
        fig_size = (fig_cols * fig_size, fig_rows * fig_size)
    elif isinstance(fig_size, (list, tuple)):
        assert len(fig_size) == 2, "Invalid figsize"
        assert all(isinstance(x, (int, float)) for x in fig_size)
    elif fig_size is not None:
        # None will use default from style
        raise ValueError(f"Invalid fig_size: {fig_size}")

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

            ax[i].tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
            plt.setp(
                ax[i].get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor"
            )
            ax[i].set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
            ax[i].set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
            ax[i].grid(which="minor", color=(0.9,) * 3, linestyle="-", linewidth=0.5)
            ax[i].tick_params(which="minor", bottom=False, left=False)

            if data_annotation_fn is not None:
                rc_ind = it.product(range(data.shape[0]), range(data.shape[1]))
                for r, c in rc_ind:
                    annotation = str(data_annotation_fn(i, r, c))
                    ax[i].text(
                        c,
                        r,
                        annotation,
                        ha="center",
                        va="center",
                        color="w",
                        fontdict={"size": "xx-small"},
                    )

        fig.colorbar(im, ax=ax, orientation="vertical", pad=0.05, shrink=0.7)

    if outfile is not None:
        savefig(fig, outfile, style=style)
        return None

    return fig, ax


def multi_bar(
    data: Dict[str, np.ndarray],
    xticklabels: List[str] = None,
    xlabel: str = None,
    ylabel: str = None,
    cmap: Union[str, mpl.colors.Colormap] = None,
    total_width=0.9,
    single_width=1.0,
    ax: Axes = None,
    fig_size: Tuple[int, int] = None,
    style=PP5_MPL_STYLE,
    outfile: Union[Path, str] = None,
) -> Optional[Tuple[Figure, Axes]]:
    """
    Plots multiple bar-plots with grouping of each data point.
    :param data: A mapping from series name to an array of points. All
    arrays must have the same length.
    :param xticklabels: Label of the individual datapoints.
    :param xlabel: Label of x axis.
    :param ylabel: Label of y axis.
    :param cmap: Colormap to use. Can be either a string (the name of a
    colormap) or a colormap object.
    :param total_width: The width of a bar group. For example, 0.8 means that
    80% of the x-axis is covered by bars and 20% will be spaces
    between the bars.
    :param single_width: The relative width of a single bar within a group.
    For example, 1 means the bars will touch each other within a group, and
    values less than 1 will make these bars thinner.
    :param ax: Axes to plot into. If not provided a new figure will be created.
    :param fig_size: If axes was not provided, this specifies the size of
    figure to create (width, height).
    :param style: Style name or stylefile path.
    :param outfile: Optional path to write output figure to.
    :return: If figure was written to file, return nothing. Otherwise
    returns Tuple of figure, axes objects.
    """

    n_points = None
    assert len(data) > 0
    for d in data.values():
        assert d.ndim == 1, "data must contain 1D arrays"
        assert xticklabels is None or len(xticklabels) == len(d)
        if n_points is None:
            n_points = len(d)
        else:
            assert n_points == len(d), "Data has inconsistent length"

    with mpl.style.context(style, after_reset=False):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=fig_size)
        else:
            fig, ax = ax.figure, ax

        # Number of bars per group
        n_bars = len(data)

        # The width of a single bar
        bar_width = total_width / n_bars

        # List containing handles for the drawn bars, used for the legend
        bars = []

        # Get a list of colors so that we give each data series it's own
        # color when plotting the bars.
        if cmap is None:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        else:
            colors = plt.cm.get_cmap(cmap).colors

        # Iterate over all data
        for i, (name, values) in enumerate(data.items()):
            # The offset in x direction of that bar
            x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
            color = colors[i % len(colors)]

            for x, y in enumerate(values):
                barcontainer = ax.bar(
                    [x + x_offset], [y], width=bar_width * single_width, color=color
                )

            # Add a handle to the last drawn bar, which we'll need for the
            # legend
            bars.append(barcontainer[0])

        ax.set_xticks(range(n_points))
        if xticklabels:
            ax.set_xticklabels(xticklabels, rotation=45)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        ax.legend(bars, data.keys())

    if outfile is not None:
        savefig(fig, outfile, style=style)
        return None

    return fig, ax


def rainbow(
    data: List[Tuple[float, float, float, float]],
    group_labels: List[str],
    point_labels: List[str] = None,
    all_groups: List[str] = None,
    xlabel: str = None,
    ylabel: str = None,
    title=None,
    cmap: Union[str, mpl.colors.Colormap] = "gist_rainbow",
    alpha: float = 0.5,
    with_regression=False,
    error_ellipse=False,
    normalize=False,
    err_scale: float = 1.0,
    with_groups_legend=True,
    ax: Axes = None,
    fig_size: Tuple[int, int] = None,
    style=PP5_MPL_STYLE,
    outfile: Union[Path, str] = None,
) -> Optional[Tuple[Figure, Axes]]:
    """
    A rainbow plot represents teh relation between two variables where is also
    a group assigned to each data point. The plot shows each point as either
    (1) an ellipse centered at the location of the point (the width of which could
    represent e.g. confidence intervals); or (2) a point with horizontal and vertical
    error bars. The color of the ellipse/bars denotes the group the point belongs to.

    :param data: List of tuples (x,y,e1,e2) where (x,y) is the point location
        and (e1,e2) its horizontal and vertical errors (e.g. std).
        The e2 is optional, and if omitted will be set equal to e1.
        Note that all tuples must either include e2 or not.
    :param group_labels: The name of the group each point belongs to.
        Should be same length as data.
    :param point_labels: A string to print inside the ellipse of each point
        to display additional data about it. Should be same length as data.
    :param all_groups: Names of all possible groups. If None, then the set
        of group names in the group_labels will be used.
    :param xlabel: x-axis label.
    :param ylabel: y-axis label.
    :param title: Axis title.
    :param cmap: Colormap to use. Will be discretized evenly so that a
        different color is assigned to each group. By default assigns the
        colors of the rainbow!
    :param alpha: Transparency level of the ellipses. Should be in [0, 1].
    :param with_regression: If true, a simple linear regression line will be
        calculated and plotted on the data, with a correlation coefficient shown.
    :param error_ellipse: Whether to plot each point as a dot with errorbars (False)
        or as an Ellipse with error radii (True).
    :param normalize: If True, each data column will be normalized to dynamic range
        [0,1]. The errors will then be normalized by the same normalization factor.
    :param err_scale: Scale factor to apply to the errors to reduce
        overlap. Applied after normalization.
    :param ax: Axis to plot on.
    :param with_groups_legend: Whether to include a legend with one entry per group.
        It will be populated from the data in all_groups.
    :param fig_size: Size of figure to create if no axis given.
    :param style: Style name or style file path.
    :param outfile: Optional path to write output figure to.
    :return: If figure was written to file, return nothing. Otherwise
        returns Tuple of figure, axes objects.
    """
    assert len(data) == len(group_labels) > 0

    if all_groups is None:
        all_groups = {g: g for g in sorted(set(group_labels))}
    else:
        assert len(all_groups) >= len(set(group_labels))
        if not isinstance(all_groups, dict):
            all_groups = {g: g for g in all_groups}

    if point_labels is not None:
        assert len(point_labels) == len(data)

    # Add r2 equal to r1 if missing
    data = np.array(data, dtype=np.float32)  # (N, 3 or 4)
    assert data.shape[1] == 3 or data.shape[1] == 4
    if data.shape[1] == 3:
        data = np.concatenate([data, data[:, [-1]]], axis=1)

    xy_vals = data[:, :2]
    err_vals = data[:, 2:]

    # Scale x,y to [0,1]
    if normalize:
        # Remove minimal value in each axis separately, but scale both axes with same
        # factor in order to preserve aspect ratio.
        xy_vals -= np.min(xy_vals, axis=0)
        xymax = np.max(xy_vals) + 1e-12
        xy_vals /= xymax
        err_vals /= xymax

    # Get and scale radii to a fixed maximal radius
    err_vals *= err_scale

    # Create a dict containing a unique color per group
    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0, 1, len(all_groups), endpoint=True))
    colors[:, -1] = alpha  # Colors has RGBA in each row
    group_colors = {g: colors[i] for i, g in enumerate(all_groups)}

    with mpl.style.context(style, after_reset=False):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=fig_size)
        else:
            fig, ax = ax.figure, ax
        fig: Figure
        ax: Axes

        patch_list = []
        for i, group in enumerate(group_labels):
            x, y = xy_vals[i]
            xerr, yerr = err_vals[i]
            color = group_colors[group]
            if error_ellipse:
                ell = patches.Ellipse((x, y), 2 * xerr, 2 * yerr)
                ell.set_color(color)
                patch_list.append(ell)
            else:
                ax.errorbar(
                    x,
                    y,
                    yerr,
                    xerr,
                    ecolor=color,
                    fmt="o",
                    mfc=color,
                    mec="k",
                    mew=0.2,
                    ms=1.5,
                )
            if point_labels is not None:
                ax.text(x, y, point_labels[i], fontsize=6)

        if patch_list:
            pc = mpl.collections.PatchCollection(patch_list, match_original=True)
            ax.add_collection(pc)

        # Create custom legend handles with group names and colors
        legend_handles = []
        for i, group in enumerate(all_groups):
            color = group_colors[group]
            label = all_groups[group]
            legend_handles.append(patches.Patch(color=color, label=label))

        # Create regression line if needed
        if with_regression:
            # Add bias-trick column to x vals
            X = np.hstack([xy_vals[:, [0]], np.ones((len(xy_vals), 1))])  # N, 2
            y = xy_vals[:, [1]]  # (N, 1)
            try:
                xerr, *_ = np.linalg.lstsq(X, y, rcond=None)  # w is (2,)
                reg_y = np.dot(X, xerr)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                ss_res = np.sum((y - reg_y) ** 2)
                rsq = 1 - ss_res / ss_tot if ss_tot > 0 else np.inf
                reg_label = rf"$R^2={rsq:.2f}$"
                yerr = ax.plot(X[:, 0], reg_y, "k:", linewidth=1.0, label=reg_label)
                legend_handles.append(yerr[0])
            except np.linalg.LinAlgError as e:
                LOGGER.warning(
                    f"Failed to fit regression line for rainbow, "
                    f"{group_labels=}, {point_labels=}"
                )

        # Set axes properties
        xmin, xmax = np.min(xy_vals[:, 0]), np.max(xy_vals[:, 0])
        ymin, ymax = np.min(xy_vals[:, 1]), np.max(xy_vals[:, 1])
        xlim = 1 * np.array([-0.1, 0.1]) + [xmin, xmax]
        ylim = 1 * np.array([-0.1, 0.1]) + [ymin, ymax]
        ax.set_xlim(xlim), ax.set_ylim(ylim)
        xyticks = np.linspace(0, 1, num=11, endpoint=True)
        ax.set_xticks(xyticks), ax.set_yticks(xyticks)
        ax.grid()
        ax.set_xlabel(xlabel), ax.set_ylabel(ylabel), ax.set_title(title)
        if with_groups_legend:
            ax.legend(handles=legend_handles, loc="center left", fontsize="x-small")
        ax.set_aspect("equal")
        fig.tight_layout()

    if outfile is not None:
        savefig(fig, outfile, style=style)
        return None

    return fig, ax


def savefig(
    fig: plt.Figure, outfile: Union[Path, str], close=True, style=PP5_MPL_STYLE
) -> Path:
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

    LOGGER.info(f"Wrote {outfile}")
    if close:
        plt.close(fig)

    return outfile


def bar_plot(data: DataFrame,
             labels: str,
             values: str,
             hue: str,
             error_minus: str,
             error_plus: str,
             palette: str,
             inv_fun: Callable = lambda x: x,
             center: float = 0.,
             margins: Sequence[float] = ()):
    """
    Plots a bar plot from a data frame.
    :param labels: Name of the column containing the labels
    :param values: Name of the column containing the values
    :param hue: Name of the column containing the categories
    :param error_minus: Name of the column containing the negative error offset from
        value.
    :param error_plus: Name of the column containing the positive error offset from
        value.
    """
    ax = plt.gca()
    ax.cla()
    output_list = []

    hues = np.unique(data[hue].values)
    N = len(hues)

    w = 0.45 / N
    eps = w * 0.2
    step = -2 * w
    offset = -step * (N - 1) / 2

    cmap = plt.cm.get_cmap(palette)
    color_list = cmap(np.linspace(0, 1, len(hues)))

    all_keys = data.query(f'{hue}==@hues[0]')[labels].values
    vals = data.query(f'{hue}==@hues[0]')[values].values
    idx = np.argsort(vals)
    all_keys = list(all_keys[idx])

    plt.plot([center, center], [-1, len(all_keys) + 1], linewidth=0.5, color=[0, 0, 0], alpha=1,
             label='_nolegend_')
    for margin in margins:
        plt.plot(inv_fun(np.array([margin, margin])),
                 [-1, len(all_keys) + 1],
                 linewidth=0.5,
                 color=[1, 0, 0], alpha=1,
                 label='_nolegend_')
    for i in range(len(all_keys) + 1):
        plt.plot([-100, 100], [i - 0.5, i - 0.5], linewidth=0.5, color=[0, 0, 0],
                 alpha=0.5, label='_nolegend_')

    for n, h in enumerate(hues):
        keys = data.query(f'{hue}==@h')[labels].values
        vals = inv_fun(data.query(f'{hue}==@h')[values].values)
        # plt.plot(vals, keys, 'o', linewidth=1, markersize=3, color=color_list[n])
        for v, k in zip(vals, keys):
            i = all_keys.index(k)
            plt.plot(
                [v, v],
                [i - w + eps + step * n + offset, i + w - eps + step * n + offset],
                linewidth=2, markersize=3, color=color_list[n],
                label=h if i == 0 else '_nolegend_'
            )

    for i, h in enumerate(hues):
        keys = data.query(f'{hue}==@h')[labels].values
        vals = data.query(f'{hue}==@h')[values].values
        std_p = data.query(f'{hue}==@h')[error_plus].values
        std_m = data.query(f'{hue}==@h')[error_minus].values
        mins = inv_fun(vals - std_m)
        maxs = inv_fun(vals + std_p)

        for k, m, M in zip(keys, mins, maxs):
            n = all_keys.index(k)
            output_list.append(
                plt.Rectangle((m, n - w + step * i + offset), M - m, 2 * w,
                              facecolor=color_list[i],
                              edgecolor=None,
                              fill=True, alpha=0.25, label='_nolegend_'))

        for r in output_list:
            ax.add_artist(r)

    plt.yticks([i for i in range(len(all_keys))], all_keys)
