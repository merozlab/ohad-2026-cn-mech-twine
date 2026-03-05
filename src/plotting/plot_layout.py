"""Utilities for creating and styling plot layouts.
Handles subplot arrangements, labels, and common axis styles."""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
import matplotlib.image as mpimg
from matplotlib.figure import SubFigure

from brokenaxes import brokenaxes
import string
import numpy as np

from .figure_config import FIGURE_STYLES

# --- figure creation ---
# fig.add_axes() for custom axes placement


def make_grid_figure(
    nrows,
    ncols,
    layout=None,
    figsize=None,
    width_ratios=None,
    height_ratios=None,
    wspace=None,
    hspace=None,
):

    margins = None

    if layout is not None:
        style = FIGURE_STYLES[layout]

        if figsize is None:
            figsize = style["figsize"]
        if wspace is None:
            wspace = style["wspace"]
        if hspace is None:
            hspace = style["hspace"]

        margins = style.get("margins", None)

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(
        nrows,
        ncols,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        wspace=wspace,
        hspace=hspace,
    )

    if margins is not None:
        fig.subplots_adjust(**margins)

    return fig, gs


def make_named_grid(fig, gs, layout_map):
    """
    layout_map example:
    {
        "A": (0, 0),
        "B": (0, 1),
        "C": (1, slice(0, 2)),
    }
    """
    axes = {}

    for name, loc in layout_map.items():
        if isinstance(loc, tuple):
            r, c = loc
            axes[name] = fig.add_subplot(gs[r, c])
        elif isinstance(loc, slice):
            axes[name] = fig.add_subplot(gs[loc])
        elif isinstance(loc, tuple) and isinstance(loc[1], slice):
            r, c = loc
            axes[name] = fig.add_subplot(gs[r, c])
        else:
            raise ValueError(f"Invalid layout entry for {name}: {loc}")
    return axes

def make_named_subfig_grid(subfig, gs, layout_map):
    """
    Create named axes inside a subfigure using its GridSpec.

    layout_map example:
    {
        "A": (0, slice(0, 2)),
        "B": (1, slice(0, 2)),
        "C1": (2, 0),
        "C2": (2, 1),
    }
    """
    axes = {}

    for name, loc in layout_map.items():
        if isinstance(loc, tuple):
            r, c = loc
            axes[name] = subfig.add_subplot(gs[r, c])
        elif isinstance(loc, slice):
            axes[name] = subfig.add_subplot(gs[loc])
        elif isinstance(loc[0], slice):
            r, c = loc
            axes[name] = subfig.add_subplot(gs[r, c])
        else:
            raise ValueError(f"Invalid layout entry for {name}: {loc}")

    return axes

# --- axis creation / manipulation ---

def create_inset(
    parent_ax,
    width="40%",
    height="40%",
    loc="upper right",
    borderpad=0.25,
    bbox_to_anchor = None,
    bbox_transform = None,
):
    inset_fontsize = plt.rcParams['font.size']-2
    ax_inset = inset_axes(
        parent_ax,
        width=width,
        height=height,
        loc=loc,
        borderpad=borderpad,
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=bbox_transform,
    )
    ax_inset.tick_params(labelsize=inset_fontsize)
    return ax_inset

def create_broken_axis(
    *,
    fig=None,
    subplot_loc=None,
    parent_ax=None,
    xlims=None,
    ylims=None,
    width=None,
    height=None,
    loc="upper right",
    borderpad=1,
    hspace=0.15,
    wspace=0.15,
    height_ratios=None,
    width_ratios=None,
    d=0.005,
    supress_ticks=False,
    origin_ax=None,
    xtick_spacing=None,
    ytick_spacing=None,
):
    """
    Create a broken axis either as a main subplot (GridSpec)
    or as an inset inside an existing axis.

    Exactly ONE of:
        - subplot_loc
        - parent_ax
    must be provided.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Required if using subplot_loc.
    subplot_loc : GridSpec location
        e.g. gs[1, 2] or gs[0, :]
    parent_ax : matplotlib Axes
        Parent axis for inset-style broken axes.
    xlims : tuple of tuples, optional
        ((xmin1, xmax1), (xmin2, xmax2))
    ylims : tuple of tuples, optional
        ((ymin1, ymax1), (ymin2, ymax2))
    """

    # ---- validation ----
    if (subplot_loc is None) == (parent_ax is None):
        raise ValueError(
            "Exactly one of subplot_loc or parent_ax must be provided."
        )

    if xlims is None and ylims is None:
        raise ValueError(
            "At least one of xlims or ylims must be provided."
        )

    # ---- shared kwargs ----
    bax_kwargs = dict(
        d=d,
        hspace=hspace,
        wspace=wspace,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
    )

    if xlims is not None:
        bax_kwargs["xlims"] = xlims
    if ylims is not None:
        bax_kwargs["ylims"] = ylims

    # ---- GridSpec placement ----
    if subplot_loc is not None:
        if fig is None:
            raise ValueError("fig must be provided when using subplot_loc.")

        bax = brokenaxes(
            fig=fig,
            subplot_spec=subplot_loc,
            **bax_kwargs,
        )

    # ---- inset placement ----
    else:
        bax = brokenaxes(
            parent=parent_ax,
            width=width,
            height=height,
            loc=loc,
            borderpad=borderpad,
            **bax_kwargs,
        )
    if supress_ticks and origin_ax is not None:
        origin_ax.axis('off')
    if xtick_spacing is not None:
        for ax in bax.axs:
            ax.xaxis.set_major_locator(plt.MultipleLocator(xtick_spacing))
    if ytick_spacing is not None:
        for ax,yspace in zip(bax.axs, ytick_spacing):
            ax.yaxis.set_major_locator(plt.MultipleLocator(yspace))

    return bax

# --- post-layout adjustment ---

def adjust_axis_position(
    ax,
    scale_width=1.0,
    scale_height=1.0,
    dx=0.0,
    dy=0.0,
):
    """
    Adjust axis size and position after creation.
    """

    pos = ax.get_position()

    new_width = pos.width * scale_width
    new_height = pos.height * scale_height

    ax.set_position([
        pos.x0 + dx,
        pos.y0 + dy,
        new_width,
        new_height
    ])


def add_panel_labels(
    axes,
    labels=None,
    loc="upper left",
    offset=(0.02, 0.98),
    fontsize=None,
    weight="bold",
    color="black",
    outside=False,
    zorder=10,
):
    """
    Add panel labels to axes.

    Parameters
    ----------
    axes : dict[str, matplotlib.axes.Axes]
        Named axes dictionary (from make_named_grid).

    labels : list[str], optional
        Explicit labels (e.g., ["A", "B", "C"]).
        If None → auto-generate A, B, C...

    loc : str
        One of: "upper left", "upper right",
                "lower left", "lower right"

    offset : tuple(float, float)
        Relative position in axis coordinates.

    fontsize : int or None
        If None → auto-scale based on figure size.

    outside : bool
        If True → place slightly outside axis box.

    zorder : int
        Drawing order.
    """

    # ---- auto-generate labels ----
    if labels is None:
        labels = list(string.ascii_uppercase[:len(axes)])

    # ---- auto fontsize scaling ----
    if fontsize is None and len(np.shape(axes)) > 0:
        fig = next(iter(axes.values())).figure
        if isinstance(fig, SubFigure):
            fig = fig.figure  # parent Figure
        width, height = fig.get_size_inches()
        fontsize = max(12, int(2.5 * min(width, height)))
    elif fontsize is None:
        fontsize = 12

    # ---- alignment logic ----
    loc_map = {
        "upper left":  dict(ha="left",  va="top"),
        "upper right": dict(ha="right", va="top"),
        "lower left":  dict(ha="left",  va="bottom"),
        "lower right": dict(ha="right", va="bottom"),
    }

    if loc not in loc_map:
        raise ValueError(f"Invalid loc: {loc}")

    align = loc_map[loc]

    # ---- placement offset adjustment ----
    dx, dy = offset

    if outside:
        # push slightly outside axis
        dx = -0.05 if "left" in loc else 1.05
        dy = 1.05 if "upper" in loc else -0.05

    # ---- apply to axes ----
    if not isinstance(axes, dict):
        axes = {"_": axes}
    for label, ax in zip(labels, axes.values()):
        ax.text(
            dx,
            dy,
            label,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight=weight,
            color=color,
            zorder=zorder,
            **align
        )

def adjust_figure_margins(
    fig,
    left=0.075,
    right=0.95,
    top=0.95,
    bottom=0.075,
):
    fig.subplots_adjust(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
    )

def style_axis(
    ax,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    grid=None,
    xscale=None,
    yscale=None,
    xticks=None,
    yticks=None,
    tick_params=None,
    label_pad=None,
    x_formatter=None,
    y_formatter=None,
    nbin_ticks=4,
    background_color=None,
):
    """
    Apply common structural axis settings.
    Typography handled by stylesheet.
    """
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if xticks is not None: ax.set_xticks(xticks)
    else: ax.xaxis.set_major_locator(MaxNLocator(nbins=nbin_ticks, prune='upper'))
    if yticks is not None: ax.set_yticks(yticks)
    else: ax.yaxis.set_major_locator(MaxNLocator(nbins=nbin_ticks, prune='upper'))
    if tick_params is not None: ax.tick_params(**tick_params)

    if grid is not None: ax.grid(**grid)
    if xscale is not None: ax.set_xscale(xscale)
    if yscale is not None: ax.set_yscale(yscale)
    if label_pad is not None:
        if xlabel is not None: ax.xaxis.labelpad = label_pad
        if ylabel is not None: ax.yaxis.labelpad = label_pad
    if x_formatter is not None:
        if isinstance(x_formatter, str):
            x_formatter = FormatStrFormatter(x_formatter)
        ax.xaxis.set_major_formatter(x_formatter)
    if y_formatter is not None:
        if isinstance(y_formatter, str):
            y_formatter = FormatStrFormatter(y_formatter)
        ax.yaxis.set_major_formatter(y_formatter)
    if background_color is not None:
        ax.set_facecolor(background_color)

def tighten_figure(fig, rect=(0, 0, 1, 1)):
    fig.tight_layout(rect=rect)


def compute_image_height_ratio(image_path):
    """
    Returns height/width ratio of image.
    """
    img = mpimg.imread(image_path)
    h, w = img.shape[:2]
    return h / w

def get_bin_format(bins):
    bin_formats = []
    for i in range(len(bins)-1):
        bin_width = bins[i+1] - bins[i]
        if bin_width < 0.01:
            bin_formats.append(".3f")
        elif bin_width < 0.1:
            bin_formats.append(".2f")
        elif bin_width < 1.0:
            bin_formats.append(".1f")
        else:
            bin_formats.append(".0f")

    return bin_formats