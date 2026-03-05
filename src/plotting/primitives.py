"""Basic plotting primitives for error bars, distributions, and images.
Knows about single axes only."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image

from src.plotting.plot_layout import style_axis

CMAP = plt.get_cmap('coolwarm')
# CMAP = plt.get_cmap('seismic')


def plot_errorbar(
    ax,
    x,
    y,
    yerr=None,
    xerr=None,
    fmt="o",
    label=None,
    markersize=5,
    capsize=3,
    color='black'
):
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        xerr=xerr,
        fmt=fmt,
        markersize=markersize,
        capsize=capsize,
        label=label,
        color=color
    )


def plot_distribution(ax, data, nbins=20, bins=None, 
                    xlabel=None, ylabel=None, title=None, 
                    density=True, bar_space=0.1, alpha = 0.75, 
                    show_mean = False, mean_legend = None, 
                    color = CMAP(0.1), mean_color = CMAP(0.9),
                    total_count=None, nbin_ticks=3, 
                    xticks = None, yticks = None,
                    ):
    """
    Plot a distribution on the given axis.

    Parameters (type):
    ax (matplotlib.axes.Axes): The axis to plot on.
    data (list or numpy.ndarray): The data to plot.
    nbins (int): Number of bins for the histogram.
    bins (int or list): Number of bins or bin edges.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    title (str): Title for the subplot.
    mean_color (color): Color for the mean line.
    fs (int): Font size for labels and title.
    density (bool): Whether to normalize the histogram.
    mean_legend (str): Legend label for the mean line.
    total_count (int): If provided, normalize counts by this total (e.g., total events).
    """
    data = [v for v in data if np.isfinite(v)]
    if bins is None:
        bins = nbins

    counts, bins = np.histogram(data, bins=bins)

    if density:
        if total_count is not None and total_count > 0:
            normalized_counts = counts / total_count
        else:
            normalized_counts = counts / counts.sum() if counts.sum() > 0 else counts
    else:
        normalized_counts = counts

    bin_width = (bins[1] - bins[0]) * (1 - bar_space)
    ax.bar(bins[:-1], normalized_counts, width=bin_width, align='edge', color=color, alpha=alpha)
    if len(normalized_counts) > 0:
        ax.set_ylim([0, max(normalized_counts)*1.2])

    if show_mean:
        mean_value = np.nanmean(data)
        ax.axvline(mean_value, color=mean_color, linestyle='solid', label=mean_legend)
        if mean_legend is not None: ax.legend()
    style_axis(ax, 
                xlabel=xlabel, 
                ylabel=ylabel,
                nbin_ticks=nbin_ticks,
                xticks=xticks if xticks is not None else None,
                yticks=yticks if yticks is not None else None,
               )

    if title is not None: ax.set_title(title)
    return ax


def plot_image(ax, file_path, aspect='equal',
               rotate=0.0, reflect=None,
               remove_axes=True, extent=(0, 1, 0, 1), preserve_aspect=True,
               shift_x=0.0, shift_y=0.0, zorder=0, alpha=1.0):  # Add alpha parameter

    """
    Plot an image from a file path on the given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on.
    file_path : str
        Path to the image file.
    aspect : str or float, optional
        Aspect ratio for the image (default is 'equal').
    rotate : float, optional
        Rotation angle for the image in degrees (default is 0).
    reflect : 'x' or 'y', optional
        Reflect the image along the specified axis.
    remove_axes : bool, optional
        Whether to remove the axes (default is True).
    extent: tuple, optional
        The bounding box in data coordinates that the image will fill.
    preserve_aspect: bool, optional
        Whether to preserve the aspect ratio of the image.
    -------
    ax : matplotlib.axes.Axes
        The axis with the plotted image.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    img = plt.imread(file_path)

    if rotate:
        try:
            img_pil = Image.fromarray(img)
            img_pil = img_pil.rotate(rotate, expand=True)
            img = np.array(img_pil)
        except Exception as e:
            if rotate % 90 != 0:
                raise ValueError("Rotation requires Pillow for non-90° multiples.") from e
            if img.ndim < 2:
                raise ValueError("Cannot rotate image with ndim < 2.") from e
            k = int((rotate // 90) % 4)
            img = np.rot90(img, k=k, axes=(0, 1))

    if reflect is not None:
        if reflect == 'x':
            img = np.flip(img, axis=0)
        elif reflect == 'y':
            img = np.flip(img, axis=1)
        else:
            raise ValueError("Invalid reflect parameter. Use 'x' or 'y'.")

    if preserve_aspect:
        h, w = img.shape[:2]
        # keep x span fixed (0..1) and scale y span to match w/h
        y0 = extent[2]
        y1 = y0 + (extent[1] - extent[0]) * (h / w)
        extent = (extent[0], extent[1], y0, y1)
    if shift_x != 0.0 or shift_y != 0.0:
        extent = (extent[0] + shift_x, extent[1] + shift_x, 
                  extent[2] + shift_y, extent[3] + shift_y)
    ax.imshow(img, aspect=aspect, extent=extent, zorder=zorder, alpha=alpha)

    if remove_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        if preserve_aspect and shift_x != 0.0:
            ax.set_xlim(0, 1+shift_x)
        # ax.set_ylim(0, 1)
        ax.set_frame_on(False)   # removes spines cleanly

    return ax

def plot_fit_line(ax, fit_func, x_range, label=None, color='r', linestyle='-', linewidth=1):
    """Plot a fitted function line."""
    x = np.linspace(x_range[0], x_range[1], 100)
    y = fit_func(x)
    ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label)

def darken_if_light(color, threshold=0.85, factor=0.7):
    """Darken the color if it is lighter than the threshold.
        factor is the amount to darken (0.7 means 30% darker). 
    """
    r, g, b = to_rgb(color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if luminance >= threshold:
        return (r * factor, g * factor, b * factor)
    return (r, g, b)
