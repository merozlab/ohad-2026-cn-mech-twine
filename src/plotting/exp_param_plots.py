''' Module for plotting experimental parameter analysis results.
    Includes functions to plot R vs L-s, E vs L-s, and histograms of various
    experimental parameters.'''

import numpy as np

import bootstrap
from src.utils.math_tools import linfunc, parafunc

from src.plotting.primitives import (
    plot_errorbar,
    plot_distribution,
)

from src.plotting.plot_layout import (
    style_axis,
    create_inset,
)


def plot_R_vs_Ls(
    ax,
    R_stats,
    data_color='black',
    plot_fit=True,
    fit_color='red'
):
    """Plot R(L-s) with fit line option"""
    
    if plot_fit:
        plot_errorbar(
            ax,
            R_stats["L-s"],
            R_stats["R_mean"],
            yerr=R_stats["R_std"] / np.sqrt(len(R_stats["R_mean"])),
            xerr = 5/np.sqrt(12),  # Bin width error
            label=None,
            color=data_color,
            markersize = 2,
            capsize=1,
        )

        # Plot fit line
        R_fit_func = lambda x: linfunc(x, R_stats["a"], R_stats["b"])
        Ls_range = np.linspace(min(R_stats["L-s"]), max(R_stats["L-s"]), 100)
        ax.plot(Ls_range, R_fit_func(Ls_range), color=fit_color, label="R")

        style_axis(
            ax,
            xlabel="L-s (cm)",
            ylabel="R (cm)",
            ylim=(0.036, 0.08),
            yticks=np.linspace(0.04, 0.08, 5),
            label_pad=1.5,
            tick_params={"pad":2},
        )
    
    return ax,R_fit_func

def plot_E_vs_Ls(
    ax,
    E_stats,
    data_color='black',
    plot_fit=True,
    fit_color='red'
):
    """Plot E(L-s) with fit line option"""
    
    if plot_fit:
        plot_errorbar(
            ax,
        E_stats["L-s"],
        E_stats["E_mean"],
        yerr=E_stats["E_std"] / np.sqrt(len(E_stats["E_mean"])),
        xerr = 5/np.sqrt(12),  # Bin width error
        label=None,
        color=data_color,
        markersize = 2,
        capsize=1,
    )
    

        # Plot fit line
        E_fit_func = lambda x: parafunc(x, E_stats["a"], E_stats["b"])
        Ls_range = np.linspace(min(E_stats["L-s"]), max(E_stats["L-s"]), 100)
        ax.plot(Ls_range, E_fit_func(Ls_range), color=fit_color, label="E")

        style_axis(
            ax,
            xlabel="L-s (cm)",
            ylabel="E (MPa)",
            ylim=(-100, 1000),
            yticks=[0,250,500,750,1000],
            label_pad=1.5,
            tick_params={"pad":2},
        )
        ax.set_ylabel("E (MPa)", color=fit_color)
        ax.yaxis.set_label_coords(0.93, 0.5)
    return ax,E_fit_func


def legend_for_R_E(axR, axE):
    """Create unified legend for R and E plots"""
    handles_R, labels_R = axR.get_legend_handles_labels()
    handles_E, labels_E = axE.get_legend_handles_labels()
    axR.legend(handles_R + handles_E, labels_R + labels_E, 
               bbox_to_anchor=(0.1, 0.98, 0.0, 0.0), borderpad=0.5)
    return axR