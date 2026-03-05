import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Ellipse
from src.plotting.plot_layout import (
    style_axis,
    get_bin_format,

)

from src.plotting.primitives import (
    darken_if_light,
    CMAP,
)

def plot_normalized_trajectories(norm_data, ax, 
                                 xlabel="t/T", 
                                 ylabel='F/A'):
    """ plot mean interpolated trajectories with std fill"""
    for traj in norm_data["all"]:
        ax.plot(norm_data["t_common"], traj, color="grey", alpha=0.2, lw=0.5)

    ax.plot(
        norm_data["t_common"],
        norm_data["mean"],
        color="orange",
        lw=1,
        label="Mean"
    )
    ax.fill_between(
        norm_data["t_common"],
        norm_data["mean"] - norm_data["std"],
        norm_data["mean"] + norm_data["std"],
        color="orange",
        alpha=0.3
    )
    style_axis(
        ax,
        xlabel=xlabel,
        ylabel=ylabel)
    return ax

def plot_grouped_trajectories(ax, trajectories, groups, 
                              colors, time_step=0.5, bin_values=[],
                              xlabel='t (min)', ylabel='F (mN)', 
                              grouping_units='MPa', bin_format=".1f"):
    for traj, g in zip(trajectories, groups):
        if g >= 0 and len(traj):
            t = time_step * np.arange(len(traj))
            clr = darken_if_light(colors[g],threshold=0.65, factor=0.8)
            ax.plot(t, traj, color=clr, alpha=0.3)
    if len(bin_values) > 0:
        ax.legend(handles=[
        mlines.Line2D([0], [0], color=colors[g], lw=2, 
        label=f'{bin_values[g]:{bin_format}}-{bin_values[g+1]:{bin_format}} {grouping_units}')
        for g in range(len(bin_values)-1)],fontsize=10, loc='best')
    style_axis(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        grid={"which": "both", "linestyle": "--", "alpha": 0.5},
    )
    return ax

def plot_group_mean_trajectory(ax, trajectories, groups, colors, 
                    time_step=0.5, xlabel='t (min)', ylabel='F (mN)',
                    group_name = 'E', grouping_units='MPa', legend_params = None, 
                    bin_values=[], bin_format=".2f", 
                    xticks=None, yticks=None, xlim=None, ylim=None):
    
    colors = [darken_if_light(clr,threshold=0.7, factor=0.8) for clr in colors]
    
    for g in np.unique(groups):
        if g < 0 or g >= len(colors):
            continue
        group_trajs = [tr for tr, gg in zip(trajectories, groups) if gg == g]
        if not group_trajs:
            continue

        max_len = max(len(tr) for tr in group_trajs)
        mean = []
        sem = []

        for i in range(max_len):
            vals = [tr[i] for tr in group_trajs if i < len(tr)]
            mean.append(np.nanmean(vals))
            sem.append(np.nanstd(vals) / np.sqrt(len(vals)))

        t = time_step * np.arange(len(mean))
        ax.plot(t, mean, color=colors[g], lw=2)
        ax.fill_between(t, np.array(mean) - sem, np.array(mean) + sem,
                        color=colors[g], alpha=0.5,linewidth=0)

    if len(bin_values) > 0:
        if legend_params is None:
            legend_params = {"loc":'best',"ncol":1,
                            "title":f'{group_name} ({grouping_units})',
                            }
        ax.legend(handles=[
            mlines.Line2D([0], [0], color=colors[g], lw=2, 
            label=f'{bin_values[g]+(bin_values[g+1]-bin_values[g])/2:{bin_format}}')
            for g in range(len(bin_values)-1)], 
            **legend_params)
    style_axis(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        label_pad=1,
        xticks=xticks,
        yticks=yticks,
        xlim=xlim,
        ylim=ylim,
    )
    return ax

def plot_grouped_scatter_with_ellipse(ax=None, x=None, y=None, 
                                groups=None, colors=None, 
                                bin_values=[], xlabel='', ylabel='',
                                xlim = [0, 0.05], ylim = [0,0.5],
                                xticks=None, yticks=None,
                                group_name='E', grouping_units='MPa', 
                                legend_params=None, plot_points=True):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,3))
    if colors is None:
        unique_groups = np.unique(groups)
        colors = CMAP(np.linspace(0, 1, len(unique_groups)))
    if x is None or y is None or groups is None:
        raise ValueError("x, y, and groups must be provided")
    
    for g in np.unique(groups):
        if g < 0:
            continue
        mask = (groups == g) & np.isfinite(x) & np.isfinite(y)
        xx = x[mask]
        yy = y[mask]

        if plot_points: ax.scatter(xx, yy, color=colors[g], alpha=0.5, s=0.25)

        if len(xx) > 0 and len(yy) > 0:
            mx, my = np.nanmean(xx), np.nanmean(yy)

            sx = np.nanstd(xx) / np.sqrt(len(xx))
            sy = np.nanstd(yy) / np.sqrt(len(yy))
            clr = darken_if_light(colors[g],threshold=0.7, factor=0.9)
            ell = Ellipse((mx, my), width = 2*sx, height = 2*sy,
                          facecolor=clr, linewidth=1,alpha=0.9)
            ax.add_patch(ell)
            ax.plot(mx, my,
                    marker='+',
                    linestyle='None',
                    color=clr,
                    markersize=3,          # smaller
                    markeredgewidth=0.25    # sharper/thinner cross
                    )

    if len(bin_values) > 0:
        if len(bin_values) > 4: ncol=2
        else: ncol=1

        if legend_params is None:
            legend_params = {"loc":'best',"ncol":ncol,
                            "title":f'{group_name} ({grouping_units})'}
        
        bin_formats = get_bin_format(bin_values)

        ax.legend(handles=[
            mlines.Line2D([0], [0], color=colors[g], lw=2, 
            # label=f'{bin_values[g]:{bin_formats[g]}}-{bin_values[g+1]:{bin_formats[g]}}')
            label=f'{(bin_values[g]+bin_values[g+1])/2:{bin_formats[g]}}')
            for g in range(len(bin_values)-1)],
            columnspacing=1,**legend_params)

    style_axis(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
        yticks=yticks,
        xticks=xticks,
        label_pad=1,
    )
    return ax
