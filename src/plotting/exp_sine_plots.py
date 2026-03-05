import matplotlib.pyplot as plt
import random
import numpy as np

import bootstrap # sets up PROJCETROOT path for imports 

from src.plotting.primitives import (
    plot_distribution,
    CMAP,
)
from src.plotting.plot_layout import (
    style_axis,
    
)
from src.utils.math_tools import sine_2param
from src.analysis.exp_sine_analysis import normalize_smoothed_trj

def plot_exp_sine_R2_dist(events, ax=None, bins = 10,
                        xlim=None, ylim=None, xticks=None, yticks=None):
    """Plot distribution of R2 values from sine fits to experimental data."""
    R2_list = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,3))

    for event in events:
        res = event.sine_fit_2param["R2"]
        R2_list.append(res)
    ax = plot_distribution(data=R2_list,ax=ax, bins=bins, show_mean=True,
                    xlabel=r"$R^2$", ylabel="frequency",
                    color = 'gray', mean_color='k',)
    style_axis(ax,
                xlim=xlim,
                ylim=ylim,
                xticks=xticks,
                yticks=yticks,
            )

    return ax

def plot_exmp_trj_w_sine_fit(events, ax=None, indxs= None, ev_num = None,
                        xlim=None, ylim=None, xticks=None, yticks=None):
    """Plot example trajectory with sine fit overlay."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,3))
    if ev_num is None: ev_num = 10
    if indxs is None: indxs = [random.randint(0, len(events) - 1) for _ in range(ev_num)]
    for cnt in range(10):
        event = events[indxs[cnt]]
        A = event.sine_fit_2param["A"]
        w = event.sine_fit_2param["w"]
        t = event.sine_fit_2param["t_fit"]
        f_raw = event.F_bean[:len(t)]

        if np.isnan(A) or np.isnan(w) or A == 0 or w == 0 or len(t) == 0:
            continue

        else:
            fitted_f = sine_2param(t, A, w)
            ax.plot(t, f_raw, color='k', alpha=0.75, linestyle='-', 
                    linewidth=2, label='raw data' if cnt == 0 else "")
            ax.plot(t, fitted_f, color=CMAP(0.95), linestyle='--', linewidth=0.5, 
                    label=r'fit A$\sin\left(\frac{2\pi}{T}t\right)$' if cnt == 0 else "")
            
    style_axis(ax, 
               xlabel=r't (min)', 
               ylabel=r'F (mN)',
               xlim=xlim, ylim=ylim,
               xticks=xticks, yticks=yticks,
               )

    # ax.legend(loc='upper left',bbox_to_anchor=(0.05, 0.9))
    ax.legend(loc='upper right',bbox_to_anchor=(1.05, 1.05))
    
    return ax, indxs

def plot_exp_mean_trj(events, ax=None):
    """Plot mean normalized smoothed trajectory from experimental data."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,3))

    normed_trj = normalize_smoothed_trj(events)
    t_common = normed_trj["t_common"]
    mean = normed_trj["mean"]
    std = normed_trj["std"]

    ax.plot(t_common, mean, color='k', linewidth=1, label=r'$F_{\text{exp}}$',alpha=1)
    ax.fill_between(t_common, mean - std, mean + std, 
                    color='grey', alpha=0.4, linewidth=0, label=None)

    style_axis(ax, xlabel='t/T', ylabel='F/A', 
               xlim=[0,0.4], ylim=[0.0,1.1])

    return ax

def plot_all_events(events, ax=None,
                    xlim=None, ylim=None, xticks=None, yticks=None):
    """Plot example trajectory with sine fit overlay."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,3))


    for ev in events:
        t = ev.timer
        f_raw = ev.F_bean[:len(t)]
        ax.plot(t/60, f_raw, color='k', alpha=0.25, linewidth=0.5)
        
    style_axis(ax, 
               xlabel=r't (min)', 
               ylabel=r'F (mN)',
               xlim=xlim, ylim=ylim,
               xticks=xticks, yticks=yticks,
               )
    return ax


