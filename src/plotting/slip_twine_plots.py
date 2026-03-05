''' Module for plotting experimental results of slip twine measurements
    for various parameters.'''
#%% imports
# ---- imports ---
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import bootstrap
from src.utils.math_tools import (
    linfunc,
    powerfunc,
    polyfunc,
    one_over_x,
    shifted_linfunc,
)
from src.utils.curve_tools import (
    fit_w_error,
)

from src.plotting.primitives import (
    plot_distribution,

)
from src.plotting.plot_layout import (
    style_axis,

)

from src.utils.array_tools import (
    assign_equal_width_bins,
    assign_percentile_bins,
    merge_groups_N_bins,
)

CMAP = plt.cm.coolwarm
#%% plotting functions
# ---- plotting functions ----

# plot slip or twine time vs ltip

# plot slip or twine time vs tau (contact time)

# plot twine sucess rate vs Fresist


# get (no)twine times and remove nans
# no twine time is the time were no twine occured. usually disconnects backward or slip forward (motor too fast in either direction)
# save contact time before motor starts in new column
# save event type and plot accordingly

def plot_slip_twine_torque_hist(ax=None, slip=None, twine=None, nbins=20, 
                            threshold=None, xlim=None, ylim=None,
                            colors=None, xlabel=r'max(F$\ell_{\text{lev}}$) (mJ)', 
                            ylabel='frequency', xticks=None, yticks=None):
    '''Plot histogram of value for slip and twine events.'''
    if slip is None or twine is None:
        raise ValueError("Slip and twine arrays must be provided.")
    if ax is None:
        fig, ax = plt.subplots()
    if colors is None:
        colors = [CMAP(0.1), CMAP(0.9)]
    

    if threshold is not None:
        slip_torque = np.array([0.01*max(ev.F_bean)*ev.Lbase for ev in slip 
                                if 0.01*max(ev.F_bean)*ev.Lbase <= threshold]) # convert to mJ with 0.01 factor
        twine_torque = np.array([0.01*max(ev.F_bean)*ev.Lbase for ev in twine 
                                if 0.01*max(ev.F_bean)*ev.Lbase <= threshold])
    else: 
        slip_torque = np.array([0.01*max(ev.F_bean)*ev.Lbase for ev in slip 
                                if np.isfinite(max(ev.F_bean)*ev.Lbase)])
        twine_torque = np.array([0.01*max(ev.F_bean)*ev.Lbase for ev in twine 
                                 if np.isfinite(max(ev.F_bean)*ev.Lbase)])
        
    all_data = np.concatenate([slip_torque, twine_torque])
    shared_bins = np.histogram_bin_edges(all_data, bins=nbins)
    total_count = len(all_data)

    plot_distribution(ax, slip_torque, bins=shared_bins,
                      xlabel=xlabel, ylabel=ylabel,
                      alpha=0.6, show_mean=True, color=colors[0],
                      mean_legend='slip', mean_color=colors[0], total_count=total_count)

    plot_distribution(ax, twine_torque, bins=shared_bins,
                      xlabel=xlabel, ylabel=ylabel,
                      alpha=0.6, show_mean=True, color=colors[1],
                      mean_legend='twine', mean_color=colors[1], total_count=total_count)

    style_axis(ax,
            xlim=xlim,
            ylim=ylim,
            xticks=xticks,
            yticks=yticks)
    
    return ax

def plot_slip_twine_lo_hist(ax=None, slip=None, twine=None, nbins=20, 
                            threshold=None, xlim=None, ylim=None,
                            colors=None,xlabel=r'$\tau/T_{\mathrm{CN}}$', 
                            ylabel='frequency',xticks=None, yticks=None):
    '''Plot histogram of value for slip and twine events.'''
    if slip is None or twine is None:
        raise ValueError("Slip and twine arrays must be provided.")
    if ax is None:
        fig, ax = plt.subplots()
    if colors is None:
        colors = [CMAP(0.1), CMAP(0.9)]
    

    if threshold is not None:
        slip_lo = np.array([ev.ltip for ev in slip if ev.ltip <= threshold])
        twine_lo = np.array([ev.ltip for ev in twine if ev.ltip <= threshold])
    else: 
        slip_lo = np.array([ev.ltip for ev in slip if np.isfinite(ev.ltip)])
        twine_lo = np.array([ev.ltip for ev in twine if np.isfinite(ev.ltip)])
        
    all_data = np.concatenate([slip_lo, twine_lo])
    shared_bins = np.histogram_bin_edges(all_data, bins=nbins)
    total_count = len(all_data)

    plot_distribution(ax, slip_lo, bins=shared_bins,
                      xlabel=xlabel, ylabel=ylabel,
                      alpha=0.6, show_mean=True, color=colors[0],
                      mean_legend='slip', mean_color=colors[0], total_count=total_count)

    plot_distribution(ax, twine_lo, bins=shared_bins,
                      xlabel=xlabel, ylabel=ylabel,
                      alpha=0.6, show_mean=True, color=colors[1],
                      mean_legend='twine', mean_color=colors[1], total_count=total_count)

    style_axis(ax,
            xlim=xlim,
            ylim=ylim,
            xticks=xticks,
            yticks=yticks,
            )
    
    return ax



def plot_slip_twine_time_hist(ax=None, slip=None, twine=None, nbins=20, 
                            threshold=None, xlim=None, ylim=None, colors=None, 
                            xlabel=r'$\tau/T_{\mathrm{CN}}$', ylabel='frequency',
                            xticks=None, yticks=None):
    '''Plot histogram of value for slip and twine events.'''
    if slip is None or twine is None:
        raise ValueError("Slip and twine arrays must be provided.")
    if ax is None:
        fig, ax = plt.subplots()
    if colors is None:
        colors = [CMAP(0.1), CMAP(0.9)]
    

    if threshold is not None:
        slip_times = np.array([ev.twine_data['slip_time_norm'] for ev in slip
                       if hasattr(ev, "twine_data") 
                       and "slip_time_norm" in ev.twine_data 
                       and ev.twine_data['slip_time_norm'] <= threshold])
        twine_times = np.array([ev.twine_data['twine_estimate_norm'] for ev in twine 
                       if hasattr(ev, "twine_data") 
                       and "twine_estimate_norm" in ev.twine_data 
                       and ev.twine_data['twine_estimate_norm'] <= threshold])
    else: 
        slip_times = np.array([ev.twine_data['slip_time_norm'] for ev in slip 
                        if hasattr(ev, "twine_data") 
                        and "slip_time_norm" in ev.twine_data 
                        and np.isfinite(ev.twine_data['slip_time_norm'])])
        twine_times = np.array([ev.twine_data['twine_estimate_norm'] for ev in twine if 
                        hasattr(ev, "twine_data") 
                        and "twine_estimate_norm" in ev.twine_data 
                        and np.isfinite(ev.twine_data['twine_estimate_norm'])])
        
    all_data = np.concatenate([slip_times, twine_times])
    shared_bins = np.histogram_bin_edges(all_data, bins=nbins)
    total_count = len(all_data)

    plot_distribution(ax, slip_times, bins=shared_bins,
                      xlabel=xlabel, ylabel=ylabel,
                      alpha=0.6, show_mean=True, color=colors[0],
                      mean_legend='slip', mean_color=colors[0], total_count=total_count)

    plot_distribution(ax, twine_times, bins=shared_bins,
                      xlabel=xlabel, ylabel=ylabel,
                      alpha=0.6, show_mean=True, color=colors[1],
                      mean_legend='twine', mean_color=colors[1], total_count=total_count)

    style_axis(ax,
               xlim=xlim,
               ylim=ylim,
               xticks=xticks,
               yticks=yticks)
    
    ax.legend(bbox_to_anchor=(0.98, 0.1, 0.0, 0.0), loc='lower right',borderpad=0.1)

    return ax


def plot_time_vs_ltip(ax=None, slip=None, twine=None,
                     log_log_scale=False, colors=None,
                     t_threshold=None,ltip_threshold=None,
                     fit_line=False, fit_func=polyfunc,
                     xlabel=r'$\tau/T_{\mathrm{CN}}$', 
                     ylabel=r'$\ell_{\mathrm{o}}$ (cm)'):
    '''Plot slip or twine time vs ltip.'''
    if slip is None or twine is None:
        raise ValueError("Slip and twine event lists must be provided.")
    if ax is None:
        fig, ax = plt.subplots()
    if colors is None:
        colors = {0: CMAP(0.1), 1: CMAP(0.9)}

    # Extract slip data
    slip_times = np.array([ev.twine_data['slip_time_norm'] for ev in slip 
                    if hasattr(ev, "twine_data") 
                    and "slip_time_norm" in ev.twine_data 
                    and np.isfinite(ev.twine_data['slip_time_norm'])])
    slip_ltip = np.array([ev.ltip for ev in slip 
                    if hasattr(ev, "twine_data") 
                    and "slip_time_norm" in ev.twine_data 
                    and np.isfinite(ev.twine_data['slip_time_norm']) 
                    and np.isfinite(ev.ltip)])
    
    # Extract twine data
    twine_times = np.array([ev.twine_data['twine_estimate_norm'] for ev in twine 
                    if hasattr(ev, "twine_data") 
                    and "twine_estimate_norm" in ev.twine_data 
                    and np.isfinite(ev.twine_data['twine_estimate_norm'])])
    twine_ltip = np.array([ev.ltip for ev in twine 
                    if hasattr(ev, "twine_data") 
                    and "twine_estimate_norm" in ev.twine_data 
                    and np.isfinite(ev.twine_data['twine_estimate_norm']) 
                    and np.isfinite(ev.ltip)])

    # Apply thresholds separately to each array
    if t_threshold is not None:
        slip_t_mask = slip_times <= t_threshold
        slip_times = slip_times[slip_t_mask]
        slip_ltip = slip_ltip[slip_t_mask]
        
        twine_t_mask = twine_times <= t_threshold
        twine_times = twine_times[twine_t_mask]
        twine_ltip = twine_ltip[twine_t_mask]
    if ltip_threshold is not None:
        slip_l_mask = slip_ltip <= ltip_threshold
        slip_times = slip_times[slip_l_mask]
        slip_ltip = slip_ltip[slip_l_mask]
        
        twine_l_mask = twine_ltip <= ltip_threshold
        twine_times = twine_times[twine_l_mask]
        twine_ltip = twine_ltip[twine_l_mask]

    ax.scatter(slip_times, slip_ltip,
               s=2.5, alpha=0.7,
               color=colors[0], label="slip")
    ax.scatter(twine_times, twine_ltip,
               s=2.5, alpha=0.7,
               color=colors[1], label="twine")
    
    fit_results = None
    if fit_line:
        valid_slip = (slip_times > 0) & (slip_ltip > 0)
        slip_times, slip_ltip = slip_times[valid_slip], slip_ltip[valid_slip]
        if log_log_scale:
            # Fit power law to slip events in log-log space
            slip_times = np.log(slip_times)
            slip_ltip = np.log(slip_ltip)
            fit_func = linfunc

        time_err = abs(slip_times*0.1)
        ltip_err =  abs(0.1*np.ones_like(slip_ltip))

        fit_results = fit_w_error(slip_times, time_err, 
                                                          slip_ltip,ltip_err, fit_func=fit_func)
        
        popt = fit_results["popt"]
        x_fit = np.linspace(np.min(slip_times), np.max(slip_times), 100)
        y_fit = fit_func(x_fit, *popt)
        if log_log_scale:
            x_fit = np.exp(x_fit)
            y_fit = np.exp(y_fit)
        ax.plot(x_fit, y_fit, color='black', linestyle='--', 
                label=f'fit')

    style_axis(ax,
               xlabel=xlabel,
               ylabel=ylabel,
               xscale='log' if log_log_scale else 'linear',
               yscale='log' if log_log_scale else 'linear',
               label_pad=0.75,
               tick_params={"rotation": 25,"pad":0.75},
               )
    
    # for spine in ax.spines.values():
    #             spine.set_edgecolor('black')
    #             spine.set_linewidth(1)
    #             spine.set_visible(True)
    return ax, fit_results

def twine_ratio_Feff(ax=None, Feff=None, twine_states=None, 
                    color='k', nbins=10, infcolor = 'gray',
                    size=10, marker='o',
                    xlim=None, ylim=(0, 1),
                    xlabel=r'$F_{\text{res}}$ (mN)', ylabel='twine ratio',
                    label_inf = r'$F_{\text{res}}\rightarrow \infty$',
                    xticks=None, yticks=None, bin_mode = 'equal_width',
                    mask=None):
    '''Plot twine ratio vs effective mass for events.'''
    if Feff is None or twine_states is None:
        raise ValueError("Feff and twine_states must be provided.")
    if ax is None:
        fig, ax = plt.subplots()
    
    # turn into numpy arrays
    Feff = np.array(Feff)
    twine_states_arr = np.array(twine_states)

    # create mask for infinite values
    if mask is not None:
        inf_mask = np.isinf(Feff) & np.isfinite(twine_states_arr) & mask
    else:
        inf_mask = np.isinf(Feff) & np.isfinite(twine_states_arr)

    # external masking (e.g. by ltip)
    if mask is not None: 
        Feff = Feff[mask]
        twine_states_arr = twine_states_arr[mask]

    # Exclude infinite values for binning
    finite_mask = np.isfinite(Feff) & np.isfinite(twine_states_arr)
    Feff_finite = Feff[finite_mask]
    twine_states_arr_finite = twine_states_arr[finite_mask]
    
    # bins = np.linspace(np.min(Feff_finite), np.max(Feff_finite), num=nbins)
    if bin_mode == 'equal_width':
        groups, bins = assign_equal_width_bins(values=Feff_finite, nbins=nbins)
    elif bin_mode == 'percentile':
        groups, bins = assign_percentile_bins(values=Feff_finite, nbins=nbins)
                            
    twine_counts, _ = np.histogram(Feff_finite[twine_states_arr_finite == 1], bins=bins)
    total_counts, _ = np.histogram(Feff_finite, bins=bins)

    # Filter to non-empty bins
    pos_mask = total_counts > 0
    twine_counts = twine_counts[pos_mask]
    total_counts = total_counts[pos_mask]
    
    # Compute bin centers correctly from original bins
    bin_left = bins[:-1]   # left edges
    bin_right = bins[1:]   # right edges
    bin_centers = (bin_left + bin_right) / 2
    
    # Filter bin centers by mask
    bin_centers = bin_centers[pos_mask]

    twine_ratio = twine_counts / total_counts

    # Plot finite values
    ratio_mask = np.isfinite(twine_ratio)
    ax.scatter(bin_centers[ratio_mask], twine_ratio[ratio_mask],
               s=size, alpha=0.7, color=color, marker=marker)
    

    # Plot infinite values
    N_inf = np.sum(inf_mask)
    if np.any(inf_mask):
        inf_twine_ratio = np.sum(np.array(twine_states)[inf_mask]) / N_inf
        ax.axhline(inf_twine_ratio, linestyle='--',
            alpha=0.9, color=infcolor,
            label=label_inf)
        
    
    style_axis(ax,
                xlabel=xlabel,
                ylabel=ylabel,
                xlim=xlim,
                ylim=ylim,
                xticks=xticks,
                yticks=yticks
               )
    ax.legend(loc='lower right')
    return ax, bin_centers, twine_ratio, twine_counts, total_counts, N_inf


def plot_motor_twine(ax=None, motor_events=None, exp_twine_events=None, 
                    colors=('gray',CMAP(0.9)),
                    fit_line=False, xlabel='f (rph)', ylabel=r'$\tau$ (min)'):
    '''Plot normalized twine time vs normalized rotation rate for 
        motor events and experimental events. Get 'no-twine' times and plot
        with different symbol as motor twine.''' 
    if motor_events is None or exp_twine_events is None:
        raise ValueError("Motor events and experimental events data must be provided.")
    if ax is None:
        fig, ax = plt.subplots()


    # Slip-twine state
    motor_twine_time = np.array([ev.twine_time_estimate 
                            if hasattr(ev, "twine_time_estimate") 
                            else np.nan for ev in motor_events])
    rotation_rate = np.array([ev.plnt.eff_rot_rph 
                            if hasattr(ev, "plnt") and hasattr(ev.plnt, "eff_rot_rph") 
                            else np.nan for ev in motor_events])
    
    # remove nans
    mask = np.isfinite(motor_twine_time) & np.isfinite(rotation_rate) 
    mask &=  (motor_twine_time > 0) & (rotation_rate > 0)
    motor_twine_time = motor_twine_time[mask]
    rotation_rate = rotation_rate[mask]

    ax.scatter(rotation_rate, motor_twine_time,
               s=5, alpha=0.7, color=colors[0], label='motor')
    

    twine_after_stop = np.array([ev.twine_state == 'twine after motor stop' 
                            if hasattr(ev, "twine_state") else False for ev in motor_events])
    
    no_twine_time = np.array([ev.no_twine_time 
                            if hasattr(ev, "no_twine_time") 
                            else np.nan for ev in motor_events])
    no_twine_rotation_rate = np.array([ev.plnt.eff_rot_rph 
                            if hasattr(ev, "plnt") and hasattr(ev.plnt, "eff_rot_rph") 
                            else np.nan for ev in motor_events])
    no_twine_time = no_twine_time[twine_after_stop]
    no_twine_rotation_rate = no_twine_rotation_rate[twine_after_stop]

    ax.scatter(no_twine_rotation_rate, no_twine_time,
               s=5, alpha=0.7, color=colors[0], label=None, marker='^')
    
    exp_twine_time = np.array([ev.twine_data['twine_estimate_min'] for ev in exp_twine_events 
                        if hasattr(ev, "twine_data") 
                        and "twine_estimate_min" in ev.twine_data
                        and hasattr(ev, "plnt") 
                        and hasattr(ev.plnt, "avgT")
                        and np.isfinite(ev.twine_data['twine_estimate_min'])
                        and np.isfinite(ev.plnt.avgT)])
    
    exp_rotation_rate = np.array([60/ev.plnt.avgT for ev in exp_twine_events 
                        if hasattr(ev, "twine_data") 
                        and "twine_estimate_min" in ev.twine_data
                        and hasattr(ev, "plnt") 
                        and hasattr(ev.plnt, "avgT")
                        and np.isfinite(ev.twine_data['twine_estimate_min'])
                        and np.isfinite(ev.plnt.avgT)])

    mean_exp = (np.nanmean(exp_rotation_rate), np.nanmean(exp_twine_time))
    # ax.scatter(*mean_exp, s=100, color=CMAP(0.5), label=None, edgecolor='k')
    ax.errorbar(mean_exp[0], mean_exp[1], 
                xerr=np.nanstd(exp_rotation_rate), yerr=np.nanstd(exp_twine_time),
                fmt='+', color='k', label=None, alpha=0.75, capsize=1, zorder=5)
    
    ax.scatter(exp_rotation_rate, exp_twine_time,
               s=5, alpha=0.7, color=colors[1], label='pendulum')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.labelpad = 0.5

    if fit_line:
        one_over_fit = fit_w_error(rotation_rate, 0.1, motor_twine_time, 1,
                                  fit_func=one_over_x)
        popt = one_over_fit["popt"]
        x_fit = np.linspace(0, np.max(rotation_rate), 100, endpoint=False)
        y_fit = one_over_x(x_fit, *popt)
        ax.plot(x_fit, y_fit, color='k', label='fit a/x+b',alpha=0.75)
        ax.legend()
        style_axis(ax=ax.axs[1],
            yticks = [100,200,300,400],
            )
        return ax, one_over_fit
    
    ax.axs[0].legend() # since i know this is a borken axis object

    return ax, None



def plot_slip_twine_curvature_hist(ax=None,arr=None, nbins=20, 
                            xlim=None, ylim=None,
                            colors=None, xlabel=r'$\kappa-\kappa_{{0}}$ (cm$^{-1}$)', 
                            ylabel='frequency'):
    '''Plot histogram of value for slip and twine events.'''
    if arr is None:
        raise ValueError("data array must be provided.")
    if ax is None:
        fig, ax = plt.subplots()
    if colors is None:
        colors = [CMAP(0.1), CMAP(0.9)]
    

    slip_curv = np.array([1e-5*row["max_F_s"]/row["B_contact"] for _, row in arr.iterrows() if 
            "max_F_s" in row and "B_contact" in row and "twine_state" in row 
            and np.isfinite(row["max_F_s"]) and np.isfinite(row["B_contact"]) 
            and row["twine_state"] == 1])
    twine_curv = np.array([1e-5*row["max_F_s"]/row["B_contact"] for _, row in arr.iterrows() if 
            "max_F_s" in row and "B_contact" in row and "twine_state" in row
            and np.isfinite(row["max_F_s"]) and np.isfinite(row["B_contact"]) 
            and row["twine_state"] == 0]
    )                

        
    all_data = np.concatenate([slip_curv, twine_curv])
    shared_bins = np.histogram_bin_edges(all_data, bins=nbins)
    total_count = len(all_data)

    plot_distribution(ax, slip_curv, bins=shared_bins,
                      xlabel=xlabel, ylabel=ylabel,
                      alpha=0.6, show_mean=True, color=colors[0],
                      mean_legend='slip', mean_color=colors[0], total_count=total_count)

    plot_distribution(ax, twine_curv, bins=shared_bins,
                      xlabel=xlabel, ylabel=ylabel,
                      alpha=0.6, show_mean=True, color=colors[1],
                      mean_legend='twine', mean_color=colors[1], total_count=total_count)

    style_axis(ax,
            xlim=xlim,
            ylim=ylim)
    
    return ax,slip_curv, twine_curv


def ltip_tcont_multi_plot(ax=None, slip=None, twine=None, 
                        xlabel=r'$\tau/T_{\mathrm{CN}}$',
                        ylabel=r'$\ell_{\mathrm{o}}$ (cm)',
                        nbins=30, colors=None, t_threshold=None, ltip_threshold=None):
    '''Plot slip or twine time vs ltip with marginal histograms.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, creates new figure.
    slip : list
        List of slip events.
    twine : list
        List of twine events.
    xlabel : str
        Label for x-axis (time).
    ylabel : str
        Label for y-axis (ltip).
    nbins : int
        Number of bins for marginal histograms.
    colors : tuple or None
        Color tuple (slip_color, twine_color). If None, uses coolwarm colormap.
    t_threshold : float or None
        Threshold for time values.
    ltip_threshold : float or None
        Threshold for ltip values.
    '''
    if slip is None or twine is None:
        raise ValueError("Slip and twine event lists must be provided.")
    if colors is None:
        colors = [CMAP(0.1), CMAP(0.9)]
    if ax is not None:
        fig = ax.figure
        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], 
                              hspace=0.05, wspace=0.05)
    else:
        # Create figure with GridSpec for main plot + marginals
        fig = plt.figure(figsize=(4, 4))
        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], 
                              hspace=0.05, wspace=0.05)
    
    # Main scatter plot (bottom-left)
    ax_main = fig.add_subplot(gs[1, 0])
    # Top histogram (time axis)
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    # Right histogram (ltip axis)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    
    # Extract slip data
    slip_times = np.array([ev.twine_data['slip_time_norm'] for ev in slip 
                    if hasattr(ev, "twine_data") 
                    and "slip_time_norm" in ev.twine_data 
                    and np.isfinite(ev.twine_data['slip_time_norm'])])
    slip_ltip = np.array([ev.ltip for ev in slip 
                    if hasattr(ev, "twine_data") 
                    and "slip_time_norm" in ev.twine_data 
                    and np.isfinite(ev.twine_data['slip_time_norm']) 
                    and np.isfinite(ev.ltip)])
    
    # Extract twine data
    twine_times = np.array([ev.twine_data['twine_estimate_norm'] for ev in twine 
                    if hasattr(ev, "twine_data") 
                    and "twine_estimate_norm" in ev.twine_data 
                    and np.isfinite(ev.twine_data['twine_estimate_norm'])])
    twine_ltip = np.array([ev.ltip for ev in twine 
                    if hasattr(ev, "twine_data") 
                    and "twine_estimate_norm" in ev.twine_data 
                    and np.isfinite(ev.twine_data['twine_estimate_norm']) 
                    and np.isfinite(ev.ltip)])
    
    # Apply thresholds
    if t_threshold is not None:
        slip_t_mask = slip_times <= t_threshold
        slip_times = slip_times[slip_t_mask]
        slip_ltip = slip_ltip[slip_t_mask]
        
        twine_t_mask = twine_times <= t_threshold
        twine_times = twine_times[twine_t_mask]
        twine_ltip = twine_ltip[twine_t_mask]
    
    if ltip_threshold is not None:
        slip_l_mask = slip_ltip <= ltip_threshold
        slip_times = slip_times[slip_l_mask]
        slip_ltip = slip_ltip[slip_l_mask]
        
        twine_l_mask = twine_ltip <= ltip_threshold
        twine_times = twine_times[twine_l_mask]
        twine_ltip = twine_ltip[twine_l_mask]
    
    # Main scatter plot
    ax_main.scatter(slip_times, slip_ltip, s=15, alpha=0.7, 
                   color=colors[0], label='slip')
    ax_main.scatter(twine_times, twine_ltip, s=15, alpha=0.7, 
                   color=colors[1], label='twine')
    ax_main.legend(loc='lower right')
    style_axis(ax_main, xlabel=xlabel, ylabel=ylabel)
    
    # Top histogram - time distribution
    all_times = np.concatenate([slip_times, twine_times])
    time_bins = np.linspace(np.nanmin(all_times), np.nanmax(all_times), nbins + 1)
    
    ax_top.hist(slip_times, bins=time_bins, alpha=0.6, color=colors[0], density=True)
    ax_top.hist(twine_times, bins=time_bins, alpha=0.6, color=colors[1], density=True)
    ax_top.axvline(np.nanmean(slip_times), color=colors[0], linestyle='--', linewidth=1)
    ax_top.axvline(np.nanmean(twine_times), color=colors[1], linestyle='--', linewidth=1)
    ax_top.tick_params(labelbottom=False, labelleft=False)
    for spine in ax_top.spines.values():
        spine.set_visible(False)
    
    # Right histogram - ltip distribution
    all_ltips = np.concatenate([slip_ltip, twine_ltip])
    ltip_bins = np.linspace(np.nanmin(all_ltips), np.nanmax(all_ltips), nbins + 1)
    
    ax_right.hist(slip_ltip, bins=ltip_bins, alpha=0.6, color=colors[0], 
                 orientation='horizontal', density=True)
    ax_right.hist(twine_ltip, bins=ltip_bins, alpha=0.6, color=colors[1], 
                 orientation='horizontal', density=True)
    ax_right.axhline(np.nanmean(slip_ltip), color=colors[0], linestyle='--', linewidth=1)
    ax_right.axhline(np.nanmean(twine_ltip), color=colors[1], linestyle='--', linewidth=1)
    ax_right.tick_params(labelbottom=False, labelleft=False)
    for spine in ax_right.spines.values():
        spine.set_visible(False)
    
    return fig, ax_main, ax_top, ax_right


def plot_slip_time_fit(ax=None, slip=None, 
                        fit_func=powerfunc,
                        xlabel=r'$\tau/T_{\mathrm{CN}}$', 
                        ylabel=r'$\ell_{\mathrm{o}}$ (cm)',
                        ):
    '''Plot slip time vs ltip with fit line.'''
    if slip is None:
        raise ValueError("Slip event list must be provided.")
    if ax is None:
        fig, ax = plt.subplots()
    ev_color = CMAP(0.1)

    slip_times = np.array([ev.twine_data['slip_time_norm'] for ev in slip 
                    if hasattr(ev, "twine_data") 
                    and "slip_time_norm" in ev.twine_data 
                    and np.isfinite(ev.twine_data['slip_time_norm'])])
    slip_ltip = np.array([ev.ltip for ev in slip 
                    if hasattr(ev, "twine_data") 
                    and "slip_time_norm" in ev.twine_data 
                    and np.isfinite(ev.twine_data['slip_time_norm']) 
                    and np.isfinite(ev.ltip)])
    
    ax.scatter(slip_times, slip_ltip,
               s=2.5, alpha=0.7,
               color=ev_color, label="slip")
    
    valid_slip = (slip_times > 0) & (slip_ltip > 0)
    slip_times, slip_ltip = slip_times[valid_slip], slip_ltip[valid_slip]
    time_err = abs(slip_times*0.1)
    ltip_err = abs(0.1*np.ones_like(slip_ltip))

    popt,pcov,r2,chi_sq_red, chi_sq_std = fit_w_error(slip_times, time_err, 
                                                      slip_ltip,ltip_err, fit_func=fit_func)
    
    x_fit = np.linspace(np.min(slip_times), np.max(slip_times), 100)
    y_fit = fit_func(x_fit, *popt)
    ax.plot(x_fit, y_fit, color='black', linestyle='--', label=f'fit')

    # estimated slip time using cantilever bending analysis
    l_lev_vec = np.linspace(0,4,100) # cm
    Ct = 0.14
    t_slip_cant = Ct * (l_lev_vec**2) 
    ax.plot(t_slip_cant, l_lev_vec, 'k--', linewidth=2, label=r'Cantilever analysis: $\tau = $' + f'{Ct}' + r'$\ell_{o}^{2}$')

    # Power-law fitted function: ltip = 1.855 * (tau/Tcn)^0.491
    tau_fit = np.linspace(0, 1.0, 200)
    ltip_fit = 1.855 * tau_fit**0.491
    ax.plot(tau_fit, ltip_fit, color='k', linestyle='-',
                label=r'Fit: $\ell_{o}=1.85\,(\tau/T_{\text{CN}})^{0.49}$')

    style_axis(ax,
               xlabel=xlabel,
               ylabel=ylabel,
               )
    
    ax.legend(loc='lower right')
    
    return ax, (popt, pcov, r2, chi_sq_red, chi_sq_std)



def plot_motor_twine_lin(ax=None, motor_events=None, exp_twine_events=None, 
                    colors=('gray',CMAP(0.9)), yticks=None,
                    fit_line=False, xlabel='f (rph)', ylabel=r'$\tau$ (min)'):
    '''Plot normalized twine time vs normalized rotation time for 
        motor events and experimental events. Get 'no-twine' times and plot
        with different symbol as motor twine.''' 
    if motor_events is None or exp_twine_events is None:
        raise ValueError("Motor events and experimental events data must be provided.")
    if ax is None:
        fig, ax = plt.subplots()


    # Slip-twine state
    motor_twine_time = np.array([ev.twine_time_estimate*ev.plnt.eff_rot_rph/60
                            if hasattr(ev, "twine_time_estimate") and hasattr(ev, "plnt") 
                            and hasattr(ev.plnt, "eff_rot_rph") 
                            else np.nan for ev in motor_events])
    rotation_time = np.array([60/ev.plnt.eff_rot_rph
                            if hasattr(ev, "plnt") and hasattr(ev.plnt, "eff_rot_rph") 
                            and np.isfinite(ev.plnt.eff_rot_rph)
                            and ev.plnt.eff_rot_rph > 0
                            else np.nan for ev in motor_events])
    
    # remove nans
    mask = np.isfinite(motor_twine_time) & np.isfinite(rotation_time) 
    mask &=  (motor_twine_time > 0) & (rotation_time > 0)
    motor_twine_time = motor_twine_time[mask]
    rotation_time = rotation_time[mask]

    ax.scatter(rotation_time, motor_twine_time,
               s=5, alpha=0.7, color=colors[0], label='motor')
    

    twine_after_stop = np.array([ev.twine_state == 'twine after motor stop' 
                            if hasattr(ev, "twine_state") else False for ev in motor_events])
    
    no_twine_time = np.array([ev.no_twine_time*ev.plnt.eff_rot_rph/60
                            if hasattr(ev, "no_twine_time") 
                            and hasattr(ev, "plnt") 
                            and hasattr(ev.plnt, "eff_rot_rph")
                            else np.nan for ev in motor_events])
    no_twine_rotation_time = np.array([60/ev.plnt.eff_rot_rph 
                            if hasattr(ev, "plnt") and hasattr(ev.plnt, "eff_rot_rph") 
                            and np.isfinite(ev.plnt.eff_rot_rph)
                            and ev.plnt.eff_rot_rph > 0
                            else np.nan for ev in motor_events])
    no_twine_time = no_twine_time[twine_after_stop]
    no_twine_rotation_time = no_twine_rotation_time[twine_after_stop]

    ax.scatter(no_twine_rotation_time, no_twine_time,
               s=5, alpha=0.7, color=colors[0], label=None, marker='^')
    

    exp_twine_time = np.array([ev.twine_data['twine_estimate_min']/ev.plnt.avgT for ev in exp_twine_events 
                        if hasattr(ev, "twine_data") 
                        and "twine_estimate_min" in ev.twine_data
                        and hasattr(ev, "plnt") 
                        and hasattr(ev.plnt, "avgT")
                        and np.isfinite(ev.twine_data['twine_estimate_min'])
                        and np.isfinite(ev.plnt.avgT)
                        and ev.plnt.avgT > 0])
    
    exp_rotation_time = np.array([ev.plnt.avgT for ev in exp_twine_events 
                        if hasattr(ev, "twine_data") 
                        and "twine_estimate_min" in ev.twine_data
                        and hasattr(ev, "plnt") 
                        and hasattr(ev.plnt, "avgT")
                        and np.isfinite(ev.twine_data['twine_estimate_min'])
                        and np.isfinite(ev.plnt.avgT)
                        and ev.plnt.avgT > 0])

    mean_exp = (np.nanmean(exp_rotation_time), np.nanmean(exp_twine_time))
    # ax.scatter(*mean_exp, s=100, color=CMAP(0.5), label=None, edgecolor='k')
    ax.errorbar(mean_exp[0], mean_exp[1], 
                xerr=np.nanstd(exp_rotation_time), yerr=np.nanstd(exp_twine_time),
                fmt='+', color='k', label=None, alpha=0.75, capsize=1, zorder=5)
    
    ax.scatter(exp_rotation_time, exp_twine_time,
               s=5, alpha=0.7, color=colors[1], label='pendulum')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.labelpad = 0.5

    # append exp data to motor data for fitting
    # rotation_time = np.append(rotation_time, exp_rotation_time)
    # motor_twine_time = np.append(motor_twine_time, exp_twine_time)

    if fit_line:
        func_fit = fit_w_error(rotation_time, 5, motor_twine_time, 0.5,
                                  fit_func=shifted_linfunc)
        popt = func_fit["popt"]
        x_fit = np.linspace(0, np.max(rotation_time), 100, endpoint=False)
        y_fit = shifted_linfunc(x_fit, *popt)
        ax.plot(x_fit, y_fit, color='k', label='fit ax+b',alpha=0.75)
        ax.legend()
        style_axis(ax=ax,
            yticks = yticks,
            )
        return ax, func_fit
    
    ax.legend() # since i know this is a borken axis object

    return ax, None


#%% 