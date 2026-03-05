import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from src.plotting.primitives import (
    plot_distribution,
    CMAP,
    darken_if_light,

)

from src.plotting.plot_layout import style_axis
from src.analysis.exp_sine_analysis import ( 
    normalize_smoothed_trj,
)

from src.utils.math_tools import sine_2param, linfunc, powerfunc
from src.utils.curve_tools import fit_w_error, calc_R2_RMSE

def plot_sine_fit_examples(events, indices, ax=None):
    """
    Plot example force trajectories and corresponding 2-param sine fits.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)

    for i in indices:
        ev = events[i]

        if not hasattr(ev, "sine_fit_2param"):
            continue

        t = ev.sine_fit_2param["t_fit"]
        y = ev.sine_fit_2param["y_fit"]
        A = ev.sine_fit_2param["A"]
        w = ev.sine_fit_2param["w"]

        ax.plot(t, y, color="k", alpha=0.7)
        ax.plot(t, A * np.sin(w * t), "r--", alpha=0.8)

    style_axis(
        ax,
        xlabel="t",
        ylabel="F (mN)",
    )

    return ax
    
def plot_sim_trj_with_sine_fit(events, indices=None, ax=None):
    """
    Plot normalized force trajectories with sine fits overlaid.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)

    if indices is None:
        indices = range(len(events))

    for i in indices:
        ev = events[i]

        if not hasattr(ev, "sine_fit_2param"):
            continue

        t = ev.sine_fit_2param["t_fit"]
        y = ev.sine_fit_2param["y_fit"]
        A = ev.sine_fit_2param["A"]
        w = ev.sine_fit_2param["w"]

        ax.plot(t, y, color="k", alpha=0.7)
        ax.plot(t, sine_2param(t, A, w), "r--", alpha=0.8)
    ax.legend(['sim','fit'])
    style_axis(
        ax,
        xlabel=r"$t/T_{cn}$",
        ylabel="F (mN)",
    )

    return ax

def plot_normalized_sim_trj(events, indices=None,ax=None, show_trj = False):
    """
    Plot normalized mean force trajectories from simulation events.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)

    if indices is None:
        indices = range(len(events))

    selected_events = [events[i] for i in indices]
    norm_data = normalize_smoothed_trj(selected_events)

    t_common = norm_data["t_common"]
    mean = norm_data["mean"]
    std = norm_data["std"]

    ax.plot(t_common, mean, color=CMAP(0.05), label=r"$F_{\text{sim}}$", 
            lw=2, linestyle='dotted')
    ax.fill_between(t_common, mean - std, mean + std, 
                    color=CMAP(0.1), linewidth=0, alpha=0.3, label=None)
    if show_trj:
        for ev in events:
            A = ev.sine_fit_2param["A"]
            trj = ev.sine_fit_2param["y_fit"]/A
            T = ev.sine_fit_2param["T"]
            t = ev.sine_fit_2param["t_fit"]/T
            ax.plot(t, trj, color="k", alpha=0.3,linewidth=0.5)
            
        ax.legend()
        style_axis(
            ax,
            xlabel=r"t/$T_{\text{cn}}$",
            ylabel="F/A",
        )

    return ax

def plot_pure_sine(ax=None,x_cutoff=1.0,
                    xlim=None, ylim=None, xticks=None, yticks=None):
    """
    Plot a clean reference sine wave.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
    
    t = np.linspace(0, 2*np.pi*x_cutoff, 200)
    ax.plot(t/(2*np.pi), np.sin(t),'--', color='r',linewidth=1,
                label=r'A$\sin\left(\frac{2\pi}{T}t\right)$')
    style_axis(
        ax,
        xlabel='t/T', 
        ylabel='F/A', 
        xlim=xlim,
        ylim=ylim,
        xticks=xticks,
        yticks=yticks,
        )
       
    return ax


def plot_sim_Fvec_vs_Fplanar(events, indices=None, ax=None):
    """
    Plot F_vec vs F_vec_planar for simulation events.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)

    if indices is None:
        indices = range(len(events))

    for idx, i in enumerate(indices):
        color = plt.cm.coolwarm(idx / len(indices))
        ev = events[i]
        F_vec = ev.F_shifted
        F_planar = ev.F_planar_shifted
        t = ev.T_shifted

        ax.plot(t, F_vec,'-', alpha=0.3, color=color)
        ax.plot(t, F_planar, '--', alpha=0.3, color=color)

    # Custom legend with black color
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='k', lw=1, label='F_vec'),
                      Line2D([0], [0], color='k', lw=1, linestyle='--', label='F_planar')]
    ax.legend(handles=legend_elements)
    
    style_axis(
        ax,
        xlabel= r"$t/T_{\text{cn}}$",
        ylabel="F (mN)",
    )

    return ax

def plot_sim_R2_distribution(events, ax=None):
    """
    Plot histogram of R^2 values from sine fits of simulation events.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)

    R2_values = [
        ev.sine_fit_2param["R2"] 
        for ev in events 
        if hasattr(ev, "sine_fit_2param")
    ]

    plot_distribution(
        data=R2_values,
        bins=20,
        ax=ax,
        alpha=0.7,
        density=True,
        xlabel="R²",
        ylabel="Frequency",
        show_mean=True,
    )


    return ax

def plot_sim_F_t_constpar(events, friction = False, 
                            constpar='E', const_value_rank=1, varypar='l_lev', 
                            ax=None, xlim=None, ylim=None, legend_on=True,
                            val_format='.0f', legend_loc=None, legend_units='',
                            legend_title=None, max_varypar=None,normalizer=None):
    """
    Plot Force vs time for simulation events with given constant parameter.
    """
    # Dictionary mapping for specific E values
    par_values = sorted(set([getattr(ev, varypar) for ev in events]))
    if max_varypar is not None:
        if max_varypar > 0:
            # Pos keep values <= max_varypar or opposite for neg
            par_values = [v for v in par_values if v <= max_varypar]
        else:
            threshold = abs(max_varypar)
            par_values = [v for v in par_values if v >= threshold]
    colors = {val: darken_if_light(
                CMAP((val-min(par_values)) / (max(par_values)-min(par_values))),threshold=0.75) 
                for val in par_values}
    
    
    
    const_vals = sorted(set([getattr(ev,constpar) for ev in events]))
    const_value = const_vals[const_value_rank]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
    i=0
    for ev in events:
        if getattr(ev, constpar) != const_value:
            continue
        if max_varypar is not None:
            ev_val = getattr(ev, varypar)
            if max_varypar > 0 and ev_val > max_varypar:
                continue
            if max_varypar < 0 and ev_val < abs(max_varypar):
                continue
        t = ev.T_shifted
        F = ev.F_shifted
        mu = ev.mu
        if not friction:
            if mu > 0:  # Skip friction cases
                continue
        norm = getattr(ev, normalizer) if normalizer is not None else 1
        ax.plot(t, F, 
                color=colors[getattr(ev, varypar)], linewidth=1,
                label = f'{getattr(ev, varypar)/norm:{val_format}} {legend_units}')
        i+=1

    if legend_on:
        # Sort legend handles and labels by varypar value
        handles, labels = ax.get_legend_handles_labels()
        label_value_pairs = [(h, l, float(l.split()[0])) for h, l in zip(handles, labels)]
        label_value_pairs.sort(key=lambda x: x[2])
        sorted_handles = [x[0] for x in label_value_pairs]
        sorted_labels = [x[1] for x in label_value_pairs]

        if legend_title is None:
            legend_title = varypar
        
        if legend_loc is not None:
            ax.legend(handles=sorted_handles, labels=sorted_labels, 
                      title=f'{legend_title}', 
                     loc='upper left',
                     bbox_to_anchor=legend_loc, 
                     bbox_transform=ax.transAxes)
        else:
            ax.legend(handles=sorted_handles, labels=sorted_labels,
                      title=f'{legend_title}', loc='lower right')

    style_axis(
        ax,
        xlabel= r"$t/T_{\text{cn}}$",
        ylabel= "F (mN)",
        xlim=xlim,
        ylim=ylim,
        label_pad=3,
        tick_params={
        'pad': 4,
        'length': 1,
        'width': 0.25
    })

    return ax

def plot_sim_F_t_constE_varL(events, friction = False, 
                            constpar='E', const_value_rank=1, varypar='l_lev', 
                            ax=None, xlim=None, ylim=None, legend_on=True,
                            val_format='.0f', legend_loc=None, legend_units='',
                            legend_title=None, max_varypar=None,normalizer=None,
                            varypar_values=None, xticks=None, yticks=None):
    """
    Plot Force vs time for simulation events with given constant parameter.
    """
    # Dictionary mapping for specific E values
    par_values = sorted(set([getattr(ev, varypar) for ev in events]))
    if max_varypar is not None:
        if max_varypar > 0:
            # Pos keep values <= max_varypar or opposite for neg
            par_values = [v for v in par_values if v <= max_varypar]
        else:
            threshold = abs(max_varypar)
            par_values = [v for v in par_values if v >= threshold]
    if varypar_values is not None:
        par_values = [par_values[i] for i in varypar_values]
    
    # Normalize colors based on original range, not filtered range
    all_par_values = sorted(set([getattr(ev, varypar) for ev in events]))
    colors = {val: darken_if_light(
                CMAP((val-min(all_par_values)) / (max(all_par_values)-min(all_par_values))),threshold=0.75) 
                for val in par_values}
    
    const_vals = sorted(set([getattr(ev,constpar) for ev in events]))
    const_value = const_vals[const_value_rank]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
    i=0
    for ev in events:
        if getattr(ev, constpar) != const_value:
            continue
        if max_varypar is not None:
            ev_val = getattr(ev, varypar)
            if max_varypar > 0 and ev_val > max_varypar:
                continue
            if max_varypar < 0 and ev_val < abs(max_varypar):
                continue
        t = ev.T_shifted
        F = ev.F_shifted
        mu = ev.mu
        if not friction:
            if mu > 0:  # Skip friction cases
                continue
        norm = getattr(ev, normalizer) if normalizer is not None else 1
        ev_val = getattr(ev, varypar)
        if ev_val in colors:  # Only plot if value is in filtered list
            ax.plot(t, F, 
                    color=colors[ev_val], linewidth=1,
                    label = f'{ev_val/norm:{val_format}} {legend_units}')
            i+=1

    if legend_on:
        # Sort legend handles and labels by varypar value
        handles, labels = ax.get_legend_handles_labels()
        label_value_pairs = [(h, l, float(l.split()[0])) for h, l in zip(handles, labels)]
        label_value_pairs.sort(key=lambda x: x[2])
        sorted_handles = [x[0] for x in label_value_pairs]
        sorted_labels = [x[1] for x in label_value_pairs]

        if legend_title is None:
            legend_title = varypar
        
        if legend_loc is not None:
            ax.legend(handles=sorted_handles, labels=sorted_labels, 
                      title=f'{legend_title}', 
                     loc='upper left',
                     bbox_to_anchor=legend_loc, 
                     bbox_transform=ax.transAxes)
        else:
            ax.legend(handles=sorted_handles, labels=sorted_labels,
                      title=f'{legend_title}', loc='lower right')

    style_axis(
        ax,
        xlabel= r"$t/T_{\text{cn}}$",
        ylabel= "F (mN)",
        xlim=xlim,
        ylim=ylim,
        xticks=xticks,
        yticks=yticks,
        label_pad=3,
        tick_params={
        'pad': 4,
        'length': 1,
        'width': 0.25
    })


def plot_sim_torque(events, friction = False, ax=None, 
                    xlim=None, ylim=None, legend_on=True):
    """
    Plot torque vs time for simulation events.
    """
    # Dictionary mapping for specific E values
    colors = [CMAP(i / 9.0) for i in range(10)]
    E_colors = {10: colors[0], 50: colors[4], 100: colors[9]}
    E_colors = {k: darken_if_light(v, threshold=0.75) for k, v in E_colors.items()}

    # define alpha based on LOC
    locs = sorted(set([ev.LOC for ev in events]))
    LOC_alpha = {l:(l/max(locs)) for l in locs} # high LOC -> high ell_lev->high alpha?

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)

    for i, ev in enumerate(events):
        t = ev.T_shifted
        Fs = ev.Fs
        mu = ev.mu
        if not friction:
            if mu > 0:  # Skip friction cases
                continue

        ax.plot(t, Fs, alpha=LOC_alpha.get(ev.LOC, 0.5), 
                color=E_colors.get(ev.E, 'grey'), linewidth=0.5)

    if xlim is None: xlim = (0, 1.1*max(t))
    if ylim is None: ylim = (0, 1.1*max([max(ev.Fs) for ev in events
                                          if ev.mu==0]))

    style_axis(
        ax,
        xlabel= r"$t/T_{\text{cn}}$",
        ylabel= r" F$\ell_{\text{lev}}$ (mJ)",
        xlim=xlim,
        ylim=ylim,
        label_pad=0.75,
        # xticks=np.arange(0, 0.5,0.2),
        background_color='0.95',
        nbin_ticks=3,
        tick_params={
        'pad': 1.5,'rotation':25,
        }
    )
    # Move y-axis ticks and label to the right
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.patch.set_alpha(0.75)

    if legend_on:
        ax.legend([f'E={E}' for E in sorted(E_colors.keys())], 
                  title='E (MPa)', loc='upper right')
    return ax


def plot_sim_FsBk(ax=None, events = None, fit = True, fit_func = linfunc,
                    xlim=None, ylim=None,
                    xticks=None, yticks=None):
    """
    Plot bending moments max(Fs) and Bk and fit if requested
    """
    if events is None:
        print("No events provided.")
        return ax, None
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
        
    max_Fs_vec = []
    Bk_vec = []
    E_vec = []

    for i, ev in enumerate(events):
        if ev.mu > 0:  # Skip friction cases
            continue

        Bk_vec.append(ev.Bk)
        max_Fs_vec.append(max(ev.Fs))
        E_vec.append(ev.E)

    ax.scatter(Bk_vec , max_Fs_vec, 
               color='k', alpha=0.5, s=8, facecolors='k',
               linewidths=0.25,label='sim')

    style_axis(
        ax,
        xlabel= r"$B \kappa_{\text{0}}$ (mJ)",
        ylabel= r"max$(F \ell_{\text{lev}})$ (mJ)",
        xlim=xlim,
        ylim=ylim,
        xticks=xticks,
        yticks=yticks,
        label_pad=3,
        tick_params={
        'pad': 3,
        }
    )

    if not fit:
        return ax, None
    
    else:
        # Fit max_Fs_vec vs Bk_vec to a specific function
        unique_E = sorted(set(E_vec))
        Bk_by_E = [[] for _ in range(len(unique_E))]
        Fs_by_E = [[] for _ in range(len(unique_E))]

        for bk, fs, e in zip(Bk_vec, max_Fs_vec, E_vec):
            idx = unique_E.index(e)
            Bk_by_E[idx].append(bk)
            Fs_by_E[idx].append(fs)

        Bk_mean = [np.mean(bk_list) for bk_list in Bk_by_E]
        Bk_std = [np.std(bk_list) for bk_list in Bk_by_E]
        Fs_mean = [np.mean(fs_list) for fs_list in Fs_by_E]
        Fs_std = [np.std(fs_list) for fs_list in Fs_by_E]

        # ax.errorbar(Bk_mean, Fs_mean, xerr=Bk_std, yerr=Fs_std, 
        #             fmt='+b', alpha=0.75, capsize=2,label='sim')

        fit_results = fit_w_error(
                                    np.array(Bk_mean), np.array(Bk_std),
                                    np.array(Fs_mean), np.array(Fs_std),
                                    fit_func=fit_func
        )
        popt, pcov, r2, chi_sq_red, chi_sq_std = (fit_results['popt'], fit_results['pcov'], 
                                    fit_results['r2'],\
                                    fit_results['chi_sq_red'], fit_results['chi_sq_std'])

        # fit_curve = [fit_func(x, *popt) for x in Bk_vec]
        # residuals = np.array(max_Fs_vec) - np.array(fit_curve)
        # ss_res = np.sum(residuals ** 2)
        # ss_tot = np.sum((np.array(max_Fs_vec) - np.mean(np.array(max_Fs_vec))) ** 2)
        # r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        
        ax.plot(np.linspace(0, max(Bk_vec), 100),
               [fit_func(x, *popt) for x in np.linspace(0, max(Bk_vec), 100)],
               color='grey',alpha=0.5, linestyle='--',label='fit')
        
        ax.legend(loc='lower right',borderpad=0.05)


        fit_results = {
            'popt': popt,
            'pcov': pcov,
            'r2': r2,
            'chi_sq_red': chi_sq_red,
            'chi_sq_std': chi_sq_std
        }

    return ax, fit_results

def plot_sim_maxF_vs_llev(ax=None, events=None, E=10.0,
                          xlim=None, ylim=None):
    """
    Plot max force vs l_tip for simulation events.
    """
    if events is None:
        print("No events provided.")
        return ax, None, None
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)

    # l_tip_vec = []
    l_lev_vec = []
    max_F_vec = []
    E_target = E  # Example target E value - lowest in sim
    for i, ev in enumerate(events):
        if ev.mu>0 or ev.E != E_target: # focus on one E value
            continue
        # l_tip_vec.append(ev.l_tip/ev.growth_zone) # normalized l_tip
        l_lev_vec.append(ev.l_lev/ev.growth_zone) # normalized l_lev
        max_F_vec.append(max(ev.F_shifted))

    ax.scatter(l_lev_vec , max_F_vec, 
               color='k', alpha=0.5, s=10, facecolors='k',
               linewidths=0.25)
    
    popt, pcov = curve_fit(
        powerfunc, np.array(l_lev_vec), np.array(max_F_vec)
    )
    r2, rmse, _ = calc_R2_RMSE(
        np.array(l_lev_vec), np.array(max_F_vec), popt, powerfunc
    )
    
    x_fit = np.linspace(0.4, max(l_lev_vec), 100)
    ax.plot(x_fit,[powerfunc(x, *popt) for x in x_fit],
           color='grey',alpha=0.85, linestyle='--', 
           label=f'fit')
    
    style_axis(
        ax,
        xlabel= r"$\ell_{\text{lev}}$ / $L_{\text{gz}}$",
        ylabel= r"F$_{\text{max}}$ (mN)",
        xlim=xlim,
        ylim=ylim,
        background_color='0.95',
        label_pad=1,
        nbin_ticks=3,
        # x_formatter='%.1f',
        # y_formatter='%.1f',
        # xticks=np.arange(0, 0.6, 0.25),
        tick_params={
        'pad': 0.75,           
        'length': 0.5,        
        'width': 0.25,  
        'rotation': 25,  
    }
    )

    fit_results = {
        'popt': popt,
        'pcov': pcov,
        'r2': r2,
        'rmse': rmse,
    }

    return ax, fit_results, E_target

def plot_sim_maxF_vs_E(ax=None, events=None, LOC = 20,
                          xlim=None, ylim=None):
    """
    Plot max force vs E for simulation events.
    """
    if events is None:
        print("No events provided.")
        return ax, None, None
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)

    E_vec = []
    max_F_vec = []
    for i, ev in enumerate(events):
        if ev.mu>0 or ev.LOC != LOC: # focus on one LOC value
            continue
        E_vec.append(ev.E) # E (MPa)
        max_F_vec.append(max(ev.F_shifted))

    ax.scatter(E_vec , max_F_vec, 
               color='k', alpha=0.5, s=10, facecolors='k',
               linewidths=0.25)

    popt, pcov = curve_fit(
        linfunc, np.array(E_vec), np.array(max_F_vec)
    )
    r2, rmse, _ = calc_R2_RMSE(
        np.array(E_vec), np.array(max_F_vec), popt, linfunc
    )

    x_fit = np.linspace(0, max(E_vec), 100)
    ax.plot(x_fit,[linfunc(x, *popt) for x in x_fit],
           color='grey',alpha=0.85, linestyle='--', 
           label=f'fit')
    
    style_axis(
        ax,
        xlabel= r"E (MPa)",
        ylabel= r"F$_{\text{max}}$ (mN)",
        label_pad = 1,
        xlim=xlim,
        ylim=ylim,
        # xticks=np.arange(0, 100,25),
        # yticks=np.arange(0, 10,2.5),
        background_color='0.95', # light grey
        nbin_ticks=3,
        tick_params={
        'pad': 1.5,           
        'length': 1,        
        'width': 0.25,
        'rotation': 25,
        }
        )

    fit_results = {
        'popt': popt,
        'pcov': pcov,
        'r2': r2,
        'rmse': rmse,
    }

    return ax, fit_results, LOC


def plot_sim_tslip(events, ax=None, xlim=None, ylim=None):
    """
    Plot t_slip vs time for simulation events.
    """
    colors = [CMAP(i / 3) for i in range(4)]
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)

    for i, ev in enumerate(events):
        if ev.mu > 0:  # skip friction cases
            continue
        t_slip = ev.t_F0
        E = ev.E
        ltip = ev.l_tip/ev.growth_zone
        # Color according to E value
        if E == 10:
            clr = colors[0]
        elif E == 50:
            clr = colors[1]
        elif E == 100:
            clr = colors[2]

        ax.scatter(ltip, t_slip, alpha=0.5, color=clr, linewidth=0.5)

    style_axis(
        ax,
        xlabel= r"$\ell_{o}$ / $L_{\text{gz}}$",
        ylabel= r"$\tau$ / $T_{\text{cn}}$",
        xlim=xlim,
        ylim=ylim,
        # grid={'which': 'both', 'linestyle': '--', 'alpha': 0.5}
    )

    return ax

def plot_sim_FvsFxy_fric(sim_events, ax=None, xlim=None, ylim=None,
                         xticks=None, yticks=None, LOC = 10, E = 10,
                         ):
    """Plot F_vec vs F_vec_planar for frictional simulation events, 
    highlighting difference between frictional and non-frictional cases.
    """
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(4, 3), dpi=300)
    for ev in sim_events:
        if ev.E != E:
            continue
        if ev.LOC != LOC:
            continue
        
        F_vec = ev.F_shifted
        F_xy = ev.F_planar_shifted
        t = ev.T_shifted
        

        if ev.mu == 0:
            ax.plot(t, F_vec, '-', alpha=0.6, color='k', label=r'$\mu=0$',linewidth=2)
            ax.plot(t, F_xy, '--', alpha=0.9, color='k',linewidth=0.5)

        else:
            if ev.mu == 1:
                clr = CMAP(1*ev.mu)
                ax.plot(t, F_vec, '-', alpha=0.6, color=clr, label=rf'$\mu=1$')
                ax.plot(t, F_xy, '--', alpha=0.9, color=clr)

    ax.legend(loc='upper right')

    style_axis(
        ax,
        xlabel=r"$t/T_{\text{cn}}$",
        ylabel="F (mN)",
        xlim=xlim,
        ylim=ylim,
        xticks=xticks,
        yticks=yticks,
    )

    return ax

def plot_all_sim_events(sim_events, ax=None, xlim=None, ylim=None,
                         xticks=None, yticks=None,
                         ):
    """ Plot all simulation events raw and normalized """

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(4, 3), dpi=300)
    for ev in sim_events:
        if ev.mu > 0:
            continue
        E = ev.E
        LOC = ev.LOC
        unique_E = sorted(set([e.E for e in sim_events if e.mu == 0]))
        unique_LOC = sorted(set([e.LOC for e in sim_events if e.mu == 0]))
        color = CMAP((unique_E.index(E)) / (len(unique_E) - 1)) if len(unique_E) > 1 else CMAP(0.5)
        color = darken_if_light(color, threshold=0.7, factor=0.7)
        alpha = (unique_LOC.index(LOC)) / (len(unique_LOC) - 1) if len(unique_LOC) > 1 else 0.5
        F_xy = ev.F_planar_shifted
        t = ev.T_shifted
        ax.plot(t, F_xy, color=color, alpha=alpha,
                label=f'{E:.0f}' if LOC == max(unique_LOC) else None)

    ax.legend(loc='upper right',title='E (MPa)')

    style_axis(
        ax,
        xlabel=r"$t/T_{\text{cn}}$",
        ylabel="F (mN)",
        xlim=xlim,
        ylim=ylim,
        xticks=xticks,
        yticks=yticks,
    )

    return ax