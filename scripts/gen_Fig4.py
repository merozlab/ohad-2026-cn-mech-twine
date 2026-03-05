"""
Create Figure 4:
    Twining initiation characterization    
"""
# %% imports
# %load_ext autoreload
# %autoreload
import bootstrap
import matplotlib.pyplot as plt
# import numpy as np

from src.io.cache import load_snapshot
from src.io.paths import PROJECT_DATA_PATH
# from src.io.paths import DATA_PATH, H5_RESULTS_PATH
from src.io.data_imports import apply_project_style
apply_project_style()

from src.plotting.primitives import (
    CMAP,
    plot_image,
)

from src.plotting.plot_layout import (
    make_grid_figure,
    make_named_grid,
    add_panel_labels,
    create_inset,
    create_broken_axis,
    adjust_axis_position,
)

from src.analysis.slip_twine_analysis import (
    get_slip_twine_arrays, 
    get_twine_Feff
)
from src.analysis.exp_grouping_analysis import (
    extract_E_grouping_data, 
    filter_scalar_ragged
)
from src.plotting.slip_twine_plots import (
    twine_ratio_Feff,
    plot_slip_twine_time_hist,
    plot_slip_twine_lo_hist,
    plot_time_vs_ltip,
    plot_slip_twine_torque_hist,
    plot_slip_twine_curvature_hist,
    plot_motor_twine,
    ltip_tcont_multi_plot,
    plot_motor_twine_lin,
)

#%% load and process data
# ---- Load data ----

# experimental data with slip-twine
exp_plants = load_snapshot("plants", "exp", stage="slip-twine")
exp_events = load_snapshot("events", "exp", stage="slip-twine",
                           bind_context=True, plants=exp_plants)
state, slip, twine = get_slip_twine_arrays(exp_events)

# extreme mass data
# xtrm_plants = load_snapshot("plants", "xtrm", stage="build") # 
xtrm_events = load_snapshot("events", "xtrm", stage="build") # 
Feff, twine_states, ltips = get_twine_Feff(exp_events, xtrm_events)

# motor-modified CN data
motor_plants = load_snapshot("plants", "motor", stage="build") #
motor_events = load_snapshot("events", "motor", stage="build",
                             bind_context=True, plants=motor_plants) #

df_scalars, data_ragged = extract_E_grouping_data(exp_events, force_stop_ind=100)
filter_params = {
    ("contact_E", "max"): 20,
    ("max_F_s", "max"): 0.75
}
df_scalars_filter, data_ragged_filter = filter_scalar_ragged(df_scalars, data_ragged,
                                                     filter_params=filter_params)
# B_vals = df_scalars_filter["B_contact"].values
#%% create figure
# ---- Create figure ----
# make a 2x3 grid with equal sized panels
# ---- Plot panels ----
#   A                                   B                   C
#   Twine ratio (Fresist)          max-Torque(dist)     Slip-Twine(lo dist)
#   D                                   E                   F
#   Slip-Twine(tau dist) + log-log    Motor schematic      Motor twine plot


fig, grid = make_grid_figure(
    2, 3,
    figsize=(7, 7/1.5),
    wspace=0.35,
    hspace=0.3,
)

layout_map = {
    "A": (0, 0),
    "B": (0, 1),
    "C": (0, 2),
    "D": (1, 0),
    "E": (1, 1),
    "F": (1, 2),
}

axes = make_named_grid(fig, gs=grid, layout_map=layout_map)

# panels

# ----- A - twine ratio vs Feff -----
bins_num = 9 # equal_width / percentile
axA, bins, twine_ratio, twine_counts, total_counts, N_inf = twine_ratio_Feff(ax=axes["A"], 
                    Feff=Feff, twine_states=twine_states, nbins=bins_num-2,
                    xlabel=r'$F_{\text{res}}$ (mN)', ylim=(0,0.85),
                    xticks=[0,10,20,30],
                    ylabel='twine ratio', yticks=[0,0.2,0.4,0.6,0.8],
                    bin_mode='percentile', size=15, color='r',marker='o', 
                    infcolor='r', label_inf=None,
                    )
axA_inset = create_inset(axes["A"], width="40%", height="40%", loc='lower right',
                            bbox_to_anchor=(0, 0.1, 1, 1), 
                            bbox_transform=axes["A"].transAxes)
ltip_threshold = 1 # 1.5
L_mask_over = [ltip>ltip_threshold for ltip in ltips]
axA_inset, bins, twine_ratio, twine_counts, total_counts_up, N_inf = twine_ratio_Feff(
                    ax=axA_inset, Feff=Feff, twine_states=twine_states, nbins=bins_num,
                    xlabel=None, 
                    ylabel=None,
                    bin_mode='percentile', mask=L_mask_over,
                    color='k', infcolor='k', size=2, marker='s',
                    label_inf=None,
                    )
L_mask_under = [ltip<=ltip_threshold for ltip in ltips]
axA_inset, bins, twine_ratio, twine_counts, total_counts_down, N_inf = twine_ratio_Feff(
                    ax=axA_inset, Feff=Feff, twine_states=twine_states, nbins=bins_num,
                    xlabel=None, ylim=(0,1.05),
                    ylabel=None, 
                    xticks=[0,10,20,30],
                    yticks=[0,0.2,0.4,0.6,0.8,1],
                    color='gray', infcolor='gray',size=2, marker='^',
                    bin_mode='percentile', mask=L_mask_under,
                    label_inf=None,
                    )
axA.legend(loc='center right', frameon=False)
# axA.legend([r'$\ell_{\text{o}} > 1$', r'$F_{\text{res}}\rightarrow\infty$',
#            r'$\ell_{\text{o}}\leq 1$', r'$F_{\text{res}}\rightarrow\infty$'],
#            loc= 'center right', frameon=True,
#            )

# ----- B - slip-twine vs torque distribution -----

axB = plot_slip_twine_torque_hist(ax=axes["B"], slip=slip, twine=twine, nbins=15,
                                  threshold=1.0, yticks=[0, 0.1, 0.2],
                                  )

# ----- C - slip-twine vs lo distribution -----
axC = plot_slip_twine_lo_hist(ax=axes["C"], slip=slip, twine=twine, 
                    xlabel=r'$\ell_{\text{o}}$ (cm)',
                    threshold = 7, nbins = 15,
                    yticks=[0,0.1,0.2],
                    )


# ----- D - slip-twine -----
axD = plot_slip_twine_time_hist(ax=axes["D"], slip=slip, twine=twine, 
                    xlabel=r'$\tau/T_{\mathrm{CN}}$',
                    threshold = 2.5, nbins = 12,
                    xticks=[0,0.5,1.0,1.5,2.0,2.5],
                    yticks=[0, 0.1, 0.2,0.3],
                    )
axD_inset = create_inset(axD, width="40%", height="40%", 
                         loc='upper right',borderpad=0.05, bbox_to_anchor=(0, 0, 1, 1), 
                         bbox_transform=axD.transAxes)

axD_inset, fit_results = plot_time_vs_ltip(ax = axD_inset, slip=slip, twine=twine, 
                    xlabel=r'$\tau/T_{\mathrm{CN}}$',
                    ylabel=r'$\ell_{\mathrm{o}}$ (cm)',
                    t_threshold=2.5, ltip_threshold=6,
                    log_log_scale=True,fit_line=True)

# ----- E insert image of motor schematic -----
motor_path = PROJECT_DATA_PATH / "images"/"motor"/"motor_schematic3.png"
axE = plot_image(axes["E"], motor_path, shift_x=-0.025, shift_y=-0.5)
adjust_axis_position(axE,scale_height=1.1, scale_width=1.1,dx=-0.03, dy=-0.025)

# ----- F motor twine and fit -----
# axes["F"].axis('off')
axF = create_broken_axis(fig=fig, subplot_loc=grid[1,2], ylims=((0, 400), (900, 1200)), 
                    hspace=0.15, height_ratios=[1, 4], d=0.005, supress_ticks=True,
                    ytick_spacing=[100,100], origin_ax=axes["F"])
axF,fit_one_over = plot_motor_twine(ax=axF, 
                    motor_events=motor_events, exp_twine_events=twine, 
                    fit_line=True, xlabel='f (rph)', ylabel=r'$\tau$ (min)')

letter_labels = layout_map.keys()
add_panel_labels(axes,labels=letter_labels)
add_panel_labels(axF.axs[0], labels=['F'])

# Save figure
save_Fig4_path = PROJECT_DATA_PATH / "test_figures" / "Fig4.svg"
plt.savefig(save_Fig4_path, format="svg", dpi=300)
plt.show()

#%% SM

ax,slip_curvature,twine_curvature = plot_slip_twine_curvature_hist(
                    arr=df_scalars_filter,
                    xlim=(0, 1), ylim=(0, 0.3), nbins=20)

fig, ax_main, ax_top, ax_right = ltip_tcont_multi_plot(slip=slip,twine=twine)




#%% tests
bins_num = 10
axA, bins, twine_ratio, twine_counts, total_counts, N_inf = twine_ratio_Feff( 
                    Feff=Feff, twine_states=twine_states, nbins=bins_num,
                    xlabel=r'$F_{\text{res}}$ (mN)',
                    ylabel='twine ratio', yticks=[0,0.2,0.4,0.6,0.8,1],
                    bin_mode='equal_width',
                    )
axA, bins, twine_ratio, twine_counts, total_counts, N_inf = twine_ratio_Feff(ax=axA,
                    Feff=Feff, twine_states=twine_states, nbins=bins_num,
                    xlabel=r'$F_{\text{res}}$ (mN)',
                    ylabel='twine ratio', yticks=[0,0.2,0.4,0.6,0.8,1],
                    bin_mode='percentile', color='red',
                    )
#%% test percentile binning for Fres with separation by ltip
bins_num = 9 # equal_width / percentile
ltip_threshold = 1 # 1.5
L_mask_over = [ltip>ltip_threshold for ltip in ltips]
axA, bins, twine_ratio, twine_counts, total_counts_up, N_inf = twine_ratio_Feff( 
                    Feff=Feff, twine_states=twine_states, nbins=bins_num,
                    xlabel=r'$F_{\text{res}}$ (mN)', 
                    ylabel='twine ratio', yticks=[0,0.2,0.4,0.6,0.8,1],
                    bin_mode='percentile', mask=L_mask_over,
                    color='red', infcolor='salmon',
                    )
L_mask_under = [ltip<=ltip_threshold for ltip in ltips]
axA, bins, twine_ratio, twine_counts, total_counts_down, N_inf = twine_ratio_Feff(ax=axA,
                    Feff=Feff, twine_states=twine_states, nbins=bins_num,
                    xlabel=r'$F_{\text{res}}$ (mN)', ylim=(0,1.1),
                    ylabel='twine ratio', 
                    xticks=[0,5,10,15,20,25,30,35],
                    yticks=[0,0.2,0.4,0.6,0.8,1],
                    color='blue', infcolor='lightblue',
                    bin_mode='percentile', mask=L_mask_under,
                    )
axA.legend([r'$\ell_{\text{o}} > 1$',r'$F_{\text{res}}\rightarrow\infty$',
           r'$\ell_{\text{o}}\leq 1$',r'$F_{\text{res}}\rightarrow\infty$'],
           loc= 'lower right', frameon=True,
           )
#%% motor lin fit
ax, fit_func_normmotor = plot_motor_twine_lin(motor_events=motor_events, 
                     exp_twine_events=twine, fit_line=True, 
                     xlabel=r'$T_{\text{rot}}$ (min)', ylabel=r'$\tau/T_{\text{rot}}$')
print(fit_func_normmotor)

#%%