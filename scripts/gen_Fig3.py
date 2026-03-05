"""
Create Figure 3:
    Mechanics comparison between simulation and experiment
    for Young's modulus and ltip groupings.     
"""
#%% Imports

# %load_ext autoreload
# %autoreload
import bootstrap
import matplotlib.pyplot as plt
import numpy as np

from src.io.cache import load_snapshot
from src.io.paths import PROJECT_DATA_PATH
from src.io.data_imports import apply_project_style
apply_project_style()

from src.analysis.exp_grouping_analysis import (
    extract_E_grouping_data,
    filter_scalar_ragged,
)

from src.plotting.plot_layout import (
    make_grid_figure,
    make_named_grid,
    add_panel_labels,
    create_inset,
)
from src.plotting.grouping_plots import (
    plot_group_mean_trajectory,
    plot_grouped_scatter_with_ellipse,

)
from src.plotting.sim_plots import (
    plot_sim_F_t_constpar,
    plot_sim_F_t_constE_varL,
    plot_sim_maxF_vs_E,
    plot_sim_maxF_vs_llev,
    plot_sim_FsBk,
    plot_sim_torque,
    # plot_sim_maxF_vs_ltip,
)

from src.utils.array_tools import (
    assign_equal_width_bins,
    assign_percentile_bins,
    merge_groups_N_bins,

)
from src.utils.curve_tools import (
    linfunc,
    multi_range_fit,

)

#%% load data
# Load data
exp_plants = load_snapshot("plants", "exp", stage="R_E_fits") # experimental data after R and E fitting
exp_events = load_snapshot("events", "exp", stage="R_E_fits",plants=exp_plants,bind_context=True) # experimental data after R and E fitting
sim_events = load_snapshot("events", "sim", stage="sine_fit") # simulation data after sine fitting
for sm_ev in sim_events:
    sm_ev.calc_sim_vars()
#%% filter and grouping
# Experimental data filter

df_scalars, data_ragged = extract_E_grouping_data(exp_events, force_stop_ind=100)
filter_params = {
    ("contact_E", "max"): 20,
    ("max_F_s", "max"): 0.75
}
df_scalars_filter, data_ragged_filter = filter_scalar_ragged(df_scalars, data_ragged,
                                                     filter_params=filter_params)
all_Econ_vals = df_scalars_filter["contact_E"].values
all_llev_vals = df_scalars_filter["l_lev"].values

merge_key = {0: [0,1,2], 1: [3,4,5,6,7], 2: [8,9]}

# first grouping (for sub-Figure C) - Young's modulus
nbins_sFigB = 10
E_groups_sFigB, E_bins_sFigB = assign_equal_width_bins(all_Econ_vals, nbins_sFigB)
E_pct_groups_sFigB, E_pcts_sFigB = assign_percentile_bins(all_Econ_vals, nbins_sFigB)
E_pct_groups_sFigB, E_pcts_sFigB = merge_groups_N_bins(
                    E_pct_groups_sFigB, E_pcts_sFigB, merge_map=merge_key)
colors_sFigB = plt.cm.coolwarm(np.linspace(0, 1, max(E_pct_groups_sFigB)+1))


# second grouping (for sub-Figure E) - llev
nbins_sFigD = 10
llev_groups_sFigD, llev_bins_sFigD = assign_equal_width_bins(all_llev_vals, nbins_sFigD)
llev_pct_groups_sFigD, llev_pcts_sFigD = assign_percentile_bins(all_llev_vals, nbins_sFigD)
merge_key = {0: [0,1,2], 1: [3,4,5,6,7], 2: [8,9]}
llev_pct_groups_sFigD, llev_pcts_sFigD = merge_groups_N_bins(
                    llev_pct_groups_sFigD, llev_pcts_sFigD, merge_map=merge_key)
colors_sFigD = plt.cm.coolwarm(np.linspace(0, 1, max(llev_pct_groups_sFigD)+1))


# third grouping (for sub-Figure G) - Young's modulus
nbins_sFigF = 10
E_groups_sFigF, E_bins_sFigF = assign_equal_width_bins(all_Econ_vals, nbins_sFigF)
E_pct_groups_sFigF, E_pcts_sFigF = assign_percentile_bins(all_Econ_vals, nbins_sFigF)
merge_key = {0: [0,1], 1: [2,3], 2: [4,5], 3: [6,7], 4: [8], 5: [9]} # current
# merge_key = {0: [0,1,2],1:[3],2:[4],3:[5],4:[6],5:[7],6:[8,9]} # test
E_pct_groups_sFigF, E_pcts_sFigF = merge_groups_N_bins(
                    E_pct_groups_sFigF, E_pcts_sFigF, merge_map=merge_key)
colors_sFigF = plt.cm.coolwarm(np.linspace(0, 1, max(E_pct_groups_sFigF)+1))


#%% create figure v2
# ---- Create figure ----
# make a grid with a 2x3 grid

fig, grid = make_grid_figure(
    2, 3,
    figsize=(7, 7/1.5), # 7 , 5
    wspace=0.35,
    hspace=0.325,
)

layout_map = {
    "A": (0, 0),
    "B": (1, 0),
    "C": (0, 1),
    "D": (1, 1),
    "E": (0, 2),
    "F": (1, 2),
}

axes = make_named_grid(fig, gs=grid, layout_map=layout_map)

# ---- Plot panels ----
# A : Simulation schematic (external)


# Predictions (simulations): B                           D           F
#                            F(E)+Fmax(E)+linear fit   F(llev)     Fs (Bk)
# Experiments:               C                           E           G
#                           F(E)                       F(llev)+Fmax(lo?)     Fs (Bk) + 2 linear fits

# A: F (E) simulations
# main - F for diff E 
axA = plot_sim_F_t_constpar(sim_events, friction=False, 
                            constpar='l_lev', const_value_rank=1, varypar='E',
                            legend_loc=(0.61,0.43), 
                            xlim=(0, 0.75), ylim=(0, 17), ax=axes["A"],
                            legend_on=True, legend_title=r'', #E (MPa)
                            legend_units='',
                            ) 

axA_inset = create_inset(axes["A"], width="40%", height="40%", loc='upper right',
                            bbox_to_anchor=(0, 0, 1, 1), 
                            bbox_transform=axes["A"].transAxes)
axA, fit_resultsFE, E_F = plot_sim_maxF_vs_E(ax=axA_inset, events=sim_events, LOC=20,
                            ylim=(0,10), xlim=(0, 100),
                            )
# B: F (E) experiments
axB = plot_group_mean_trajectory(ax=axes["B"], trajectories=data_ragged_filter["f"], 
                            groups=E_pct_groups_sFigB, colors=colors_sFigB, 
                            bin_values=E_pcts_sFigB, 
                            group_name='E', grouping_units='MPa',
                            bin_format=".0f",
                            xlim=(0, 110), ylim=(0,0.6), 
                            # yticks=[0,0.2,0.4,0.6,0.8,1.0,1.2],
                            # legend_params={"loc":'center left'},
    )

axB_inset = create_inset(axes["B"], width="40%", height="40%", loc='lower right',
                            bbox_to_anchor=(0, 0.175, 1, 1),  
                            bbox_transform=axes["B"].transAxes)

nbins_E = 10
all_Econ_vals = df_scalars_filter["contact_E"].values
E_pct_groups, E_pcts = assign_percentile_bins(all_Econ_vals, nbins_E)
colors_E = plt.cm.coolwarm(np.linspace(0, 1, max(E_pct_groups)+1))
axB_inset = plot_grouped_scatter_with_ellipse(ax=axB_inset,
                            y=df_scalars_filter["F_amp"], 
                            x=df_scalars_filter["contact_E"], 
                            groups = E_pct_groups, colors = colors_E, 
                            # bin_values = E_pcts,
                            bin_values=[],
                            grouping_units='MPa',
                            ylabel=r'$F_{\text{max}}$ (mN)',
                            xlabel=r'$E$ (MPa)',
                            xlim = [-1, 13],
                            ylim = [0.1,0.8],
                            # xticks = [0, 5, 10],
                            # yticks = [0,0.2,0.4,0.6,0.8],
                            plot_points=False,
                            )                        

# C: F (llev) simulations
axC = plot_sim_F_t_constE_varL(sim_events, friction=False, 
                            constpar='E', const_value_rank=0, varypar='l_lev',
                            xlim=(0, 0.75), ylim=(0,3), ax=axes["C"],
                            legend_on=True, legend_loc=(0.61,0.43), 
                            legend_title=r'', val_format=".1f", #$\ell_{\text{o}}$ (cm)
                            normalizer='growth_zone', # max_varypar=-30,
                            varypar_values=[0,2,4], yticks=[0,1,2,3]
                            )

axC_inset = create_inset(axes["C"], width="40%", height="40%", loc='upper right',
                            bbox_to_anchor=(0, 0, 1, 1), 
                            bbox_transform=axes["C"].transAxes)
axC_inset, fit_resultsFllev, E_Fllev = plot_sim_maxF_vs_llev(ax=axC_inset, 
                            events=sim_events, E=10.0,
                            # xlim=(0, 0.6), ylim=(0.75, 1.5),
                            )


# D: F (llev) experiments
axD = plot_group_mean_trajectory(ax=axes["D"], trajectories=data_ragged_filter["f"], 
                            groups=llev_pct_groups_sFigD, colors=colors_sFigD, 
                            bin_values=llev_pcts_sFigD,
                            group_name=r'$\ell_{\text{lev}}$', 
                            grouping_units='cm', bin_format=".0f",
                            xlim=(0, 110), ylim=(0,0.6), 
                            # yticks=[0,0.2,0.4,0.6,0.8,1.0,1.2],
                            # legend_params={"loc":'center left'},
                            )
axD_inset = create_inset(axes["D"], width="40%", height="40%", loc='lower right',
                            bbox_to_anchor=(0, 0.175, 1, 1), 
                            bbox_transform=axes["D"].transAxes)
nbins_ltip = 10
all_llev_vals = df_scalars_filter["l_lev"].values
llev_pct_groups, llev_pcts = assign_percentile_bins(all_llev_vals, nbins_ltip)
colors_llev = plt.cm.coolwarm(np.linspace(0, 1, max(llev_pct_groups)+1))
axD_inset = plot_grouped_scatter_with_ellipse(ax=axD_inset,
                            y=df_scalars_filter["F_amp"], 
                            x=df_scalars_filter["l_lev"], 
                            groups = llev_pct_groups, colors = colors_llev, 
                            bin_values=[],
                            grouping_units='cm', group_name=r'$\ell_{\text{lev}}$',
                            ylabel=r'$F_{\text{max}}$ (mN)',
                            xlabel=r'$l_{\text{lev}}$ (cm)',
                            xlim = [25, 65],
                            ylim = [0.2,0.7],
                            # xticks = [0, 5, 10, 15],
                            # yticks = [0,0.2,0.4,0.6,0.8,1.0],
                            plot_points=False,
                            )

# E: Fs (Bk) simulations 
axE, fit_results_simFsBk = plot_sim_FsBk(ax=axes["E"], events=sim_events, 
                            fit_func=linfunc, ylim=(0, 0.5), xlim=(0, 0.2),
                            xticks=[0,0.05,0.1,0.15,0.2], yticks=[0, 0.1, 0.2, 0.3, 0.4])


# Force for diff llev 
axE_inset = create_inset(axE, width="40%", height="40%",loc='upper left',
                            bbox_to_anchor = (0, 0, 1, 1), # x, y, width, height
                            bbox_transform = axE.transAxes,
                            )
axE_inset = plot_sim_torque(sim_events, friction=False, 
                            xlim=(0, 0.5), 
                            ax=axE_inset, legend_on=False)

# F: Fs (Bk) experiments
axF = plot_grouped_scatter_with_ellipse(
                            axes["F"], df_scalars_filter["B_contact_kappa"], 
                            df_scalars_filter["max_F_s"], 
                            E_pct_groups_sFigF, colors_sFigF, 
                            bin_values = E_pcts_sFigF,
                            grouping_units='MPa',
                            xlabel=r'$B\kappa_{0}$ (mJ)',
                            ylabel=r'max$(F\ell_{\text{lev}})$ (mJ)',
                            xlim = [-0.0025, 0.031],
                            ylim = [0,0.275],
                            xticks = [0, 0.01, 0.02, 0.03],
                            yticks = [0, 0.1, 0.2,0.3],
                            )
# fit ranges in G
fit_results_expFsBk = multi_range_fit(
                            x=df_scalars_filter["B_contact_kappa"].values,
                            y=df_scalars_filter["max_F_s"].values,
                            dx=np.full(len(df_scalars_filter), 0.005),
                            dy=np.full(len(df_scalars_filter), 0.05), 
                            fitwerr=True, ranges=[(0.001, 0.0135)], #(0, 0.001),(0.0045,0.04)
                            plot = True, ax=axes["F"],
                            )

letter_labels = layout_map.keys()
add_panel_labels(axes,labels=letter_labels)

# Save figure
save_Fig3_path = PROJECT_DATA_PATH / "test_figures" / "Fig3.svg"
plt.savefig(save_Fig3_path, format="svg", dpi=300)
plt.show()
#%% SM
# Fmax ltip/L - sim
plot_sim_maxF_vs_ltip(sim_events, ax=None, E=10.0, xlim=(0, 0.6), ylim=(0.75, 1.5))
# Fmax E - exp
# 1. fit mean F of group to sine and get amp
# 2. get max of mean F of group
# Fmax ltip - exp
# same as for E with ltip groups

# ltip vs tau/Tcn exp +: where to cutoff?
# cantilever analysis plot? tau = 1.14*ltip^2
# Fits: ltip = 1.85 (t/Tcn)^0.49? 

# ---- simulation plot options  ----
# plot_sim_R2_distribution(sim_events)
# plot_normalized_sim_trajectories(sim_events, show_trj=True)
# plot_sim_Fvec_vs_Fplanar(sim_events, indices=[7,11,14])
# plot_sim_maxF_vs_ltip(sim_events)


#%% binning for Fitted force maximum (using fitted sine amplitude as proxy for max force)
nbins_E = 15
all_Econ_vals = df_scalars_filter["contact_E"].values
E_pct_groups, E_pcts = assign_percentile_bins(all_Econ_vals, nbins_E)
colors_E = plt.cm.coolwarm(np.linspace(0, 1, max(E_pct_groups)+1))
ax = plot_grouped_scatter_with_ellipse(
                            y=df_scalars_filter["F_amp"], 
                            x=df_scalars_filter["contact_E"], 
                            groups = E_pct_groups, colors = colors_E, 
                            bin_values = E_pcts,
                            grouping_units='MPa',
                            ylabel=r'$A$ (mN)',
                            xlabel=r'$E_{\text{con}}$ (MPa)',
                            xlim = [0, 16],
                            ylim = [0,1.1],
                            xticks = [0, 5, 10, 15],
                            yticks = [0,0.2,0.4,0.6,0.8,1.0]
                            )

nbins_ltip = 15
all_llev_vals = df_scalars_filter["l_lev"].values
llev_pct_groups, llev_pcts = assign_percentile_bins(all_llev_vals, nbins_ltip)
colors_llev = plt.cm.coolwarm(np.linspace(0, 1, max(llev_pct_groups)+1))
ax = plot_grouped_scatter_with_ellipse(
                            y=df_scalars_filter["F_amp"], 
                            x=df_scalars_filter["l_lev"], 
                            groups = llev_pct_groups, colors = colors_llev, 
                            bin_values = llev_pcts,
                            grouping_units='cm', group_name=r'$\ell_{\text{lev}}$',
                            ylabel=r'$A$ (mN)',
                            xlabel=r'$l_{\text{lev}}$ (cm)',
                            xlim = [30, 65],
                            ylim = [0,1.1],
                            # xticks = [0, 5, 10, 15],
                            yticks = [0,0.2,0.4,0.6,0.8,1.0]
                            )

all_ltip_vals = df_scalars_filter["l_tip"].values
nbins_ltip = 15
ltip_pct_groups, ltip_pcts = assign_percentile_bins(all_ltip_vals, nbins_ltip)
# merge_key = {0: [0], 1: [1,2], 2: [3,4], 3: [5,6,7], 4: [8,9]}
# ltip_pct_groups, ltip_pcts = merge_groups_N_bins(
#                     ltip_pct_groups, ltip_pcts, merge_map=merge_key)
colors_ltip = plt.cm.coolwarm(np.linspace(0, 1, max(ltip_pct_groups)+1))

ax = plot_grouped_scatter_with_ellipse(
                            y=df_scalars_filter["F_amp"], 
                            x=df_scalars_filter["l_tip"], 
                            groups = ltip_pct_groups, colors = colors_ltip, 
                            bin_values = ltip_pcts,
                            grouping_units='cm', group_name=r'$l_{\text{o}}$',
                            ylabel=r'$A$ (mN)',
                            xlabel=r'$l_{\text{tip}}$ (cm)',
                            xlim = [0,8],
                            ylim = [0,1.1],
                            xticks = [0, 2,4,6,8,10],
                            yticks = [0,0.2,0.4,0.6,0.8,1.0]
                            )

for group_id in np.unique(ltip_pct_groups):
    group_size = np.sum(ltip_pct_groups == group_id)
    print(f"Group {group_id}: {group_size} samples")

#%%
