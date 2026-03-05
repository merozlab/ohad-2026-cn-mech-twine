#%%
# %load_ext autoreload
# %autoreload

import bootstrap
import matplotlib.pyplot as plt
import numpy as np

from src.plotting.plot_layout import (
    make_grid_figure,
    make_named_grid,
    add_panel_labels,
)
from src.plotting.grouping_plots import (
    plot_grouped_trajectories,
    plot_group_mean_trajectory,
    plot_grouped_scatter_with_ellipse,
)

from src.analysis.exp_grouping_analysis import (
    extract_E_grouping_data,
    filter_scalar_ragged,
)
from src.utils.array_tools import (
    assign_equal_width_bins,
    assign_percentile_bins)

from src.io.cache import load_snapshot

#%% 
# Load data

plants = load_snapshot("plants", "exp", stage="R_E_fits")
events = load_snapshot("events", "exp", stage="R_E_fits",
                       bind_context=True, plants=plants)

#%% 
# Analysis

df_scalars, data_ragged = extract_E_grouping_data(events)
filter_params = {
    ("contact_E", "max"): 15,
    ("max_F_s", "max"): 0.5
}
df_scalars_filter, data_ragged_filter = filter_scalar_ragged(df_scalars, data_ragged,
                                                     filter_params=filter_params)

#%%
# ---- Create figure ----
fig, grid = make_grid_figure(3, 2, figsize=(10, 12), wspace=0.4)

layout_map = {
    "A": (0, 0), "B": (0, 1),
    "C": (1, 0), "D": (1, 1),
    "E": (2, 0), "F": (2, 1),
    }

axes = make_named_grid(fig, gs=grid, layout_map=layout_map)

# first grouping
nbins = 3
colors = plt.cm.coolwarm(np.linspace(0, 1, nbins))
all_Econ_vals = df_scalars_filter["contact_E"].values
E_groups, E_bins = assign_equal_width_bins(all_Econ_vals, nbins)
E_pct_groups, E_pcts = assign_percentile_bins(all_Econ_vals, nbins)

# --- Raw trajectories
plot_grouped_trajectories(axes["A"], data_ragged_filter["f"],
                           E_groups, colors, bin_values=E_bins, 
                           grouping_units='MPa')
plot_grouped_trajectories(axes["B"], data_ragged_filter["f"], 
                          E_pct_groups, colors, bin_values=E_pcts,
                          grouping_units='MPa')

# --- Mean trajectories
plot_group_mean_trajectory(axes["C"], data_ragged_filter["f"], 
                           E_groups, colors, bin_values=E_bins, 
                           grouping_units='MPa')
plot_group_mean_trajectory(axes["D"], data_ragged_filter["f"], 
                           E_pct_groups, colors, bin_values=E_pcts, 
                           grouping_units='MPa')

# --- F_s vs 2Bkappa

# second grouping
nbins = 10
colors = plt.cm.coolwarm(np.linspace(0, 1, nbins))
all_Econ_vals = df_scalars_filter["contact_E"].values
E_groups, E_bins = assign_equal_width_bins(all_Econ_vals, nbins)
E_pct_groups, E_pcts = assign_percentile_bins(all_Econ_vals, nbins)

plot_grouped_scatter_with_ellipse(
    axes["E"], df_scalars_filter["B_contact_kappa"], df_scalars_filter["max_F_s"], 
    E_groups, colors, 
    bin_values = E_bins, 
    grouping_units='MPa',
    xlabel=r'$B\kappa$ (mJ)',
    ylabel='F * s (mJ)',
    xlim = [0, 0.035],
    ylim = [0, 0.35],
)
plot_grouped_scatter_with_ellipse(
    axes["F"], df_scalars_filter["B_contact_kappa"], df_scalars_filter["max_F_s"], 
    E_pct_groups, colors, 
    bin_values = E_pcts, 
    grouping_units='MPa',
    xlabel=r'$B\kappa$ (mJ)',
    ylabel='F * s (mJ)',
    xlim = [0, 0.035],
    ylim = [0,0.35],
)

add_panel_labels(axes, fontsize=18)

plt.show()

#%%

E_values_largest = df_scalars_filter["max_F_s"].values[E_groups == 6]
print(E_values_largest)
#%%