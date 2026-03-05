'''
Create Figure 1:
    Setup and forces  
     - A. Setup schematic + inset of side view images from 3 time points 
     - B. Several raw force trajectories with fitted sine curve  
     - C. Smoothed and normalized force trajectories with mean and std curves + normalized simulation curves
     
'''
# %% imports
# %load_ext autoreload
# %autoreload
# from IPython.display import display # to display figures inline
# display(fig) # use this to display figures interactively
# %matplotlib widget # another option for interactive figures

import bootstrap
import matplotlib.pyplot as plt

from src.io.paths import PROJECT_DATA_PATH
from src.io.cache import load_snapshot

from src.plotting.primitives import (
    plot_image,
)

from src.plotting.plot_layout import (
    make_grid_figure,
    make_named_grid,
    add_panel_labels,
    create_inset,
)

from src.plotting.exp_sine_plots import (
    plot_exmp_trj_w_sine_fit,
    plot_exp_mean_trj,
    plot_exp_sine_R2_dist,
    plot_all_events,
)
from src.plotting.sim_plots import (
    plot_normalized_sim_trj,
    plot_pure_sine,
    plot_sim_FvsFxy_fric,
    plot_all_sim_events,
)

from src.io.data_imports import apply_project_style
apply_project_style()

#%% load data
# --------- load data for Fig1 ---------
exp_plants = load_snapshot("plants","exp",stage="curvature")
exp_events = load_snapshot("events","exp",stage="sine_fit", 
                           bind_context=True, plants=exp_plants)
sim_events = load_snapshot("events","sim",stage="sine_fit")

#%% create figure 1
# create figure and grid
fig, grid = make_grid_figure(1, 3, figsize=(7, 2), 
                            wspace=0.3, hspace=0.3,
                            )
layout_map = {
    "A": (0,0),
    "B": (0,1),
    "C": (0,2),
}
axes = make_named_grid(fig, gs = grid, layout_map=layout_map)
add_panel_labels(axes, labels=["A", "B", "C"], 
            weight='bold', loc='upper left')

# plot panels
# panel A : setup schematic + inset of side view images from 3 time points
schematic_path = PROJECT_DATA_PATH / "images" / "pend_schem" /"pend_scheme_barak1.jpg"
axA = plot_image(axes["A"], file_path=schematic_path,
                 preserve_aspect=True, shift_x=0.1, shift_y=-0.05)

# panel B : Several raw force trajectories with fitted sine curve
indxs = [67, 93, 87, 150, 115, 142, 57, 5, 133, 72]
axB, indxs = plot_exmp_trj_w_sine_fit(exp_events, ax=axes["B"], indxs=indxs,
                        xlim=[0,75], ylim=[0,1.2],
                        xticks=[0,20,40,60], 
                        yticks =[0.0,0.2,0.4,0.6,0.8,1.0],
                        )

# panel C : Smoothed and normalized force trajectories with mean and std curves + normalized simulation curves
axC = plot_exp_mean_trj(exp_events, ax=axes["C"])
axC = plot_normalized_sim_trj(sim_events, ax=axC, show_trj=False)
axC = plot_pure_sine(ax=axC, x_cutoff=0.4,
                     xlim=[0,0.5], ylim=[0,1.2], 
                     xticks=[0,0.1,0.2,0.3,0.4,0.5], 
                     yticks=[0,0.2,0.4,0.6,0.8,1.0])

# Get only handles and labels that have actual labels (not None or empty)
handles, labels = axC.get_legend_handles_labels()
# Filter out entries where label is None or starts as '_' , & reverse order
filtered = [(h, l) for h, l in zip(handles, labels) if l and not l.startswith('_')][::-1] 
if filtered:
    handles, labels = zip(*filtered)
#     axC.legend(handles, labels, loc='center left',
#                bbox_to_anchor=(0.01, 0.75),
#                bbox_transform=axC.transAxes, 
# )
    axC.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.05, 1.05),
                bbox_transform=axC.transAxes,
    )

axC_inset = create_inset(axes["C"], width="50%", height="50%", 
                         loc='lower right',
                         borderpad=0, bbox_to_anchor=(-0.015, -0.0, 1, 1), 
                         bbox_transform=axC.transAxes,
)   
cantiliver_path = PROJECT_DATA_PATH / "images" / "cantilever" / "cantilever_schematic_crop.png"
axC_inset = plot_image(axC_inset, file_path=cantiliver_path, 
                 preserve_aspect=True, shift_x=-0.1, shift_y=-0.0,zorder=-1, alpha=0.9)

# save Fig1
save_Fig1_path = PROJECT_DATA_PATH / "test_figures" / "Fig1.svg"
plt.savefig(save_Fig1_path, format="svg", dpi=300)
# plt.close(fig)
# plt.show(save_Fig1_path)

#%% Supplementary plots

ax = plot_exp_sine_R2_dist(exp_events, bins = 70,
                         ax=None,
                         xlim=[0.8, 1],
                         xticks=[0.8,0.9,1], yticks=[0, 0.2, 0.4, 0.6],
                         )

ax = plot_all_events(exp_events,
                    xlim=[0,150],
                    xticks=[0,50,100,150],
                    yticks =[0.0,1,2,3]
                    )


ax = plot_sim_FvsFxy_fric(sim_events, ax=None,
                        LOC = 10,
                        xlim=[0,0.6], ylim=[0,1.5],
                        xticks=[0,0.2,0.4,0.6], yticks=[0,0.5,1,1.5,2],
                        )

ax = plot_all_sim_events(sim_events,
                         xlim=[0,0.5], xticks=[0,0.2,0.4],
                         ylim=[0,12], yticks=[0,5,10],
                         )
#%%