"""
Create Figure 2:
    Material properties and distributions
     - A. R(L-s) with fit and inset histogram with a Contact R histogram
     - B. E(L-s) with fit and inset histogram with a Contact E histogram
     - C. l_tip histogram
     - D. L_base histogram
     - E. T_cn histogram
     - F. Max curvature histogram
     * merge R and E to 1 plot with 2 axis, above it place R(s), below them Rcont and Econt distributions?
"""
#%% Imports
# %load_ext autoreload
# %autoreload

import bootstrap
import matplotlib.pyplot as plt

from src.io.paths import PROJECT_DATA_PATH
from src.io.cache import load_snapshot, save_snapshot
from src.io.data_imports import apply_project_style
apply_project_style()

from src.plotting.primitives import (
    plot_image,

)
from src.plotting.plot_layout import (
    make_grid_figure,
    make_named_grid,
    add_panel_labels,
    create_inset,
    make_named_subfig_grid,
    adjust_axis_position,
)

from src.plotting.exp_param_plots import (
    plot_R_vs_Ls,
    plot_E_vs_Ls,
    plot_distribution,
    legend_for_R_E
)


from src.analysis.exp_param_analysis import collect_scalar_distributions

#%% Load data
exp_plants = load_snapshot("plants", "exp", stage="R_E_fits")
exp_events = load_snapshot("events", "exp", stage="R_E_fits", plants=exp_plants, bind_context=True)
#%% Analysis
# allR,allE = collect_radius_E_by_section(exp_events)
# fitR,fitE = assign_contact_material_props(exp_plants,exp_events, min_count=2)

fitR, fitE = exp_plants[1].R_fit_results, exp_plants[1].E_fit_results
data = collect_scalar_distributions(exp_plants, exp_events)

plants_data = {
    "R": data["R"],
    "E": data["E"],
    "T_cn": data["T_cn"],
    "max_curvature": data["max_curvature"],
}

events_data = {
    "L_base": data["L_base"],
    "contact_E": data["contact_E"],
    "contact_R": data["contact_R"],
    "ltip": data["ltip"],
}
#%% Save event analysis
# save_snapshot(exp_events, "events", "exp", stage="R_E_fits")

#%% Fig2 v1
fig, grid = make_grid_figure(layout = "main_2x3", ncols=3, nrows=2)

layout_map = {
    "A": (0, 0),
    "B": (1, 0),
    "C": (0, 1),
    "D": (1, 1),
    "E": (0, 2),
    "F": (1, 2),
}

axes = make_named_grid(fig, gs=grid, layout_map=layout_map)

nbins = 12

# Plot panels
# Panel A: R(L-s) with fit and inset
axA = plot_R_vs_Ls(
    axes["A"],
    R_stats=fitR,
    contact_R=events_data["contact_R"],
)

# Panel B: E(L-s) with fit and inset
axB = plot_E_vs_Ls(
    axes["B"],
    E_stats=fitE,
    contact_E=events_data["contact_E"],
)

# Panel C: Lo histogram
axC = plot_l_tip_hist(axes["C"], events_data["ltip"], bins=nbins)

# # Panel D: L_base histogram
axD = plot_L_base_hist(axes["D"], events_data["L_base"], bins=nbins)

# # Panel E: T_cn histogram
axE = plot_T_cn_hist(axes["E"], plants_data["T_cn"], bins=nbins)

# # Panel F: Max curvature histogram
axF = plot_max_curvature_hist(axes["F"], plants_data["max_curvature"], bins=nbins)

# Save figure
save_Fig2_path = PROJECT_DATA_PATH / "test_figures" / "Fig2.svg"
plt.savefig(save_Fig2_path, format="svg", dpi=300)
plt.show()

#%% Fig2 v2
# Fig2 v2
# set left 2 columns to match ratio of R(S) image
Rs_image_path = PROJECT_DATA_PATH / "images" / "stem_img.jpg"
Rs_ratio = compute_image_height_ratio(Rs_image_path)
nbins = 12

fig, sL, sR = make_two_subfig_layout(
    figsize=(12, 8),
    subfig_width_ratios=(2.5, 1.5),  # left ~62%, right ~38%
    subfig_wspace=0.08
)

# ---- LEFT ----
gsL = sL.add_gridspec(
    3, 2,
    height_ratios=[1, 1.5, 1],  # A, B, (C,D)
    width_ratios=[1, 1],
    hspace=0.35,
    wspace=0.4,
)

layout_left = {
    "A": (0, slice(0, 2)),  # A spans both columns (row 0)
    "B": (1, slice(0, 2)),  # B spans both columns (row 1)
    "C": (2, 0),            # C in row 2, col 0
    "D": (2, 1),            # D in row 2, col 1
}

axes_left = make_named_subfig_grid(sL, gsL, layout_left)

axA = plot_image(axes_left["A"], file_path=Rs_image_path, rotate=-90)
# axA.set_axis_on()

axB_twin = axes_left["B"].twinx()
axB = plot_R_vs_Ls(
    axes_left["B"],
    R_stats=fitR,
    contact_R=events_data["contact_R"],
    data_color='black',
    plot_fit=True,
    fit_color='black'
)

axB_twin = plot_E_vs_Ls(
    axB_twin,
    E_stats=fitE,
    contact_E=events_data["contact_E"],
    data_color='blue',
    plot_fit=True,
    fit_color='blue'
)

adjust_axis_position(axes_left["B"], scale_width=1.0, scale_height=1.2,
                     dx=0.0, dy=0.0)

axC = plot_distribution(axes_left["C"], events_data["contact_R"],
                    bins=15, density=True, show_mean=True,
                    color = 'gray', mean_color='k',
                    xlabel = 'R (cm)',
                    ylabel = "Frequency"
)

axD = plot_distribution(axes_left["D"], events_data["contact_E"],
                    bins=15, density=True, show_mean=True,
                    color = 'gray', mean_color='k',
                    xlabel = 'E (MPa)',
                    ylabel = "Frequency"
)

# ---- RIGHT ----
gsR = sR.add_gridspec(
    2, 2,
    width_ratios=[1.0, 1.0],
    height_ratios=[0.8, 1.8],  # top row smaller, bottom row larger for G
    hspace=0.3,
    wspace=0.4,
)
layout_right = {
    "E": (0, 0),
    "F": (0, 1),
    "G": (1, slice(0, 2)),  # G spans both columns in row 1
}

axes_right = make_named_subfig_grid(sR, gsR, layout_right)
axes_right["E"].set_box_aspect(1)
axes_right["F"].set_box_aspect(1)


add_panel_labels(axes_left, labels=["A", "B", "C", "D"], 
            weight='bold', loc='upper left')
add_panel_labels(axes_right, labels=["E", "F", "G"], 
            weight='bold', loc='upper left')

# # Panel E: T_cn histogram
# axE = plot_T_cn_hist(axes_right["E"], plants_data["T_cn"], bins=nbins)
plot_distribution(axes_right["E"], plants_data["T_cn"],
                    bins=nbins, density=True, show_mean=True,
                    color = 'gray', mean_color='k',
                    xlabel = r'$T_{cn}$ (min)',
                    ylabel = "Frequency",
                    )

# Panel F: Max curvature histogram
# axF = plot_max_curvature_hist(axes_right["F"], plants_data["max_curvature"], bins=nbins)
plot_distribution(axes_right["F"], plants_data["max_curvature"], 
                    bins=nbins, density=True, show_mean=True,
                    color = 'gray', mean_color='k',
                    xlabel = r'$\kappa_{max}$ (cm$^{-1}$)',
                    ylabel = "Frequency",
                    )

# Panel G: Simulation schematic
schematic_path = PROJECT_DATA_PATH / "images" / "sim_unlabeled_crop.png"
plot_image(axes_right["G"], file_path=schematic_path)
adjust_axis_position(axes_right["G"], scale_width=1.65, scale_height=1.65,
                     dx=0.15, dy=-0.1)

# Save figure
save_Fig2_path = PROJECT_DATA_PATH / "test_figures" / "Fig2.svg"
plt.savefig(save_Fig2_path, format="svg", dpi=300)
plt.show()


#%% create zoomed in image
from src.utils.img_tools import zoom
import cv2
zoomed_Rs, roi = zoom(filename=str(Rs_image_path), zoom_target='Rs_zoom')
plt.imshow(zoomed_Rs)
Rs_zoomed = PROJECT_DATA_PATH / "images" / "R(s)" / "cropped"
cv2.imwrite(str(Rs_zoomed / "zoomed2.png"), zoomed_Rs)

#%% Fig2 v3
# Create two subfigures (left wider than right)
fig, subfig_left, subfig_right = make_two_subfig_layout(
    figsize=(7,7/1.5),
    subfig_width_ratios=(1, 1),  # 2.5, 1.5
    subfig_wspace=0.025,
)

# Left subfigure: 3 rows × 2 columns
gs_left = subfig_left.add_gridspec(
    3, 2,
    height_ratios=[1, 1.5, 1],  # A, B, (C,D)
    width_ratios=[1, 1],           
    hspace=0.35,
    wspace=0.35,

)
layout_left = {
    "A": (0, slice(0, 2)),  # A spans both columns (row 0)
    "B": (1, slice(0, 2)),  # B spans both columns (row 1)
    "C": (2, 0),            # C in row 2, col 0
    "D": (2, 1),            # D in row 2, col 1
}
axes_left = make_named_subfig_grid(subfig_left, gs_left, layout_left)

# Right subfigure: 2 rows × 2 columns
gs_right = subfig_right.add_gridspec(
    2, 2,
    width_ratios=[1.0, 1.0],
    height_ratios=[0.75, 2],  # top row smaller, bottom row larger for G
    hspace=0.35,
    wspace=0.35
)
layout_right = {
    "E": (0, 0),
    "F": (0, 1),
    "G": (1, slice(0, 2)),  # G spans both columns in row 1
}
axes_right = make_named_subfig_grid(subfig_right, gs_right, layout_right)

# Combine left and right axes dictionaries
axes = {**axes_left, **axes_right}

# Add panel labels
add_panel_labels(axes, labels=["A", "B", "C", "D", "E", "F", "G"], 
                weight='bold', loc='upper left')

nbins = 12

Rs_image_path = PROJECT_DATA_PATH / "images" / "stem" / "stem_img.jpg"
axA = plot_image(axes["A"], file_path=Rs_image_path, rotate=-90, reflect='y',
                 preserve_aspect=True, shift_x=0.1, shift_y=-0.05)

axB_twin = axes_left["B"].twinx()
axB = plot_R_vs_Ls(
    axes_left["B"],
    R_stats=fitR,
    data_color='black',
    plot_fit=True,
    fit_color='black'
)

axB_twin = plot_E_vs_Ls(
    axB_twin,
    E_stats=fitE,
    data_color='blue',
    plot_fit=True,
    fit_color='blue'
)
# Create unified legend from both axes
legend_for_R_E(axB, axB_twin)

adjust_axis_position(axes_left["B"], scale_width=1.0, scale_height=1.2,
                     dx=0.0, dy=0.0)

axC = plot_distribution(axes_left["C"], events_data["contact_R"],
                    bins=15, density=True, show_mean=True,
                    color = 'gray', mean_color='k',
                    xlabel = 'R (cm)',
                    ylabel = "Frequency"
)

axD = plot_distribution(axes_left["D"], events_data["contact_E"],
                    bins=15, density=True, show_mean=True,
                    color = 'gray', mean_color='k',
                    xlabel = 'E (MPa)',
                    ylabel = ""
)

# # Panel E: T_cn histogram
# axE = plot_T_cn_hist(axes_right["E"], plants_data["T_cn"], bins=nbins)
plot_distribution(axes_right["E"], plants_data["T_cn"],
                    bins=nbins, density=True, show_mean=True,
                    color = 'gray', mean_color='k',
                    xlabel = r'$T_{cn}$ (min)',
                    ylabel = "Frequency",
                    )

# Panel F: Max curvature histogram
# axF = plot_max_curvature_hist(axes_right["F"], plants_data["max_curvature"], bins=nbins)
plot_distribution(axes_right["F"], plants_data["max_curvature"], 
                    bins=nbins, density=True, show_mean=True,
                    color = 'gray', mean_color='k',
                    xlabel = r'$\kappa_{max}$ (cm$^{-1}$)',
                    ylabel = "",
                    )

# Panel G: Simulation schematic
schematic_path = PROJECT_DATA_PATH / "images" /"sim" /  "sim_unlabeled_crop.png"
axG = plot_image(axes["G"], file_path=schematic_path,
                 preserve_aspect=True, shift_x=0.2, shift_y=-0.045)

adjust_axis_position(axG, scale_width=1.175, scale_height=1.175,
                     dx=-0.15, dy=-0.05)


# Save figure
save_Fig2_path = PROJECT_DATA_PATH / "test_figures" / "Fig2.svg"
plt.savefig(save_Fig2_path, format="svg", dpi=300)
plt.show()
#%% Fig2 v4
# Create three equal subfigures

fig = plt.figure(figsize=(7,7/1.75)) # 7/1.5 ratio
subfigs = fig.subfigures(1, 3, width_ratios=(1.3,1.3,0.9), wspace=0.075)

# Left subfigure: 3 rows × 2 columns
gs = []
gs.append(subfigs[0].add_gridspec(
    3, 2,
    height_ratios=[0.4, 0.9, 0.8],  # A, B, (C,D)
    width_ratios=[1, 1],           
    hspace=0.35,
    wspace=0.5,
))
gs.append(subfigs[1].add_gridspec(
    2,1,
    height_ratios=[1, 1],
    hspace=0.3,
))
gs.append(subfigs[2].add_gridspec(1,1))

layouts = []
layouts.append({
    "A": (0, slice(0, 2)),  # A spans both columns (row 0)
    "B": (1, slice(0, 2)),  # B spans both columns (row 1)
    "C": (2, 0),            # C in row 2, col 0
    "D": (2, 1),            # D in row 2, col 1
})
layouts.append({
    "E": (0, 0),
    "F": (1, 0),
})
layouts.append({
    "G": (0, 0),
})

sub_axes = []
sub_axes.append(make_named_subfig_grid(subfigs[0], gs[0], layouts[0]))
sub_axes.append(make_named_subfig_grid(subfigs[1], gs[1], layouts[1]))
sub_axes.append(make_named_subfig_grid(subfigs[2], gs[2], layouts[2]))


# Combine left and right axes dictionaries
axes = {**sub_axes[0], **sub_axes[1], **sub_axes[2]}

# Add panel labels
add_panel_labels(axes, labels=["A", "B", "C", "D", "E", "F", "G"], 
                weight='bold', loc='upper left')

nbins = 12

Rs_image_path = PROJECT_DATA_PATH / "images" / "stem" / "stem_img.jpg"
axA = plot_image(axes["A"], file_path=Rs_image_path, rotate=-90, reflect='y',
                 preserve_aspect=True, shift_x=0.1, shift_y=-0.05)


axB_twin = axes["B"].twinx()
axB,R_fit_func = plot_R_vs_Ls(
    ax=axes["B"],
    R_stats=fitR,
    data_color='black',
    plot_fit=True,
    fit_color='black'
)

axB_twin,E_fit_func = plot_E_vs_Ls(
    ax=axB_twin,
    E_stats=fitE,
    data_color='blue',
    plot_fit=True,
    fit_color='blue'
)
# Create unified legend from both axes
legend_for_R_E(axB, axB_twin)

adjust_axis_position(ax=axes["B"], scale_width=1.0, scale_height=1.2,
                     dx=0.0, dy=0.0)

axC = plot_distribution(axes["C"], events_data["contact_R"],
                    bins=18, density=True, show_mean=True,
                    color = 'gray', mean_color='k',
                    xlabel = 'R (cm)',
                    ylabel = "frequency", nbin_ticks=2,
                    xticks = [0.041,0.045],
                    yticks = [0,0.1,0.2]
)

axD = plot_distribution(axes["D"], events_data["contact_E"],
                    bins=18, density=True, show_mean=True,
                    color = 'gray', mean_color='k',
                    xlabel = 'E (MPa)',
                    ylabel = "", 
                    xticks = [0,5,10,15],
                    yticks= [0,0.2,0.4,0.6],
)

# # Panel E: T_cn histogram
axE = plot_distribution(axes["E"], plants_data["T_cn"],
                    bins=nbins, density=True, show_mean=True,
                    color = 'gray', mean_color='k',
                    xlabel = r'$T_{cn}$ (min)',
                    ylabel = "frequency",
                    xticks = [80,100,120,140],
                    yticks = [0,0.1,0.2,0.3]
                    )

# Panel F: Max curvature histogram
axF = plot_distribution(axes["F"], plants_data["max_curvature"], 
                    bins=nbins, density=True, show_mean=True,
                    color = 'gray', mean_color='k',
                    xlabel = r'$\kappa_{max}$ (cm$^{-1}$)',
                    ylabel = "frequency",
                    xticks = [0,0.1,0.2,0.3],
                    yticks = [0,0.1,0.2,0.3]
                    )

# axF_inset = create_inset(axF, width="43%", height="43%", 
#                          loc='upper right',
#                          borderpad=0, bbox_to_anchor=(0.0, -0.0, 1, 1), 
#                          bbox_transform=axF.transAxes,
# )
# curvature_path = PROJECT_DATA_PATH / "images" / "curvature" /  "curvature_crop.png"
# axF_inset = plot_image(axF_inset, file_path=curvature_path,
#                  preserve_aspect=True, shift_x=0.1, shift_y=0.0)


# Panel G: Simulation schematic
schematic_path = PROJECT_DATA_PATH / "images" /"sim" /  "sim_unlabeled_crop_2.png"
axG = plot_image(axes["G"], file_path=schematic_path,
                 preserve_aspect=True, shift_x=-0.1, shift_y=-0.1)
img_scaling = 1.6
adjust_axis_position(axG, scale_width=img_scaling, scale_height=img_scaling,
                     dx=-0.3, dy=-0.15)


# Save figure
save_Fig2_path = PROJECT_DATA_PATH / "test_figures" / "Fig2.svg"
plt.savefig(save_Fig2_path, format="svg", dpi=300)
plt.show()

#%% SM plots
# ----- SM plots
# lo and llev distributions
fig,ax = plt.subplots(figsize=(4,3))
ax_Llev = plot_distribution(ax=ax, data=events_data["L_base"],
                    bins=18, density=True, show_mean=True,
                    color = 'gray', mean_color='k',
                    xlabel = r'$\ell_{\text{lev}}$ (cm)',
                    ylabel = "frequency",
                    xticks=[30,45,60], yticks=[0,0.1,0.2],
)

fig,ax = plt.subplots(figsize=(4,3))
ax_Lo = plot_distribution(ax=ax, data=events_data["ltip"],
                    bins=18, density=True, show_mean=True,
                    color = 'gray', mean_color='k',
                    xlabel = r'$\ell_{\text{o}}$ (cm)',
                    ylabel = "frequency",
                    xticks=[0,2,4,6], yticks=[0,0.1,0.2],
)
print(f"Lo mean: {events_data['ltip'].mean():.2f} cm plus minus std: {events_data['ltip'].std():.2f} cm")
print(f"Llev mean: {events_data['L_base'].mean():.2f} cm plus minus std: {events_data['L_base'].std():.2f} cm")

#%%