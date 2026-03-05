#%% imports

# %load_ext autoreload
# %autoreload

import bootstrap # sets up PROJCETROOT path for imports
# import matplotlib.pyplot as plt

from src.analysis.sim_analysis import (
    build_simulation_events,
    fit_sim_events_sine_2param,
)

# from src.plotting.sim_plots import (
#     plot_sim_tslip,
#     plot_sim_trajectories_with_sine_fit,
#     plot_sim_R2_distribution,
#     plot_normalized_sim_trajectories,
#     plot_sim_Fvec_vs_Fplanar,
#     plot_sim_torque,
#     plot_Fs_Bk,
#     plot_sim_maxF_vs_ltip,
#     plot_sim_maxF_vs_E,
#     plot_sim_tslip,
# )

from src.io.cache import load_snapshot, save_snapshot
from src.io.paths import PROJECT_DATA_PATH

#%% build

sim_events = build_simulation_events(PROJECT_DATA_PATH)
#%% save
save_snapshot(sim_events, file_type="events", data_type="sim", stage="build")
#%% load
# sim_events = load_snapshot("events", "sim", stage="build")
print(f"{sim_events[1].l_tip=}") # test

#%% fit
fit_results = fit_sim_events_sine_2param(sim_events)
save_snapshot(sim_events, file_type="events", data_type="sim", stage="sine_fit")
#%% test
# sim_events = load_snapshot("events", "sim", stage="sine_fit")
print(f"{sim_events[1].sine_fit_2param['R2']=}") # test

#%% plotting

# sim_events = load_snapshot("events", "sim", stage="w_sine_fit")
# plot_sim_trajectories_with_sine_fit(sim_events, indices=[7,11,14])
# plot_sim_R2_distribution(sim_events)
# plot_normalized_sim_trajectories(sim_events, show_trj=True)
# plot_sim_Fvec_vs_Fplanar(sim_events, indices=[7,11,14])
# plot_sim_torque(sim_events, friction=False)
# Fs_Bk_fit_results = plot_Fs_Bk(sim_events, fit_func=linfunc)
# plot_sim_maxF_vs_ltip(sim_events)
# plot_sim_maxF_vs_E(sim_events)
# plot_sim_tslip(sim_events)
#%% test of l_tip


# for ev in sim_events:
#     print(f"{ev.LOC=},{ev.l_tip=:.2f}, {ev.growth_zone-min(ev.s_contact[ev.s_contact>0]):.2f},{ev.growth_zone-max(ev.s_contact):.2f}")
#%%

