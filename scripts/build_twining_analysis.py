#%% load modules and set up paths
# %load_ext autoreload
# %autoreload

import bootstrap # sets up PROJCETROOT path for imports
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.slip_twine_analysis import (
    get_slip_twine_arrays,
    attach_slip_twine_to_events,
    extract_slip_twine_scalars,
    # extract_non_h5_twine_times,
)
from src.plotting.slip_twine_plots import (
    plot_slip_twine_time_hist,
    plot_time_vs_ltip,
    plot_motor_twine,
)

from src.io.paths import PROJECT_DATA_PATH, H5_RESULTS_PATH
from src.io.cache import load_snapshot, save_snapshot
from src.io.data_imports import (
    load_h5_twine_results,
    load_support_tracking,
    get_h5_frames,
)

#%% import pendulum events
# -------- load pendulum events --------
exp_plants = load_snapshot("plants", "exp", stage="R_E_fits")

# build registry and context for event binding
# reg = {p.exp_num: p for p in exp_plants}
# ctx = PlantContext(reg)
# Event.bind_context(ctx)

exp_events = load_snapshot("events", "exp", stage="R_E_fits", plants=exp_plants, bind_context=True)

#%%  import pendulum twine results
# ------ load twine results and tracking data for compensation ------

h5_results = load_h5_twine_results(PROJECT_DATA_PATH)
track_dict = load_support_tracking(PROJECT_DATA_PATH)
h5_name_file = H5_RESULTS_PATH / "interekt_files.txt"
h5_frames = get_h5_frames(h5_name_file)

# extract twine time for non-h5 events and attach to events
# non_h5_results = extract_non_h5_twine_times(exp_events)

attach_slip_twine_to_events(exp_events, h5_results, h5_frames, track_dict) # add non_h5_results

slip_twine_results = extract_slip_twine_scalars(exp_events)

state, slip, twine = get_slip_twine_arrays(exp_events)
#%% pendulum plots
# ------ plots --------
plot_slip_twine_time_hist(ax=None, slip=slip, twine=twine, 
                     xlabel=r'$\tau/T_{\mathrm{CN}}$',
                     threshold = 2.5, nbins = 12)
# ax.set_ylim(0, 0.25)


ax = plot_time_vs_ltip(ax = None, slip=slip, twine=twine, 
                xlabel=r'$\tau/T_{\mathrm{CN}}$',
                ylabel=r'$l_{\mathrm{tip}}$',
                t_threshold=2.5, ltip_threshold=6)

plot_time_vs_ltip(ax = None, slip=slip, twine=twine, 
                xlabel=r'$\tau/T_{\mathrm{CN}}$',
                ylabel=r'$l_{\mathrm{tip}}$',
                t_threshold=2.5, ltip_threshold=6,
                log_log_scale=True)

#%% save results
# ----- save results -----
save_snapshot(exp_plants, "plants", "exp", "slip-twine")
save_snapshot(exp_events, "events", "exp", "slip-twine", plants=exp_plants)

#%% tests
# h5_name_file = H5_RESULTS_PATH / "interekt_files.txt"
# h5_frames = get_h5_frames(h5_name_file)
# h5_twine_initiation_compensation(h5_frames, 23, 1, track_dict)
# exp_events[1].twine_data['h5_compensation']
# fig,ax = plt.subplots()
# time
# angle
# smoothed_angle
# frames_till_h5
# till_h5
# for ev in exp_events:
#     # print(ev.exp_num, ev.event_num)

#     if hasattr(ev, "twine_data") and ev.twine_data is not None:
#         if ev.twine_data.get('twine_estimate_min') is not None:
#             if ev.twine_data.get('twine_estimate_min')<50:
#                 h5_twine_initiation_compensation(h5_frames, ev.exp_num, ev.event_num, track_dict)
#                 print(ev.exp_num, ev.event_num) 
#                 print(ev.frm0_side if hasattr(ev, "frm0_side") else 'N/A')
#                 print( ev.twine_data['twine_estimate_min'])
#                 print(ev.twine_data.get('h5_compensation', 'N/A'))
#                       ev.twine_data.get('contact_start_time', 'N/A'))
    #              ax.plot(h5_results[(ev.exp_num, ev.event_num)]['time']/60, 
    #                      h5_results[(ev.exp_num, ev.event_num)]['smoothed_angle'], 
    #                     label=f"Exp {ev.exp_num} Event {ev.event_num}")
                #  print(h5_results[(ev.exp_num, ev.event_num)]['frames_till_h5'])
                #  print(h5_results[(ev.exp_num, ev.event_num)]['twine_estimate_raw'])
# ax.legend()

#         comp = ev.twine_data['h5_compensation']
#         if comp is not None and comp >0:
#             print(ev.twine_data['h5_compensation'], ev.exp_num, ev.event_num)
# np.array([ev.twine_state for ev in exp_events])

#%% import motor twine results
motor_plants = load_snapshot("plants", "motor", stage="build")
motor_events = load_snapshot("events", "motor", stage="build", 
                             bind_context=True, plants=motor_plants)  
#%% plot motor twine results

plot_motor_twine(ax=None, motor_events=motor_events, exp_twine_events=twine, 
                fit_line=False, xlabel='f (rph)', ylabel=r'$\tau$ (min)')


#%%