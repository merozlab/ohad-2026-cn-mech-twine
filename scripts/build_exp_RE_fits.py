'''
build_final.py
Script to build experimental plants and 
save with final material property fits.
'''

#%% imports
# %load_ext autoreload
# %autoreload

import bootstrap

from src.io.cache import load_snapshot, save_snapshot

from src.analysis.exp_param_analysis import assign_contact_material_props

#%% fit R and E to experimental plants and events, using contact material properties
# save plants with fit to material properties

RE_fit_plants = load_snapshot("plants", "exp", stage="curvature")
RE_fit_events = load_snapshot("events", "exp", stage="sine_fit")

fitR,fitE = assign_contact_material_props(
                    RE_fit_plants,RE_fit_events, min_count=2)
#%% save results
snapshot_path = save_snapshot(RE_fit_plants, file_type="plants", 
                              data_type="exp", stage="R_E_fits")
snapshot_path = save_snapshot(RE_fit_events, file_type="events",
                                data_type="exp", stage="R_E_fits")

#%%