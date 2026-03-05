#%% imports
# %load_ext autoreload
# %autoreload
# imports
import matplotlib.pyplot as plt
import random
import numpy as np

import bootstrap # sets up PROJCETROOT path for imports 

from src.analysis.exp_sine_analysis import (
    fit_events_sine_2param,
)
from src.io.cache import load_snapshot, save_snapshot

#%% load events from cache (with plant context)
# load cached regular events with plant context
events = load_snapshot("events", "exp", stage="build", bind_context=True)

#%% fit to sine 
# preform analysis
results = fit_events_sine_2param(
    events,
    window_size=10,
    start_indx=5,
    pTcn=0.75,
    # spike_threshold = 0.1,
    # spike_window = 20,
    spike=False
)
#%% save results to cache
save_snapshot(events, file_type="events", data_type="exp", stage="sine_fit")
#%%
