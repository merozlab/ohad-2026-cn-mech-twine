#%% imports
# imports, project setup and function definition
# %load_ext autoreload
# %autoreload

import re
import os
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path
import importlib

# Paths
from src.io.paths import PROJECT_DATA_PATH
# Extreme Plant class
from src.core.xtrm_plant import Xtrm_Plant
# Extreme Event class
from src.core.xtrm_event import Xtrm_Event

# Load and save snapshots
from src.io.cache import load_snapshot, save_snapshot

# Data imports
from src.io.data_imports import (
    load_xtrm_data
)

print('Libraries imported successfully.')
#%% load data
# --------- Load data --------------
xtrm_data = load_xtrm_data(PROJECT_DATA_PATH)

print(f"Loaded extreme mass experiment data for {len(xtrm_data)} experiments.")

#%% build Extreme Plant and Event objects
# ----------- define function to build Extreme Plant and Event objects ----------
def build_xtrm_events(DATA_PATH):
    """
    Build Extreme Plant and Event objects from Excel files.
    """

    xtrm_data = load_xtrm_data(DATA_PATH)

    xtrm_plants = {}
    xtrm_events = []

    for (exp_num, mass_label), plant_data in xtrm_data.items():
        df = plant_data["df"]

        # ---- build plant (ONCE per experiment) ----
        plant = Xtrm_Plant(exp_num=exp_num, mass_label=mass_label)
        xtrm_plants[exp_num] = plant

        last_event_num = None
        ev = None

        # ---- build events ----
        for i in tqdm(range(len(df)), desc=f"Building xtrm events for {exp_num}"):

            event_num = int(df.at[i, "event_num"])

            if event_num != last_event_num:
                ev = Xtrm_Event(Xtrm_Plant=plant, event_num=event_num)
                xtrm_events.append(ev)
                
                # Get params ONCE when event is created
                ev.get_params(df.iloc[i])
                ev.compute_resistance()
                last_event_num = event_num

    return xtrm_plants, xtrm_events

#%% build Extreme Plant and Event objects
# Build Extreme Plant and Event objects
xtrm_plants, xtrm_events = build_xtrm_events(PROJECT_DATA_PATH)
print(f"Built {len(xtrm_events)} events.")
#%% save snapshots
# ----- save snapshots -----
save_snapshot(xtrm_plants, file_type="plants", data_type="xtrm", stage="build")
save_snapshot(xtrm_events, file_type="events", data_type="xtrm", stage="build")

#%% test
# Find the first event of stable experiment 3
xtrm_events = load_snapshot("events", "xtrm", stage="build", bind_context=True)
event_exp_3 = [e for e in xtrm_events if e.xplnt.exp_num == 3 and e.xplnt.support_mass_label == "stable"]

if event_exp_3:
    ev = event_exp_3[0]
    print(ev.__dict__)
else:
    print("No events found for experiment 3 with stable mass label")
#%%