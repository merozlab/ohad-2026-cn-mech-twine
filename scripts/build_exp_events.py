#%% imports
# imports, project setup and function definition
import re
import os
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path
import importlib

import bootstrap

# Paths
from src.io.paths import PROJECT_DATA_PATH, CACHE_PATH
# Event class
from src.core.exp_event import Event
# importlib.reload(Event)
# Load and save snapshots
from src.io.cache import save_snapshot, load_snapshot
# Experiment context
from src.core.plant_context import PlantContext
from src.core.exp_event import Event
# Data imports
from src.io.data_imports import (
    load_metadata,
    load_support_tracking,
    load_contact_tracking,
)

print('Libraries imported successfully.')
#%% load data
# -------------------------------------------------
# Load data
# -------------------------------------------------
print(PROJECT_DATA_PATH)
print(CACHE_PATH)

data_panda, problem_exp = load_metadata(PROJECT_DATA_PATH)
track_dict = load_support_tracking(PROJECT_DATA_PATH)
contact_dict = load_contact_tracking(PROJECT_DATA_PATH)

print(
    f"imported {len(track_dict)} track files, "
    f"and {len(contact_dict)} contact files"
)

plants = load_snapshot("plants", "exp", stage="curvature")
print(f"loaded {len(plants)} plants from latest snapshot")
 
# definition of build_events
def build_events(data_panda, plants, track_dict, contact_dict):
    """ Build Event objects from metadata and associate with Plant objects."""
    
    # Bind Event to Plant context
    plant_registry = {p.exp_num: p for p in plants} # create plant registry
    ctx = PlantContext(plant_registry)
    Event.bind_context(ctx)

    events = []
    last_key = None

    for i in tqdm(range(len(data_panda)), desc="Building events"):
        exp_num = int(re.findall(r"\d{3,4}", data_panda.at[i, "Exp_num"])[0 ])
        event_num = int(re.findall(r"_[0-9]", data_panda.at[i, "Exp_num"])[0][1])
        view = data_panda.at[i, "View"]

        plant = plant_registry[exp_num]
        key = (exp_num, event_num)

        if key != last_key:
            ev = Event(exp_num = exp_num, event_num = event_num)
            events.append(ev)
            last_key = key
        else:
            ev = events[-1]
        

        ev.load_view_metadata(data_panda, i, view)
        ev.extract_tracking(view, track_dict, contact_dict)

    # second pass: only after both views exist
    for ev in events:
        if {"side", "top"} <= ev.views_seen:
            ev.compute_force()
            ev.save_Lbase()

    return events

#%% build events
events = build_events(data_panda, plants, track_dict, contact_dict)
print(f"Built {len(events)} events across {len(plants)} plants.")
#%% save events
save_snapshot(events, file_type="events", data_type="exp", stage="build",
              plants=plants)
print("Events snapshot saved.")
#%% test
# Verify one event
import matplotlib.pyplot as plt
ev0 = events[35]
print(f"Event {ev0.event_num} of Plant {ev0.exp_num}:")
print(f"  Views seen: {ev0.views_seen}")
plt.plot(ev0.timer, ev0.F_bean)
print(f"{ev0.Lbase=}")
print(f"{ev0.plnt.free_shape['s_cm'][-1]=}")
print(f"{ev0.plnt.r_measured=}")
print(f"{ev0.ltip=}")
print(f"{ev0.plnt.start_height=}")
print(f"{ev0.plnt.arm_cm=}")
print(f"{ev0.plnt.free_shape['s_cm'][-1]=}")
print(f"{ev0.twine_state=}")
print(f"{1/np.max(ev0.plnt.free_shape['curvature_cm'])=}")
#%% test loading
plants = load_snapshot("plants", "exp", stage="curvature")
events = load_snapshot("events", "exp", stage="build", bind_context=True, plants=plants)
#%%