#%% imports
# %load_ext autoreload
# %autoreload
# imports, project setup and function definition
import numpy as np
from tqdm import tqdm
import pandas as pd

# Paths
from src.io.paths import PROJECT_DATA_PATH
# Motor Plant class
from src.core.motor_plant import Motor_Plant
# Motor Event class
from src.core.motor_event import Motor_Event

# Load and save snapshots
from src.io.cache import save_snapshot, load_snapshot

# Data imports
from src.io.data_imports import (
    load_motor_data
)

print('Libraries imported successfully.')

#%% define build function
# ------------ define build function ------------
def build_motor_events(DATA_PATH):
    motor_df, motor_track_dict = load_motor_data(DATA_PATH)

    motor_plants = {}
    motor_events = []

    for _, row in motor_df.iterrows():
        exp_num = int(row["exp_num"])

        # Build plant once per experiment
        if exp_num not in motor_plants:
            plant = Motor_Plant.from_dataframe(row, exp_num)
            motor_plants[exp_num] = plant

        # Build event
        ev = Motor_Event(row, exp_num)
        motor_events.append(ev)

    # ---- Bind context ----
    from src.core.plant_context import PlantContext
    ctx = PlantContext(motor_plants)
    Motor_Event.bind_context(ctx)

    # Verify that all events have valid plant references
    for event in motor_events:
        if event.exp_num not in motor_plants:
            raise ValueError(f"Event {event.event_num} references an unknown experiment number: {event.exp_num}")

    return motor_plants, motor_events


#%% Build Motor Events
# --------- Build Motor Plant and Event objects --------
motor_plants, motor_events = build_motor_events(PROJECT_DATA_PATH)
print(f"Built {len(motor_events)} events.")
#%% test
# Test first event
# print(motor_events[0].plnt.arm_length, motor_events[0].twine_time_estimate)
#%% Save snapshots
save_snapshot(motor_plants, file_type="plants", data_type="motor", stage="build")
save_snapshot(motor_events, file_type="events", data_type="motor", stage="build")
#%%