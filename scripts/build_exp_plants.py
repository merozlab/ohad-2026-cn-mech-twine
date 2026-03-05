# scripts/build_plants.py

#%% imports
# General imports
import re
import os
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path
import importlib

import bootstrap

# %load_ext autoreload
# %autoreload
# -------------------------------------------------
# Paths
from src.io.paths import PROJECT_ROOT, PROJECT_DATA_PATH
# -------------------------------------------------
# Load Plant class
from src.core.exp_plant import Plant
# -------------------------------------------------
# data import functions
from src.io.data_imports import (
    load_metadata,
    load_support_tracking,
    load_contact_tracking,
    load_E_dict,
)
from src.io.cache import load_snapshot, save_snapshot

print('Libraries imported successfully.')

#%% Functions
# -------------------------------------------------
# Load data
# -------------------------------------------------
data_panda, problem_exp = load_metadata(PROJECT_DATA_PATH)
track_dict = load_support_tracking(PROJECT_DATA_PATH)
contact_dict = load_contact_tracking(PROJECT_DATA_PATH)
E_dict = load_E_dict(PROJECT_DATA_PATH)

print(
    f"imported {len(track_dict)} track files, "
    f"{len(contact_dict)} contact files, "
    f"and {len(E_dict)} Young's modulus files."
)

def build_plants_from_dataframe(data_panda,E_dict):
    """
    Parameters
    ----------
    data_panda : pandas.DataFrame
        Experimental metadata table
    E_dict : dict
        Legacy Young's modulus dictionary

    Returns
    -------
    plants : list[Plant]
    """

    plants = []
    last_exp = None

    N = len(data_panda)

    for i in tqdm(range(N)):
        try:
            # extract experiment number exactly like before
            exp = int(re.findall(r"\d{3,4}", data_panda.at[i, "Exp_num"])[0])
            view = data_panda.at[i, "View"]

            # new plant if experiment changes
            if i == 0 or exp != last_exp:
                plant = Plant.from_dataframe(data_panda.iloc[i], exp)

                # view-dependent data
                plant.load_view_data()

                # CN data
                plant.load_cn_data()

                # material properties (E + r)
                plant.load_material_properties(E_dict)

                plants.append(plant)
                last_exp = exp

            # top-view row updates pix2cm_t (matches your original logic)
            if view == "top":
                plants[-1].pix2cm_t = float(
                    data_panda.at[i, "Top_pix2cm"]
                )

        except Exception as e:
            print(f"[WARN] failed at row {i}: {e}")

    return plants

#%% build plants from dataframe
# build Plant instances
plants_fromdf = build_plants_from_dataframe(data_panda,E_dict)
print(f"built {len(plants_fromdf)} Plant instances from dataframe")
#%% save snapshot of plants built from dataframe (before curvature)
# save snapshot
snapshot_path = save_snapshot(plants_fromdf, file_type="plants", 
                              data_type="exp", stage="build")
#%% load free-form skeleton data, compute curvature and attach to plants
# Load free-form skeleton data, compute curvature and attach to plants
from src.io.data_imports import load_skeleton_xl

skeleton_dict = load_skeleton_xl(PROJECT_DATA_PATH)

plants_w_curvature = plants_fromdf.copy()
for plant in tqdm(plants_w_curvature):
    if plant.exp_num in skeleton_dict:
        skel_df, recalc_df = skeleton_dict[plant.exp_num]
        plant.load_free_shape(skel_df, recalc_df)
    else:
        print(f"[WARN] no skeleton data for exp {plant.exp_num}")


#%% save snapshot of plants with curvature
snapshot_path = save_snapshot(plants_w_curvature, file_type="plants", 
                              data_type="exp", stage="curvature")

#%% test
k=10
p = plants_w_curvature[k]
print(p.exp_num)
print(p.arm_cm, p.start_height, p.L0)
print(p.avgT, p.omega0)
print(p.E_sections)
print(p.r_sections)
print(p.pix2cm_t, p.pix2cm_s)
print(p.camera)
print(p.free_shape.keys())
print(p.free_shape["curvature_cm"])

#%%
