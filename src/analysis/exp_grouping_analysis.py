import numpy as np
import pandas as pd


def extract_E_grouping_data(events, force_stop_ind=110):
    """
    Extract event-level quantities for E-grouped analysis.
    
    Returns:
        df_scalars: pandas DataFrame (rows = events, columns = scalar properties)
        data_ragged: dict of object arrays (variable-length data per event)
    """

    scalars = {}
    ragged = {}
    
    # Initialize lists
    for key in ["contact_E", "contact_R", "near_R", 
                "B_contact", "B_near", 
                "B_contact_kappa", "B_near_kappa", 
                "l_tip", "l_lev", "max_F_s", "Tcn","twine_state",
                "F_amp"]:
        scalars[key] = []
    
    for key in ["f", "F_s"]:
        ragged[key] = []

    for ev in events:
        plant = ev.plnt
        keys = np.array(list(plant.r_sections.keys()), dtype=float)
        vals = np.array(list(plant.r_sections.values()), dtype=float)

        # Nearest measured R
        if len(keys):
            idx = np.nanargmin(np.abs(keys - ev.ltip))
            near_R = vals[idx] if vals[idx] != 0 else np.nan
        else:
            near_R = np.nan

        # Max curvature
        if hasattr(plant, "free_shape"):
            max_kappa = np.max(plant.free_shape["curvature_cm"])
        else:
            max_kappa = np.nan

        # Scalars
        E_contact = getattr(ev, "contact_E", np.nan)
        R_contact = getattr(ev, "contact_R", np.nan)
        
        B_contact = E_contact * np.pi * R_contact**4 # MPa * cm^4
        B_near = E_contact * np.pi * near_R**4

        scalars["contact_E"].append(E_contact)
        scalars["contact_R"].append(R_contact)
        scalars["near_R"].append(near_R)
        scalars["B_contact"].append(B_contact)
        scalars["B_near"].append(B_near)
        scalars["B_contact_kappa"].append(B_contact * max_kappa * 1e3) # conversion to mJ
        scalars["B_near_kappa"].append(B_near * max_kappa * 1e3)
        scalars["l_tip"].append(ev.ltip)
        scalars["l_lev"].append(ev.Lbase)
        scalars["Tcn"].append(ev.plnt.avgT)
        scalars["twine_state"].append(ev.twine_state)
        scalars["F_amp"].append(ev.sine_fit_2param["A"])
        # Ragged (variable length)
        f = ev.F_bean[:force_stop_ind]
        ragged["f"].append(f)
        ragged["F_s"].append(f * ev.Lbase * 0.01)

        scalars["max_F_s"].append(np.nanmax(f * ev.Lbase * 0.01))
        # scalars["max_F_s"].append(ev.sine_fit_2param["A"] * ev.Lbase * 0.01)

    # Convert to DataFrame (keeps rows aligned)
    df_scalars = pd.DataFrame(scalars)
    data_ragged = {k: np.array(v, dtype=object) for k, v in ragged.items()}

    return df_scalars, data_ragged

def filter_scalar_ragged(scalar_data, ragged_data, filter_params={}):
    """ Filter data based on a parameter and value.
        scalar_data: pandas DataFrame of scalar quantities
        ragged_data: dict of object arrays of ragged data
        filter_params: dict of {param: (min/max, value)} to filter on
        
        returns:
        filtered_scalar_data, filtered_ragged_data
    """

    filtered_scalar_data = scalar_data.copy()
    filtered_ragged_data = {k: v.copy() for k, v in ragged_data.items()}

    for (param,minmax), filter_value in filter_params.items():
        if minmax == "max":
            mask = filtered_scalar_data[param] <= filter_value
        elif minmax == "min":
            mask = filtered_scalar_data[param] >= filter_value
        else:
            continue
            
        filtered_scalar_data = filtered_scalar_data[mask]
        filtered_ragged_data = {k: v[mask] for k, v in filtered_ragged_data.items()}
    
    return filtered_scalar_data, filtered_ragged_data