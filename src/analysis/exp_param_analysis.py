import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

import bootstrap

from src.utils.curve_tools import calc_R2_RMSE, fit_w_error
from src.utils.math_tools import linfunc, parafunc


# Aggregate radius and Young's modulus by stem section

def _safe_r2(x, y, popt, fit_func):
    """Return R2 or NaN if popt is invalid/non-numeric."""
    try:
        p = np.asarray(popt, dtype=float).ravel()
        if p.size == 0 or np.any(~np.isfinite(p)):
            return np.nan
        r2, _, _ = calc_R2_RMSE(x, y, p, fit_func)
        return r2
    except Exception:
        return np.nan
    
def collect_radius_E_by_section(plants, min_count=2):
    """
    Collect radius and Young's modulus values by stem section
    across all plants.

    Parameters
    ----------
    plants : list[Plant]
        Plant objects with attributes:
        - r_sections : dict {L-s : R}
        - avgE_sections : dict {L-s : E}
    min_count : int
        Minimum number of samples per section to keep

    Returns
    -------
    all_r_df : pd.DataFrame
        Rows = plants, columns = L-s sections (cm), values = radius (cm)
    all_E_df : pd.DataFrame
        Rows = plants, columns = L-s sections (cm), values = Young's modulus (MPa)
    """

    all_r_df = pd.DataFrame()
    all_E_df = pd.DataFrame()

    for plant in plants:
        # Radius
        if hasattr(plant, "r_sections"):
            r_df = pd.DataFrame(plant.r_sections, index=[0])
            all_r_df = pd.concat([all_r_df, r_df], ignore_index=True)

        # Young's modulus
        if hasattr(plant, "E_sections"):
            E_df = pd.DataFrame(plant.E_sections, index=[0])
            all_E_df = pd.concat([all_E_df, E_df], ignore_index=True)

    # Drop columns with too few samples
    r_counts = all_r_df.count()
    E_counts = all_E_df.count()

    all_r_df = all_r_df.loc[:, r_counts >= min_count]
    all_E_df = all_E_df.loc[:, E_counts >= min_count]

    # Remove section 0 from E if present (often artificial)
    if 0 in all_E_df.columns:
        all_E_df = all_E_df.drop(columns=0)

    # Add E=0 at section 0 explicitly (for fitting / interpolation)
    all_E_df[0] = 0.0

    return all_r_df, all_E_df



def fit_R_Ls(all_r_df):
    """
    Linear fit for radius: R(L-s) = a*(L-s) + b
    """

    Ls = np.array(sorted(all_r_df.columns))
    Ls_err = 5 / np.sqrt(12)  # Bin width error
    R_mean = np.array([all_r_df[col].mean() for col in Ls if np.isfinite(all_r_df[col].mean())])
    R_std = np.array([all_r_df[col].std() for col in Ls if np.isfinite(all_r_df[col].std())])
    counts = np.array([all_r_df[col].count() for col in Ls if np.isfinite(all_r_df[col].count())])
    R_std = np.where(R_std == 0, np.max(R_std), R_std)

    # Fit linear model
    fit_res = fit_w_error(
            Ls, dx=Ls_err, y=R_mean, dy=R_std, 
            fit_func = linfunc)
    # r2 = _safe_r2(Ls, R_mean, popt, linfunc)
    # r2,_,_ = calc_R2_RMSE(Ls, R_mean, popt, linfunc)

    R_lin_fit_results = {
        "a": fit_res["popt"][0], # fit slope
        "b": fit_res["popt"][1], # fit intercept
        "R2": fit_res["r2"], # coefficient of determination
        "Chi2_red": fit_res["chi_sq_red"], # reduced chi-squared
        "Chi2_std": fit_res["chi_sq_std"], # chi-squared std dev
        "L-s": Ls, # x data used for fitting
        "L-s_err": Ls_err, # x data error used for fitting
        "R_mean": R_mean, # y data used for fitting
        "R_std": R_std, # y std dev used for fitting
        "R_counts": counts, # counts per section
    }

    return R_lin_fit_results

def fit_E_Ls(all_E_df):
    """
    Parabolic fit for Young's modulus: E(L-s) = a*(L-s)^2 + b
    """

    Ls = np.array(sorted(all_E_df.columns))
    counts = np.array([all_E_df[col].count() for col in Ls if np.isfinite(all_E_df[col].count())])
    Ls_err = 5 / np.sqrt(12)  # Bin width error
    E_mean = np.array([all_E_df[col].mean() for col in Ls if np.isfinite(all_E_df[col].mean())])
    E_std = np.array([all_E_df[col].std() for col in Ls if np.isfinite(all_E_df[col].std())])
    E_std = np.where(E_std == 0, np.max(E_std), E_std)
    E_std[0] = 0.0 

    # Fit parabolic model
    fit_res = fit_w_error(
            Ls, dx=Ls_err, y=E_mean, dy=E_std, 
            fit_func = parafunc)
    # r2 = _safe_r2(Ls, E_mean, popt, parafunc)
    # r2, _, _ = calc_R2_RMSE(Ls, E_mean, popt,parafunc)

    E_para_fit_results = {
        "a": fit_res["popt"][0],  # fit coefficient for (L-s)^2
        "b": fit_res["popt"][1],  # fit intercept
        "R2": fit_res["r2"],  # coefficient of determination
        "Chi2_red": fit_res["chi_sq_red"], # reduced chi-squared
        "Chi2_std": fit_res["chi_sq_std"], # chi-squared std dev
        "L-s": Ls,  # x data used for fitting
        "L-s_err": Ls_err, # x data error used for fitting
        "E_mean": E_mean,  # y data used for fitting
        "E_std": E_std,  # y std dev used for fitting
        "E_counts": counts, # counts per section
    }

    return E_para_fit_results


# Assign contact material properties to events

def assign_contact_material_props(plants,events, min_count=2):
    """
    Assign contact radius and Young's modulus to Event objects
    based on their contact position via fitted E,R functions

    Modifies events in-place.

    Parameters
    ----------
    events : list[Event]
        Event objects with attribute:
        - ltip
    R_fit_func : callable
        Radius interpolation function
    E_fit_func : callable
        Young's modulus interpolation function

    Returns
    -------
    R_fit_results : dict
        Results from fit_R_Ls
    E_fit_results : dict
        Results from fit_E_Ls
    Saves contact_R and contact_E to each Event
    """
    all_R, all_E = collect_radius_E_by_section(plants, min_count=min_count)
    R_fit_results = fit_R_Ls(all_R)
    E_fit_results = fit_E_Ls(all_E)
    R_fit_func = lambda x: linfunc(x, R_fit_results["a"], R_fit_results["b"])
    E_fit_func = lambda x: parafunc(x, E_fit_results["a"], E_fit_results["b"])

    for p in plants:
        p.R_fit_results = R_fit_results
        p.E_fit_results = E_fit_results

    for ev in events:
        if not hasattr(ev, "ltip"):
            ev.contact_R = np.nan
            ev.contact_E = np.nan
            continue

        lo = ev.ltip

        try:
            ev.contact_R = float(R_fit_func(lo))
        except Exception:
            ev.contact_R = np.nan

        try:
            ev.contact_E = float(E_fit_func(lo))
        except Exception:
            ev.contact_E = np.nan

    return R_fit_results,E_fit_results

# Collect scalar distributions for histograms

def collect_scalar_distributions(plants, events):
    """
    Collect scalar experimental and event-level parameters
    into simple arrays for plotting or statistics.

    Parameters
    ----------
    plants : list[Plant]
    events : list[Event]

    Returns
    -------
    data : dict[str, np.ndarray]
    """

    data = {}

    # Plant-level

    data["T_cn"] = np.array([
        getattr(p, "avgT", np.nan) for p in plants
    ])
    all_E = []
    for p in plants:
        if hasattr(p, "E_sections"):
            all_E.extend(p.E_sections.values())
    data["E"] = np.array(all_E)

    all_R = []
    for p in plants:
        if hasattr(p, "r_sections"):
            all_R.extend(p.r_sections.values())
    data["R"] = np.array(all_R)

    data["max_curvature"] = np.array([
        np.max(p.free_shape["curvature_cm"]) if hasattr(p, "free_shape") 
        and "curvature_cm" in p.free_shape else np.nan
        for p in plants
    ])


    # Event-level
    data["contact_R"] = np.array([
        getattr(ev, "contact_R", np.nan) for ev in events
    ])
    data["contact_E"] = np.array([
        getattr(ev, "contact_E", np.nan) for ev in events
    ])
    data["ltip"] = np.array([
        getattr(ev, "ltip", np.nan) for ev in events
    ])
    data["L_base"] = np.array([
        getattr(ev, "Lbase", np.nan) for ev in events
    ])

    return data