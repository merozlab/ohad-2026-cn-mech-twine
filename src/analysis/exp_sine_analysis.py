import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, UnivariateSpline

import bootstrap # sets up PROJCETROOT path for imports 
from src.utils.curve_tools import gaussian_smooth, despike, calc_R2_RMSE
from src.utils.array_tools import closest
from src.utils.math_tools import sine_2param

def fit_events_sine_2param(
    events,
    window_size=25,
    start_indx=8,
    pTcn=0.75,
    t_cutoff=10,
    A_fit=0.1,
    w_fit=0.03,
    spike_threshold=5.0,
    spike_window = 5,
    spike=False
):
    results = []

    for ev in events:
        try:
            f_raw = ev.F_bean
            # Despike before smoothing
            if spike:
                f_despiked, _ = despike(
                    f_raw,window=spike_window,threshold=spike_threshold,replace="median"
                )
            else:
                f_despiked = f_raw
            f_smooth = gaussian_smooth(
                f_despiked, win_size=window_size, start_indx=start_indx
            )

            n_fit = closest(ev.timer, ev.plnt.avgT * 60 * pTcn)
            t_fit = ev.timer[:n_fit] / 60.0  # minutes
            y_fit = f_smooth[:n_fit]

            # Drop NaNs from fit vectors
            valid = (~np.isnan(t_fit)) & (~np.isnan(y_fit))
            t_fit = t_fit[valid]
            y_fit = y_fit[valid]

            if len(t_fit) < t_cutoff:
                raise ValueError("insufficient points")

            # Initial guesses
            A0 = float(np.nanmax(np.abs(y_fit))) if len(y_fit) else A_fit
            A0 = A0 if A0 > 0 else A_fit
            w0 = w_fit

            popt, pcov = curve_fit(
                sine_2param, t_fit, y_fit,
                p0=[A0, w0]
            )
            # sine_2param, t_fit, y_fit, p0=[np.max(y_fit), 0.03]
            r2, rmse, Std = calc_R2_RMSE(t_fit, y_fit, popt, sine_2param)

        except Exception:
            popt = [np.nan, np.nan]
            r2, rmse, Std = np.nan, np.nan, np.nan
            t_fit, y_fit = np.array([]), np.array([])

        ev.sine_fit_2param = {
            "A": popt[0], # fit amplitude
            "w": popt[1], # fit angular frequency
            "T": 2 * np.pi / popt[1] if not np.isnan(popt[1]) and popt[1] != 0 else np.nan, # fit period
            "R2": r2, # coefficient of determination
            "rmse": rmse, # root mean square error
            "Std": Std, # standard error
            "pTcn": pTcn,
            "window_size": window_size,
            "start_indx": start_indx,
            "n_fit": len(t_fit),
            "t_fit": t_fit, # fitted time (minutes)
            "y_fit": y_fit, # smoothed force used for fitting
        }

        results.append({
            "event": ev,
            "A": popt[0],
            "w": popt[1],
            "R2": r2,
            "t_fit": t_fit,
            "y_fit": y_fit,
        })

    return results


def normalize_smoothed_trj(events, t_max=0.3, n_points=100, spline_k=2):
    """
    Normalize smoothed fit segments by amplitude (A) and period (T),
    then stack onto a common normalized time grid using UnivariateSpline.
    Returns dict with t grid, mean, std, and all stacked curves.
    """
    t_common = np.linspace(0.0, t_max, n_points)
    stacked = []

    for ev in events:
        meta = getattr(ev, "sine_fit_2param", {})
        A = meta.get("A", np.nan)
        T = meta.get("T", np.nan)
        t_fit = meta.get("t_fit", np.array([]))
        y_fit = meta.get("y_fit", np.array([]))
        if np.isnan(A) or np.isnan(T) or A == 0:
            continue

        # Normalize time by period and force by amplitude
        t_norm = t_fit / T
        f_norm = y_fit / A

        # Drop NaNs and ensure monotonic increasing x for spline
        valid = (~np.isnan(t_norm)) & (~np.isnan(f_norm))
        t_norm = t_norm[valid]
        f_norm = f_norm[valid]
        if len(t_norm) < 4:
            continue
        try:
            interp = interp1d(t_norm, f_norm, 
                                kind='linear', bounds_error=False, 
                                fill_value=np.nan)
            y_grid = interp(t_common)
        except Exception:
            continue
        stacked.append(y_grid)

    stacked = np.array(stacked) if len(stacked) else np.empty((0, n_points))
    mean = np.nanmean(stacked, axis=0) if stacked.size else np.zeros_like(t_common)
    std = np.nanstd(stacked, axis=0) if stacked.size else np.zeros_like(t_common)

    return {
        "t_common": t_common,
        "mean": mean,
        "std": std,
        "all": stacked,
    }
