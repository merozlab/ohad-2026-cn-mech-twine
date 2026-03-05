# fit to sine 2 param
# F(t) vs ltip and vs E 
# max F vs ltip and E
# time to zero force vs ltip and E
# save to cache
# note in 2025 data file time is normalized by Tcn

#%%
import numpy as np
from scipy.optimize import curve_fit

import bootstrap # sets up PROJCETROOT path for imports 
from src.io.data_imports import load_simulation_pickle
from src.core.sim_event import SimulationEvent

from src.utils.curve_tools import calc_R2_RMSE
from src.utils.array_tools import closest
from src.utils.math_tools import sine_2param

# from scipy.optimize import fsolve
# def func(x,LOC,growth_zone):
#     return [ (1-np.cos(growth_zone*x[0]))/x[0]-LOC]
# root=fsolve(func, [0.5*KAPPA0])
# kappa_fit.append(root[0])

def build_simulation_events(DATA_PATH):
        
    sim_pkl = load_simulation_pickle(DATA_PATH)

    MU_VEC = np.array(sim_pkl[0])
    E_VEC = np.array(sim_pkl[1])
    F_VECS = np.array(sim_pkl[2])
    T_VEC = np.array(sim_pkl[3])
    F_LOC_VECS_B = np.array(sim_pkl[4])
    F_LOC_VECS_T = np.array(sim_pkl[5])
    B_VEC = np.array(sim_pkl[6])
    LOC_VECS = np.array(sim_pkl[7])
    F_VECS_planar = np.array(sim_pkl[8])
    KAPPA0=0.04
    growth_zone=50.0 # is full organ length!
    sim_events=[]

    for ind in range(0,len(MU_VEC)):
        # root=fsolve(func, [0.5*KAPPA0])
        # kappa_fit.append(root[0])
        # F_loc_vec_bottom=F_LOC_VECS_B[ind]
        # F_loc_vec_top=F_LOC_VEC_T[ind]
        # t_contact = np.where(F_loc_vec_bottom>0)[0][0]
        # T_shifted = T_VEC[ind][t_contact:] - T_VEC[ind][t_contact]
        # l_tip = growth_zone - np.arccos(1 - KAPPA0 * LOC_VECS[ind]) / KAPPA0
        
        sim_ev = SimulationEvent(
            mu=MU_VEC[ind],
            E=E_VEC[ind],
            F_vec=F_VECS[ind],
            T_vec=T_VEC[ind],
            F_loc_vec_bottom=F_LOC_VECS_B[ind],
            F_loc_vec_top=F_LOC_VECS_T[ind],
            B=B_VEC[ind],
            LOC=LOC_VECS[ind],
            F_vec_planar=F_VECS_planar[ind],
            kappa0=KAPPA0,
            growth_zone=growth_zone)
        sim_ev.calc_sim_vars()
        sim_events.append(sim_ev)
    return sim_events


def fit_sim_events_sine_2param(sim_events, guess = None,p_T = 0.25):
    ''' Fitting simulations to sine- guess of "amp","omega" '''

    results = []
    
    for ev in sim_events:
        # Initial guesses
        if not guess:
            A0 = float(np.nanmax(np.abs(ev.F_shifted))) 
            w0 = 0.1
        else:    
            A0 = guess[0]
            w0 = guess[1]

        # fit variables
        nTcn = closest(ev.T_shifted, p_T) # find index closest to p_T normalized time
        t_fit = ev.T_shifted[:nTcn]
        y_fit = ev.F_shifted[:nTcn]

        try:
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
            print("sine fit failed")

        ev.sine_fit_2param = {
            "A": popt[0], # fit amplitude
            "w": popt[1], # fit angular frequency
            "T": 2 * np.pi / popt[1] if not np.isnan(popt[1]) and popt[1] != 0 else np.nan, # fit period
            "R2": r2, # coefficient of determination
            "rmse": rmse, # root mean square error
            "Std": Std, # standard error
            "t_fit": t_fit, # fitted time (minutes)
            "y_fit": y_fit, # smoothed force used for fitting
        }

        results.append({
            "event": ev,
            "t_fit": t_fit,
            "y_fit": y_fit,
            "A": popt[0],
            "w": popt[1],
            "R2": r2
        })

    return results
