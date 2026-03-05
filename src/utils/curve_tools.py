from matplotlib import axes
from scipy.ndimage import gaussian_filter1d
import numpy as np
import pandas as pd
import scipy.optimize

from src.utils.math_tools import (
        linfunc,
        powerfunc,
        polyfunc,
        one_over_x,
)


def gaussian_smooth(data ,win_size=3, start_indx=2):
    '''smooth with 1D gaussian filter'''
    try:
        start_indx = max(1,min(start_indx, int(len(data)/2))) # ensure start index
        # Ensure window length
        win_size = max(2, min(win_size, int(len(data[start_indx:]) / 4)))
        data_pd = np.array(data)

        # Gaussian smoothing
        f_temp = gaussian_filter1d(data_pd, sigma=win_size)

        # replace initial values with original data
        if start_indx > 0:
            f_temp[:2*start_indx] = data[:2*start_indx]

        # Replace infs with nan for uniformity
        f_temp[np.isinf(f_temp)] = np.nan

        return f_temp
    except Exception as e:
        print(f"Error in smoothing data: {e}")


def despike(
    y,
    window=5,
    threshold=3.0,
    replace="median"
):
    """
    Remove spikes from 1D time series using local median filtering.

    Parameters
    ----------
    y : array-like
        Input time series
    window : int
        Half-window size (total window = 2*window + 1)
    threshold : float
        Spike threshold in units of MAD
    replace : str
        'median' or 'interp'

    Returns
    -------
    y_clean : np.ndarray
        Despiked signal
    mask : np.ndarray
        Boolean mask of detected spikes
    """

    y = np.asarray(y)
    y_clean = y.copy()
    n = len(y)

    mask = np.zeros(n, dtype=bool)

    for i in range(window, n - window):
        segment = y[i - window : i + window + 1]
        median = np.median(segment)
        mad = np.median(np.abs(segment - median))

        if mad == 0:
            continue

        if np.abs(y[i] - median) > threshold * mad:
            mask[i] = True
            if replace == "median":
                y_clean[i] = median

    if replace == "interp":
        idx = np.arange(n)
        y_clean[mask] = np.interp(idx[mask], idx[~mask], y[~mask])

    return y_clean, mask


def fit_w_error(x,dx,y,dy,fit_func = linfunc):
    '''Fit data to the provided function and plot with error bars.

    Parameters:
    ax : matplotlib axis object
    x, dx : array-like, data and errors in x
    y, dy : array-like, data and errors in y
    fit_func : callable, function to fit the data

    Returns:
    popt : array, optimal values for the parameters
    pcov : 2D array, the estimated covariance of popt
    goodness of fit: R^2 and chi^2_red
    '''

    # fit with errors via curve fit (use odr?)
    sigma = np.sqrt(dx**2 + dy**2)
    popt,pcov = scipy.optimize.curve_fit(fit_func, x, y,
                p0=None, sigma=sigma, absolute_sigma=True)

    # goodness of fit
    ss_total = np.sum((y - np.mean(y)) ** 2)
    line_of_best_fit =  fit_func(x, *popt) # popt[0] * x + popt[1]
    residuals = y - line_of_best_fit
    ss_residual = np.sum(residuals**2)

    # calc reduced chi squared with errors in x and y
    chi_sq_red = np.sum(residuals**2 / \
                            (dy**2+(fit_func(x+dx,*popt)- fit_func(x-dx,*popt)) **2))/ \
                            (len(x)-len(popt))
    chi_sq_std = np.sqrt(2/(len(x)-len(popt)))
    r2 = 1 - (ss_residual / ss_total)
    fit_results = {
        'popt': popt,
        'pcov': pcov,
        'r2': r2,
        'chi_sq_red': chi_sq_red,
        'chi_sq_std': chi_sq_std
    }
    return fit_results

def calc_R2_RMSE(tt, yy, popt, fit_func):
    if any(np.isnan(popt)):
        return np.nan, np.nan, np.nan
    residuals = yy - fit_func(tt, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yy - np.mean(yy))**2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(ss_res / (len(yy) - len(popt)))
    Std = np.sqrt(ss_res / (len(yy) - len(popt) - 1)) # Standard error
    return r2, rmse, Std


def compact_letter_display2(pairs, p_values, labels, alpha=0.05):
    """
    Generate Compact Letter Display (CLD) for statistical groupings with selective propagation.

    Args:
        pairs (list of tuple): List of index pairs representing comparisons between groups.
        p_values (list of float): List of p-values corresponding to each pairwise comparison.
        labels (list of str): List of group labels (e.g., ['Group 1', 'Group 2', ...]).
        alpha (float): Significance level for determining differences.

    Returns:
        dict: Dictionary mapping each group label to its corresponding letter(s).
    """
    n = len(labels)
    significance_matrix = np.ones((n, n), dtype=bool)  # True means not significantly different

    # Fill the significance matrix: False where groups are significantly different
    for (i, j), p_val in zip(pairs, p_values):
        if p_val < alpha:  # Significant difference
            significance_matrix[i, j] = False
            significance_matrix[j, i] = False
    print("\nInitial Significance Matrix (True = Not Significant):")
    print(significance_matrix)

    # Initialize letters
    final_letters = {label: "" for label in labels}
    letter_sources = {label: set() for label in labels}  # Tracks which groups contribute each letter
    used_letters = []

    # Assign letters
    for i in range(n):
        print(f"\nProcessing group {labels[i]}...")
        current_letters = set()

        for j in range(n):
            if significance_matrix[i, j]:  # Groups i and j are NOT significantly different
                # Share only the letters that correspond to groups mutually not significantly different
                for letter in final_letters[labels[j]]:
                    origin_group = letter_sources[labels[j]]
                    if all(significance_matrix[k][i] for k in origin_group):
                        current_letters.add(letter)

        if not current_letters:
            # Assign a new letter if no shared letters exist
            new_letter = chr(ord('a') + len(used_letters))
            used_letters.append(new_letter)
            current_letters.add(new_letter)
            letter_sources[labels[i]].add(i)  # Track the origin of this letter
            print(f"  Assigning new letter '{new_letter}' to group {labels[i]}")

        # Update group i with the current letters
        final_letters[labels[i]] = "".join(sorted(current_letters))
        letter_sources[labels[i]].update({i for letter in current_letters if letter in final_letters[labels[i]]})

        # Propagate the current letters to all groups not significantly different
        for j in range(n):
            if significance_matrix[i, j]:  # i and j are not significantly different
                for letter in current_letters:
                    origin_group = letter_sources[labels[i]]
                    if all(significance_matrix[k][j] for k in origin_group):
                        final_letters[labels[j]] += letter
                        final_letters[labels[j]] = "".join(sorted(set(final_letters[labels[j]])))
                        letter_sources[labels[j]].add(i)

        print(f"  Final letters for group {labels[i]}: {final_letters[labels[i]]}")

    # Debug output: Final letters for all groups
    print("\nFinal Letter Assignments:")
    for label, letter in final_letters.items():
        print(f"Group {label}: {letter}")

    # Debug output: P-Values and Pairs
    print("\nPairwise Comparisons (Group Labels, P-Value):")
    for (i, j), p_val in zip(pairs, p_values):
        print(f"  Groups: ({labels[i]}, {labels[j]}), P-Value: {p_val:.3f}")

    return final_letters

def multi_range_fit(x, y, ranges=None, fit_func=linfunc, 
                    dx=None, dy=None, fitwerr=False,
                    plot=False, ax=None):      
    """
    Fit multiple ranges of data with the specified fitting functions    .

    Parameters:
    x : array-like
        Independent variable data.
    y : array-like
        Dependent variable data.
    ranges : list of tuple
        List of (start, end) tuples defining the ranges to fit.
    fit_func : callable or list of callables
        Functions to fit the data.
    dx : array-like, optional
        Errors in x data.
    dy : array-like, optional
        Errors in y data.
    fitwerr : bool
        If True, use fit_w_error for fitting with errors.
        
    Returns:
    list of tuple
        List of (popt, pcov) for each fitted range.
    """
    fit_results = []
    if ranges is None: # if no range given, single range over all data
        ranges = [(x[0], x[-1])]
    if not isinstance(fit_func, list): # single function provided
        fit_func = [fit_func] * len(ranges)
    elif len(fit_func) != len(ranges): # if func # mismatch, repeat linfunc for rest
        fit_func = fit_func + [linfunc] * (len(ranges) - len(fit_func))

    for i, (start, end) in enumerate(ranges):  
        mask = (x >= start) & (x <= end)
        x_range = x[mask]
        y_range = y[mask]
        if len(x_range) < 2:
            fit_results.append((None, None))
            continue
        if fitwerr and dx is not None and dy is not None:
            dx_range = dx[mask]
            dy_range = dy[mask]
            fit_results.append(
                fit_w_error(x_range, dx_range, y_range, dy_range, fit_func=fit_func[i])
                )
            popt = fit_results[-1]['popt']
        else:
            popt, pcov = scipy.optimize.curve_fit(fit_func[i], x_range, y_range)
            fit_results.append((popt, pcov))

        if plot: # plot fit line
            x_range = np.linspace(start, end, 100)
            y_fit = fit_func[i](x_range, *popt)
            ax.plot(x_range, y_fit, color='black', linestyle='--')

    return fit_results

#popt,pcov,r2,chi_sq_red, chi_sq_std