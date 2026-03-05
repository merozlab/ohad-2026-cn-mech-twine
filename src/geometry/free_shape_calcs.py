import numpy as np
from scipy.ndimage import gaussian_filter1d

# x,y coordinates
# x,y = coords_centered[:, 0], coords_centered[:, 1]

def compute_s_ds(x,y):
    """Compute arc-length parameter s and differential ds along skeleton."""
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s = np.concatenate(([0], np.cumsum(ds)))
    return s, ds

def compute_angle_from_coord(x,y,smooth_frac=10):
    """Compute angles between segments of the skeleton, and smooth"""
    # Compute angles between segments (relative to vertical)
    angles = np.arctan2(np.diff(x), np.diff(y)) # in radians
    win = len(angles)//smooth_frac # Smoothing window size
    smooth_angles = gaussian_filter1d(angles, sigma=win)  # Smooth angles
    return smooth_angles

def compute_curvature_from_angle(angle,ds,smooth_frac=10):
    """Compute curvature from angle between segments."""
    dtheta = np.diff(angle)  # Absolute change in angle between segments in radians
    curvature = np.divide(dtheta, ds[1:])  # Curvature is change in angle per unit length
    win = len(curvature)//smooth_frac # Smoothing window size
    smooth_curvature = gaussian_filter1d(curvature, sigma=win)  # Smooth curvature
    return smooth_curvature 

def compute_Lbase(lskel, height, yskel, ltip):
    """Compute base length Lbase from lengths: 
    lskel - skeleton length, 
    height - measured plant height,
    yskel - height of extracted skeleton, 
    ltip  - stem length beyond contact
    """
    Lbase = lskel + height - yskel - ltip
    return Lbase


    
# Load skeleton and curvature from saved Excel
# skeleton_df = data_xl["Skeleton Analysis"]
# x_reoriented = skeleton_df['x_reoriented(pix)'].values
# y_reoriented = skeleton_df['y_reoriented(pix)'].values
# s_pix = skeleton_df['s(pix)'].values
# curvature_pix = skeleton_df['curvature_smooth(1/pix)'].values

# # Remove NaN values from curvature array for calculations
# valid_curvature_mask = ~np.isnan(curvature_pix)
# curvature_pix_valid = curvature_pix[valid_curvature_mask]

# # Convert to real-world units using new calibration
# s_tot_new = max(s_pix) * pix2cm_new
# curvature_cm_new = curvature_pix_valid / pix2cm_new

# # Compute vertical and horizontal distance between 2 farthest points in skeleton (in real-world units)
# yskel_new = (np.max(y_reoriented) - np.min(y_reoriented))* pix2cm_new
# xskel_new = (np.max(x_reoriented) - np.min(x_reoriented))* pix2cm_new

# Lbase_new = s_tot_new + height - yskel_new - l_tip
