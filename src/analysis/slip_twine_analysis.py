'''
    calculate twine or slip times for exp, sim, xtrm?, motor events
    using h5 analysis or 2-point tracking -> angle near support
'''

#%% imports
import numpy as np
import re
from src.physics.forces import (
    calc_effective_resistance,
)
from src.io.read_tracking_file import (
    get_tracked_data,
)

import bootstrap
from scipy.ndimage import gaussian_filter1d
#%% definitions

def attach_slip_twine_to_events(events, h5_results, h5_file_list, track_dict):
    """
    Match twine results to Events and attach corrected timing.
    """

    for ev in events:
        if ev.twine_state == 0:
            slip_time_min = ev.timer[-1]/60 # slip time in minutes
            slip_time_norm = slip_time_min / ev.plnt.avgT # normalized to plant's average period
            contact_start_time = np.nan
            twine_est_min = np.nan  # Not applicable for slip events
            twine_est_norm = np.nan  # Not applicable for slip events
            h5_compensation = np.nan  # Not applicable for slip events

        else: 
            key = (ev.exp_num, ev.event_num)

            if key not in h5_results:
                print(f"Warning: No twine result for event {key}. Skipping.")
                continue

            slip_time_min = np.nan  # Not applicable for twine events
            slip_time_norm = np.nan  # Not applicable for twine events
            twine_est_raw = h5_results[key]["twine_estimate_raw"] 
            save_till_h5 = h5_results[key]["frames_till_h5"]
            computed_till_h5 = h5_twine_initiation_compensation(
                                h5_file_list=h5_file_list,
                                exp_num=ev.exp_num,
                                event_num=ev.event_num, 
                                track_dict=track_dict,
            )
            # print(f"Event {key}: computed_till_h5={computed_till_h5}, "
            #       f"save_till_h5={save_till_h5}")

            # Compensation in minutes for diff between computed and saved frames_till_h5
            if computed_till_h5 is not None and save_till_h5 is not None:
                if np.isfinite(computed_till_h5) and np.isfinite(save_till_h5):
                    h5_compensation = (computed_till_h5 - save_till_h5)/2
            else: 
                h5_compensation = computed_till_h5/2 if computed_till_h5 is not None else np.nan

            # Correct relative to contact
            if hasattr(ev, "frm0_side"):
                contact_start_time = ev.frm0_side / 2  # minutes
                twine_est_min = (twine_est_raw - contact_start_time + h5_compensation) 
                twine_est_norm = twine_est_min / ev.plnt.avgT
            else:
                contact_start_time = np.nan
                twine_est_min = np.nan
                twine_est_norm = np.nan
            
        ev.twine_data = {
        # return{
            "slip_time": slip_time_min,
            "slip_time_norm": slip_time_norm,
            "twine_estimate_min": twine_est_min,
            "twine_estimate_norm": twine_est_norm,
            "contact_start_time": contact_start_time,
            "h5_compensation": h5_compensation
            }

def extract_slip_twine_scalars(events):
    """
    Returns a dict of numpy arrays with aligned indices for slip/twine state, timing, geometry, and mechanics.
    """
    event_time = []
    ltip = []
    twine_state = []  # 0 = slip, 1 = twine
    max_force = []
    max_torque = []

    for ev in events:
        if not hasattr(ev, "twine_state") or not hasattr(ev, "twine_data"):
            continue

        twine_state.append(ev.twine_state)

        # time
        if ev.twine_state == 1:
            t = ev.twine_data["twine_estimate_norm"]
        else:
            t = ev.twine_data["slip_time_norm"]
        event_time.append(t)

        # geometry
        ltip.append(getattr(ev, "ltip", np.nan))

        # mechanics
        if hasattr(ev, "F_bean"):
            maxF = np.nanmax(ev.F_bean)
        else:
            maxF = np.nan
        max_force.append(maxF)

        # torque = force * moment arm (Lbase)
        lbase = getattr(ev, "Lbase", np.nan)
        max_torque.append(maxF * lbase if not np.isnan(maxF) and not np.isnan(lbase) else np.nan)

    return {
        "twine_state": np.array(twine_state),
        "event_time": np.array(event_time),
        "ltip": np.array(ltip),
        "max_force": np.array(max_force),
        "max_torque": np.array(max_torque),
    }

def get_slip_twine_arrays(events):
    '''Extract aligned arrays for slip/twine state, timing, and geometry.'''
    state = np.array([ev.twine_state for ev in events])

    slip = [ev for ev in events if ev.twine_state == 0]
    twine = [ev for ev in events if ev.twine_state == 1]

    slip = np.array(slip, dtype=object)
    twine = np.array(twine, dtype=object)
    return state, slip, twine
#%% h5 compensation function

def h5_twine_initiation_compensation(h5_file_list, exp_num, event_num, track_dict):
    '''from track file get starting image number,
    (from events-class get index of contact index,)
    from h5 file name get number of first analyzed image, 
    return difference between h5 start and first frame'''
    key = (exp_num, event_num)
    # Case 1: h5_file_list is a dict of (exp,event)->{'first_frame':...}
    if isinstance(h5_file_list, dict) and key in h5_file_list and "first_frame" in h5_file_list[key]:
        h5_first_image = int(h5_file_list[key]["first_frame"])
    else:
        # Fallback: parse from filenames
        pattern = rf'interekt_{exp_num}_e_{event_num}.*'
        h5_files = [
            match for file_list in h5_file_list.values()
            for match in re.findall(pattern, file_list[0])
        ]
        if not h5_files:
            raise ValueError(f"No h5 files found for exp {exp_num}, event {event_num}")
        h5_first_image_match = re.findall(r'\d{3,5}-', h5_files[0])
        if not h5_first_image_match:
            raise ValueError(f"Could not parse h5 filename: {h5_files[0]}")
        h5_first_image = int(h5_first_image_match[0].rstrip('-'))
    
    # Find track file for target exp-event
    track_file_name = None
    direct_key = (exp_num, event_num, 'side')
    if isinstance(track_dict, dict) and direct_key in track_dict:
        track_file_name = track_dict[direct_key]
    else:
        # Fallback: previous scan logic for list-of-lists format
        all_track_files = list(track_dict.values())
        for file_list in all_track_files:
            file_path = (file_list[0] if isinstance(file_list, (list, tuple)) else file_list).replace('\\', '_')
            if (re.search(rf'0{exp_num}', file_path) and 
                re.search(rf'_{event_num}', file_path) and 
                'side' in file_path):
                track_file_name = file_list[0] if isinstance(file_list, (list, tuple)) else file_list
                break

    if not track_file_name:
        raise ValueError(f"No track file found for exp {exp_num}, event {event_num}")
    
    # Get first frame number from track file
    with open(track_file_name, 'r') as file:
        first_line = file.readline()
    
    dsc_match = re.findall(r'DSC_(\d+)', first_line)
    if dsc_match:
        zero_frame = int(dsc_match[0])
    else:
        crop_match = re.findall(r'(\d+)_CROPED', first_line)
        if crop_match:
            zero_frame = int(crop_match[0])
        else:
            raise ValueError(f"Could not parse track file first line: {first_line}")
    
    frms_till_h5 = h5_first_image - zero_frame
    # print(f"Frames till h5: {frms_till_h5}, h5 first image: {h5_first_image}, "
    #         f"track first image: {zero_frame}, exp {exp_num}, event {event_num}")
    return frms_till_h5

#%% twine F-effective resistance calculation
def get_twine_Feff(exp_events,xtrm_events):
    '''calculate effective resistance using function for 
    pendulum and xtrm events and twine state'''
    Feff = []
    twine_states = []
    ltips = []
    for ev in exp_events:
        Feff.append(calc_effective_resistance(ev))
        twine_states.append(ev.twine_state)
        ltips.append(ev.ltip)
    
    for xev in xtrm_events:
        Feff.append(calc_effective_resistance(xev))
        twine_states.append(xev.twine_state)
        ltips.append(xev.cont2stemtip_dist_cm)

    return Feff, twine_states, ltips
#%% extract_non_h5_twine_times

def extract_non_h5_twine_times(non_h5_events):
    '''For events without h5 twine analysis, 
        estimate twine initiation time using contact start time and angle data from tracking'''
    non_h5_results = {}
    for ev in non_h5_events:
        if ev.twine_state == 1 and hasattr(ev, "frm0_side"):
            # Estimate twine initiation as contact start plus time to reach 45 or 50 deg
            non_h5_results[(ev.exp_num, ev.event_num)] = analyze_two_point_angle(ev.track_file)
            # estimated_twine_time = ev.frm0_side / 2 + 0.5
            # non_h5_results[(ev.exp_num, ev.event_num)] = {
            #     "twine_estimate_min": estimated_twine_time,
            #     "contact_start_time": ev.frm0_side / 2
            # }
    return non_h5_results


def compute_angle_vs_horizontal(p0, p1):
    """
    Returns angle in degrees between line (p0->p1) and horizontal.
    """
    dx = p1[:, 0] - p0[:, 0]
    dy = p1[:, 1] - p0[:, 1]

    angles = np.degrees(np.arctan2(dy, dx))

    return angles

def smooth_angle(angle, sigma=2):
    """
    Gaussian smoothing of angle signal.
    sigma is in number of frames.
    """
    return gaussian_filter1d(angle, sigma=sigma)

def find_first_threshold_crossing(times, angle, threshold, min_frames=3):
    for i in range(len(angle) - min_frames):
        window = angle[i:i+min_frames]
        if np.all(window >= threshold):
            return angle[i], times[i] / 60, i

    return np.nan, np.nan, np.nan

def analyze_two_point_angle(filepath,
                            threshold=50,
                            sigma=2,
                            min_frames=3):
    """
    Full pipeline:
        load → compute angle → smooth → detect crossing
    """

    times, p0, p1 = load_two_point_track(filepath)

    raw_angle = compute_angle_vs_horizontal(p0, p1)

    smooth_ang = smooth_angle(raw_angle, sigma=sigma)

    angle_cross, time_cross, idx = find_first_threshold_crossing(
        times, smooth_ang, threshold, min_frames=min_frames
    )

    return {
        "times_sec": times,
        "raw_angle": raw_angle,
        "smooth_angle": smooth_ang,
        "cross_angle": angle_cross,
        "cross_time_min": time_cross,
        "cross_index": idx
    }

#%%