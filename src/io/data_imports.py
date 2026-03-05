import re
# import glob
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
from src.io.paths import PROJECT_ROOT
# -------------------------------------------------
# Plotting style
# -------------------------------------------------
def apply_project_style():
    style_path = PROJECT_ROOT / "src" / "plotting" / "stylesheet_jxb.mplstyle"
    plt.style.use(style_path)

# -------------------------------------------------
# Metadata
# -------------------------------------------------

def load_metadata(DATA_PATH):
    """
    Load and clean the Excel metadata table for experimental events.
    Replicates legacy filtering logic.
    """
    excel_path = DATA_PATH / "Exp2_supplementary_measurements_events_xl.xlsx"
    data_panda = pd.read_excel(excel_path)

    delete_rows = []
    problem_exp = []

    for i in range(len(data_panda)):
        if (
            data_panda.at[i, "problem"] != "na"
            or data_panda.at[i, "Bean_Strain"] != "Helda"
        ):
            delete_rows.append(i)
            problem_exp.append(data_panda.at[i, "Exp_num"])

    data_panda = (
        data_panda
        .drop(data_panda.index[delete_rows])
        .reset_index(drop=True)
    )

    return data_panda, problem_exp


# -------------------------------------------------
# Tracking files
# -------------------------------------------------

def load_support_tracking(DATA_PATH):
    """
    Load support bottom tracking files for pendulum events
    Returns dict keyed by (exp, event, view).
    """
    track_folder_path = DATA_PATH / "track_logs"
    track_dict = {}

    for file in track_folder_path.glob("*.txt"):
        file = str(file)
        exp = int(re.findall(r"_\d{3,4}_", file)[0].replace("_", ""))
        event = int(re.findall(r"_\d\D", file)[0][1])
        viewt = re.findall(r"(side|top)", file)[0]
        track_dict[(exp, event, viewt)] = file

    return track_dict


def load_contact_tracking(DATA_PATH):
    """
    Load stem–support contact tracking files for pendulum events
    Returns dict keyed by (exp, event).
    """
    contact_folder_path = DATA_PATH / "contact_track_logs"
    contact_dict = {}

    for file in contact_folder_path.glob("*.txt"):
        file = str(file)
        exp = int(re.findall(r"_\d{3,4}_", file)[0].replace("_", ""))
        event = int(re.findall(r"_\d_", file)[0].replace("_", ""))
        contact_dict[(exp, event)] = file

    return contact_dict


# -------------------------------------------------
# Young's modulus
# -------------------------------------------------

def load_E_dict(DATA_PATH):
    """
    Load Young's modulus CSV files.
    Returns dict: exp_num -> DataFrame
    """
    E_folder_path = DATA_PATH / "Young_moduli"
    E_dict = {}

    for file in E_folder_path.glob("*.csv"):
        exp = int(re.findall(r"\d{2,4}", file.name)[0])
        E_dict[exp] = pd.read_csv(file, header=None)

    return E_dict

# -------------------------------------------------
# Free shape data imports
# -------------------------------------------------
def load_skeleton_xl(DATA_PATH):
    """
    Returns:
        skeleton_df
        recalculated_df
    """
    skeleton_folder_path = DATA_PATH / "side_view_cn_perpendicular/saved_variables"
    skeleton_dict = {}

    for file in tqdm(skeleton_folder_path.glob("*.xlsx")):
        skeleton_xl_path = file
        skeleton_df = pd.read_excel(skeleton_xl_path, sheet_name="Skeleton Analysis")
        recalculated_df = pd.read_excel(skeleton_xl_path, sheet_name="Recalculated lengths")
        exp = int(re.findall(r"\d{1,3}", file.name)[0])
        skeleton_dict[exp] = (skeleton_df, recalculated_df)
    return skeleton_dict

# -------------------------------------------------
# Simulation data imports
# -------------------------------------------------
def load_simulation_pickle(DATA_PATH):
    """ Load simulation data from pickle file. """
    path = DATA_PATH / "simulations" / "twining23_3_July25.dat"
    with open(path, "rb") as f:
        return pickle.load(f)


# -------------------------------------------------
# Twine timing data imports
# -------------------------------------------------
def get_h5_frames(txt_path):
    """
    Get H5 file frame ranges from a text file listing 'interekt' filenames.

    Args:
        txt_path (Path | str): Path to a text file with one filename per line.

    Returns:
        dict[(exp_num, event_num)] = {"first_frame": int, "last_frame": int}
        where first_frame is the smallest start and last_frame is the largest end.
    """
    h5_frame_dict = {}

    with open(txt_path, "r", encoding="utf-8") as f:
        filenames = [line.strip() for line in f if line.strip()]

    for name in filenames:
        exp_m = re.search(r"interekt_(\d{1,3})_", name)
        event_m = re.search(r"_e_(\d{1,2})_", name)
        first_m = re.search(r"_(\d{1,5})-", name)
        last_m = re.search(r"-(\d{1,5})\.h5$", name)

        if not (exp_m and event_m and first_m and last_m):
            continue

        exp = int(exp_m.group(1))
        event = int(event_m.group(1))
        first_frame = int(first_m.group(1))
        last_frame = int(last_m.group(1))

        key = (exp, event)
        if key not in h5_frame_dict:
            h5_frame_dict[key] = {"first_frame": first_frame, "last_frame": last_frame}
        else:
            h5_frame_dict[key]["first_frame"] = min(h5_frame_dict[key]["first_frame"], first_frame)
            h5_frame_dict[key]["last_frame"] = max(h5_frame_dict[key]["last_frame"], last_frame)

    return h5_frame_dict


def load_h5_twine_results(DATA_PATH):
    """
    Load twine timing results from analyzed H5 files 

    Returns
    -------
    dict[(exp, event)] -> dict
    """
    results = {}
    h5_folder = DATA_PATH / "twine_init_logs" / "2025_analysis"
    h5_files = h5_folder.glob("h5_results_*.h5")

    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            exp = int(f.attrs.get('exp', None))
            event = int(f.attrs.get('event', None))
            t = f['t'][:]
            angle = f['event_theta'][:]
            smoothed_angle = f['smoothed_event_theta'][:]
            twine_estimate = float(f['twine_estimate'][0])
            till_h5 = f.attrs.get('till_h5', None)
            threshold = f.attrs.get('threshold', None)

            results[(exp, event)] = {
                "time": t,
                "angle": angle,
                "smoothed_angle": smoothed_angle,
                "twine_estimate_raw": twine_estimate,
                "frames_till_h5": till_h5,
                "threshold": threshold,
                "h5_file": h5_file,
            }

    return results

# -------------------------------------------------
# Extreme mass exp data imports
# -------------------------------------------------

def load_xtrm_data(DATA_PATH):
    """
    Load extreme support-mass experiment Excel files.

    Returns:
        dict[(exp_num, mass_label)] = {
            "df": DataFrame,
            "mass_label": "light" | "stable"
        }
    """
    xtrm_folder = DATA_PATH / "extreme_msup"
    xtrm_dict = {}

    for file in xtrm_folder.glob("*.xlsx"):
        df = pd.read_excel(file)

        exp = int(re.findall(r"\d{1,3}", file.name)[0])

        fname = file.name.lower()
        if "light" in fname:
            mass_label = "light"
        elif "stable" in fname:
            mass_label = "stable"
        else:
            raise ValueError(f"Cannot infer support mass from filename: {file.name}")

        xtrm_dict[(exp, mass_label)] = {
            "df": df,
            "mass_label": mass_label
        }

    return xtrm_dict

# -------------------------------------------------
# Motor experiment data imports
# -------------------------------------------------

def load_motor_data(DATA_PATH):
    """
    Load motor experiment Excel files.

    Returns:
        dict[(exp_num, mass_label)] = {
            "df": DataFrame,
            containing event metadata on rotation, slip/twine state
        }
    """
    motor_xl_path = DATA_PATH / "motor_exp" / "motor mod_misx_new.xlsx"
    motor_xl = pd.read_excel(motor_xl_path, sheet_name="meta-data motor_mod")
    
    motor_track_dict = {}
    motor_folder = DATA_PATH / "motor_exp" / "motor_twine_init_logs"
    for file in motor_folder.glob("*.txt"):
        exp_num = int(re.findall(r"CSRT_(\d+)", file.name)[0])
        event_num = int(re.findall(r"_(\d+)_", file.name)[0])
        motor_track_dict[(exp_num, event_num)] = file

    return motor_xl, motor_track_dict