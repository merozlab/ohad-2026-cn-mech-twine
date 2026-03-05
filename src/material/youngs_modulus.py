import re
import numpy as np

def extract_E_and_r_sections(E_dict, exp_num, r_measured):
    """
    Extract Young's modulus and radius per section.

    Priority:
    1. r from E_dict section line
    2. fallback to measured r if E_dict r is missing / invalid

    Parameters
    ----------
    E_dict : dict
        Young's modulus dictionary
        structure of E_dict entries:
        section_E[section,R][mass] = [E_asym/100,popt[0]/100,fit[0]]

        E_asym: direct calculation of E from max deflection via
        E_asym = (np.sqrt(3)*P*b*L*(L**2-(b*L)**2)**(3/2))/(27*L*I*max(y)) - lengths in cm
        popt[0] is the fitted value for E for the 3-point bending curve
        fit[0] is the fitted value for the above curve-fit
        Young's modulus in MPa

    exp_num : int
        Experiment number
    r_measured : dict
        {position_cm: r_cm} from physical measurements

    Returns
    -------
    E_sections : dict
        {sect_mid_cm: E_avg}
    r_sections : dict
        {sect_mid_cm: r_cm}
    r_source : dict
        {sect_mid_cm: "E_dict" or "measured"}
    """

    E_sections = {0.0: 0.0}   # your convention
    r_sections = {}
    r_source = {}

    if exp_num not in E_dict:
        return E_sections, r_sections, r_source

    labels = E_dict[exp_num][0]
    values = E_dict[exp_num][1]

    current_sect = None
    current_r = None

    for label, val in zip(labels, values):

        # skip NaNs
        if val != val:
            continue

        # section definition line → radius from bending analysis
        m = re.search(r"(\d{1,2})-", str(label))
        if m:
            start_cm = float(m.group(1))
            current_sect = start_cm + 2.5   # midpoint of 5 cm section
            current_r = float(val)
            continue

        # avg line → Young's modulus
        if str(label).strip().lower() == "avg":
            if current_sect is None:
                continue

            # save E
            E_sections[current_sect] = float(val)

            # decide radius source
            if current_r is not None and np.isfinite(current_r) and current_r > 0:
                r_sections[current_sect] = current_r
                r_source[current_sect] = "E_dict"
            else:
                # fallback to measured radius
                # find closest measured location
                closest_pos = min(
                    r_measured.keys(),
                    key=lambda x: abs(x - current_sect)
                )
                r_sections[current_sect] = r_measured[closest_pos]
                r_source[current_sect] = "measured"

    return E_sections, r_sections, r_source
