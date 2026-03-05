import numpy as np
import math as m
from src.core.xtrm_event import Xtrm_Event
from src.core.exp_event import Event

# Calculate force using torque equilibrium in mN
def calc_F_equib(l_c,l_sup_cm,alpha_t,m_sup):
    '''calc using distance of contact from support hinge - l_c'''
    gcgs=980
    dyne2mN = 1/100
    F_mN = gcgs * m_sup * l_sup_cm * (m.tan(alpha_t)/(2*l_c)) * dyne2mN
    return abs(F_mN)

# Calculate force for time series
def F_of_t(event):
    l_c = event.l_contact
    l_sup_cm = event.plnt.Lsup_cm
    m_sup = event.plnt.m_sup
    alpha = event.alpha

    return np.array([calc_F_equib(l_c[i], l_sup_cm, alpha[i], m_sup) for i in range(len(alpha))])

def calc_effective_resistance(event):
    """
    Calculate effective resistance of support based on its properties
    and contact location.
    Returns resistance in mN.
    """
    # Unpack necessary parameters from event and plant

    if isinstance(event, Xtrm_Event):
        plant = event.xplnt
        m_sup = plant.m_sup  # mass of support in grams
        L_sup = plant.Lsup_cm  # length of support in cm
        cont2suptip_dist = event.cont2suptip_dist_cm  # contact to support tip distance in cm
        # Calculate length of the support from contact point to hinge (at time of contact)
        L_c = L_sup - cont2suptip_dist  # cm

    elif isinstance(event, Event):
        plant = event.plnt
        m_sup = plant.m_sup  # mass of support in grams
        L_sup = plant.Lsup_cm  # length of support in cm
        L_c = event.l_contact[0]  # contact distnace from hinge at t=0
    
    if m_sup == np.inf or L_c <= 0:
        # Immovable support or contact at/beyond base
        R_eff = np.inf

    else:
        # Calculate effective resistance torque balance
        R_eff = 0.01 * (m_sup * 981 * L_sup) / (2 * L_c) # 0.01*: 10 uN-> mN

    return R_eff