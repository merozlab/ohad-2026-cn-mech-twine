import os
import re
import numpy as np
import math as m

class Xtrm_Plant:
    def __init__(self, exp_num,mass_label):
        '''Initialize Extreme Plant object with experimental data.'''
        self.exp_num = exp_num
        self.support_mass_label = mass_label  # "light" / "stable"
        if mass_label == "light":
            self.m_sup = 0.08  # grams
            self.Lsup_cm = 25.0 # cm
            self.dsup_cm = 0.1 # cm
        elif mass_label == "stable":
            self.m_sup = np.inf  # immovable pole
            self.Lsup_cm =  np.inf # infinite pole
            self.dsup_cm = 0.8 # cm