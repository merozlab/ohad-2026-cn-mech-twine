import numpy as np

class Xtrm_Event:
    """
    Event belongs to exactly one Plant.
    Stores minimal event data for Extreme Plants.
    """

    def __init__(self, Xtrm_Plant, event_num):
        """ Initialize Extreme Event with reference to parent Extreme Plant. """
        self.xplnt = Xtrm_Plant
        self.exp_num = Xtrm_Plant.exp_num
        self.event_num = event_num

    def get_params(self,dfrow):
        """ Get data from DataFrame row. """
        self.cont2stemtip_dist_cm = dfrow['cont2stemtip_dist_cm']
        self.cont2suptip_dist_cm = dfrow['cont2suptip_dist_cm']
        self.twine_state = dfrow['twine_status']

    def compute_resistance(self):
        from src.physics.forces import calc_effective_resistance
        self.R_eff_mN = calc_effective_resistance(self)