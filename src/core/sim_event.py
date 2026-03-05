import numpy as np

class SimulationEvent:
        def __init__(self, mu, E, F_vec, T_vec, 
                     F_loc_vec_bottom, F_loc_vec_top, B, LOC, F_vec_planar, 
                     kappa0, growth_zone): # l_tip, t_contact, T_shifted):
            '''Initialize SimulationEvent with simulation parameters and data.
            0.5 mm radius'''
            self.mu = mu
            self.E = E
            self.F_vec = F_vec
            self.T_vec = T_vec
            self.F_loc_vec_bottom = F_loc_vec_bottom
            self.F_loc_vec_top = F_loc_vec_top
            self.B = B
            self.LOC = LOC # high LOC -> low l_tip
            self.F_vec_planar = F_vec_planar
            self.kappa0 = kappa0
            self.growth_zone = growth_zone
            # self.l_tip = l_tip
            # self.t_contact = t_contact
            # self.T_shifted = T_shifted # time vector shifted to contact time
            # self.s_contact = 0.5 * (F_loc_vec_bottom + F_loc_vec_top)

        
        def calc_sim_vars(self):
            '''Calculate additional simulation variables.'''
            # time when contact point is positive
            idxF = np.where(self.F_loc_vec_bottom > 0)[0]
            self.t_contact = idxF[0] if len(idxF) > 0 else np.nan # vector index when contact begins
            self.T_shifted = self.T_vec[self.t_contact:] - self.T_vec[self.t_contact] # shifted time vector after contact
            self.F_shifted = self.F_vec[self.t_contact:] # shifted force vector after contact
            self.F_shifted = self.F_shifted - self.F_shifted[0]  # zeroing force at contact time?
            self.F_planar_shifted = self.F_vec_planar[self.t_contact:]  # shifted planar force vector after contact
            self.F_planar_shifted = self.F_planar_shifted - self.F_planar_shifted[0]  # zeroing planar force at contact time?
            # time when force 1st zeros
            idx0 = np.where(self.F_shifted[1:] <= 0)[0]
            self.t_F0 = self.T_shifted[idx0[0]] if len(idx0) > 0 else np.nan 
            # Calculate tip location based on curvature and growth zone as free section will retain free curvature value
            self.l_tip = self.growth_zone - np.arccos(1 - self.kappa0 * self.LOC) / self.kappa0
            self.l_lev = self.growth_zone - self.l_tip
            # diff between F_loc_vec_bottom and F_loc_vec_top is ~0.5mm, s contact is mean of s top and s bottom
            # so the mean of the upper and lower contact points on the cylindrical organ 
            self.s_contact = 0.5 * (self.F_loc_vec_bottom + self.F_loc_vec_top)
            self.s_contact = self.s_contact[self.t_contact:]  # shifted contact location after contact
            # bending moments in mN
            self.Fs = self.s_contact * self.F_shifted*1e-3  # external moment in mN
            self.Bk = self.B * self.kappa0 * 1e-3  # internal bending moment in mN

            
