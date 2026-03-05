import numpy as np
import math as m
# recalculate CN?
class Motor_Plant:
    def __init__(self, dfrow, exp_num):
        '''Initialize Motor-Plant object with experimental data.'''
        self.exp_num = exp_num
        self.dfrow = dfrow

        # physical parameters
        self.start_height = float(dfrow['initial_height(cm)'])
        self.arm_length = np.nan

        # rotation values
        self.motor_base_rpm = float(dfrow['motor_base_vel(rpm)'])
        self.motor_pwm = float(dfrow['motor_pwm']) # can be negative for CW rotation
        self.stage_rot_rph = float(dfrow['stage_rotation(rph)'])
        self.CN_all = np.array(dfrow['CN_periods(min)'])
        self.avgT = float(dfrow['CN_avg(min)']) # average CN rate in min, always positive - CCW rotation
        self.avgfreq_rph = 60 / self.avgT # convert to rph
        self.avgfreq_rph2 = float(dfrow['CN_avg_rate(rph)']) # from df, should match avgfreq_rph
        self.eff_rot_rph = self.stage_rot_rph + self.avgfreq_rph # eff rot. rate, sum stage and CN rates
        self.eff_rot_rph2 = float(dfrow['effective_CN_rate(rph)']) # from df
        self.sec_diff = float(dfrow['sec_diff(sec)']) # track frame time step in seconds

    @classmethod
    def from_dataframe(cls, dfrow, exp_num):
        '''Create Plant instance from dataframe row.'''
        return cls(dfrow, exp_num)

    # def load_view_data(self):
    #     '''Load view-specific data from dataframe.'''
    #     self.pix2cm_s = np.nan
    #     self.pix2cm_t = np.nan

    # def CN_analysis(self,CNxl): # slice_CN.py


