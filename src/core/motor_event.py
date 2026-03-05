import numpy as np



class Motor_Event:
    """Event belongs to exactly one Plant.
    Stores geometry, force, and analysis results.
    """
    _ctx = None   # class-level
    
    @classmethod
    def bind_context(cls, ctx):
        """ Bind PlantContext for dynamic Plant resolution. """
        cls._ctx = ctx

    @property
    def plnt(self):
        """
        Resolve Plant dynamically.
        Preserves event.plnt syntax without ownership.
        """
        if self._ctx is None:
            raise RuntimeError("Event context not bound")
        return self._ctx.get_plant(self.exp_num)
    
    def __init__(self, dfrow, exp_num):
        '''Initialize Motor-Event object with experimental data.'''
        self.exp_num = exp_num
        self.event_num = int(dfrow['event_num'])
        self.dfrow = dfrow

        # Slip-twine state
        self.twine_state = dfrow['event type'] # 'slip with motor', 'slip after motor', 'twine after motor stop', 'twine with motor', 'motor reverse-decouple'
        self.event_frame = int(dfrow['frame till event']) # frame number of event onset
        self.no_twine_time = float(dfrow['no twine time(min)']) # duration of no-twine state in seconds, -1 for twining events
        self.twine_time_estimate = float(dfrow['twine_time_estimate(min)']) # estimated duration of twining state in minutes, -1 for no-twine events


    
    # def extract_tracking(self, view, motor_track_dict):
    #     from src.geometry.kinematics import (
    #         compute_support_tracking, # compute motor tracking?
    #         compute_angle,
    #     )


