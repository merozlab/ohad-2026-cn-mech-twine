import numpy as np

class Event:
    """
    Event belongs to exactly one Plant.
    Stores geometry, force, and analysis results.
    """
    _ctx = None   # class-level, not instance-level

    def __init__(self, exp_num, event_num):
        """ Initialize Event with reference to parent Plant. """
        self.exp_num = exp_num
        self.event_num = event_num
        self.views_seen = set() # track loaded views

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

    def load_view_metadata(self, df, i, view):
        """ Load event metadata from DataFrame row. """
        if view == "side":
            self.frm0_side = int(df.at[i, "First_contact_frame"])
            self.frm_dec_side = int(df.at[i, "Slip/Twine_frame"])
            self.event_label = df.at[i, "Event Label"]

        elif view == "top":
            self.frm0_top = int(df.at[i, "First_contact_frame"])
            self.frm_dec_top = int(df.at[i, "Slip/Twine_frame"])
            self.ltip = (
                self.plnt.pix2cm_t *
                float(df.at[i, "Contact_distance_from_stem_tip(pixels)"])
            )
            self.twine_state = float(df.at[i, "Twine_status"])

        self.views_seen.add(view)

    def extract_tracking(self, view, track_dict, contact_dict):
        from src.geometry.kinematics import (
            compute_support_tracking,
            compute_contact_tracking,
            compute_pend_angle,
        )

        if view == "side":
            compute_support_tracking(self,view, track_dict)

        elif view == "top":
            compute_support_tracking(self, view, track_dict)
            compute_contact_tracking(self, contact_dict)
            compute_pend_angle(self)

    
    def compute_force(self):
        from src.physics.forces import F_of_t
        self.F_bean = F_of_t(self)

    def save_Lbase(self):
        from src.geometry.free_shape_calcs import compute_Lbase
        try:
            lskel_cm = self.plnt.free_shape["s_cm"][-1]
            dy_skel_cm = (np.max(self.plnt.free_shape["y_cm"]) - np.min(self.plnt.free_shape["y_cm"]))

            self.Lbase = compute_Lbase(
                            lskel=lskel_cm,
                            height=self.plnt.start_height,
                            yskel=dy_skel_cm,
                            ltip=self.ltip,
                            )
        except Exception as e:
            self.Lbase = self.plnt.L0
            
