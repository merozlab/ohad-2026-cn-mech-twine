import numpy as np

class Plant:
    def __init__(self, dfrow, exp_num):
        '''Initialize Plant object with experimental data.'''
        self.exp_num = exp_num
        self.dfrow = dfrow
        # identifiers
        self.genus = dfrow['Bean_Strain']
        self.camera = dfrow['Camera']

        # masses & geometry
        self.m_sup = float(dfrow['Straw_Weight(gr)'])
        self.arm_cm = float(dfrow['Exp_start_arm_length(cm)'])
        self.start_height = float(dfrow['Exp_start_height(cm)'])
        self.m20cm = float(dfrow['Weight20cm(gr)'])
        self.L0 = float(dfrow['initial_length(cm)'])

    @classmethod
    def from_dataframe(cls, dfrow, exp_num):
        '''Create Plant instance from dataframe row.'''
        return cls(dfrow, exp_num)

    def load_view_data(self):
        '''Load view-specific data from dataframe.'''
        row = self.dfrow

        self.pix2cm_s = float(row['Side_pix2cm'])
        self.pix2cm_t = float(row['Top_pix2cm'])

        from src.geometry.kinematics import compute_support_length
        self.Lsup_cm = compute_support_length(row, self.pix2cm_s)

        self.support_base_z_pos_pix = float(row['side_equil_ypos-bot_sup(pixels)'])
        val = float(row['new_y_pos_supp_bot(pixels)'])
        self.support_base_z_pos_pix_new = (
            self.support_base_z_pos_pix if np.isnan(val) else val
        )

    def load_cn_data(self):
        '''Load C.N. period data.'''
        row = self.dfrow
        self.avgT = float(row['C.N_time(minutes)'])
        self.omega0 = 2*np.pi / self.avgT

    def load_material_properties(self, E_dict):
        '''Load material properties: Young's modulus and radius sections.'''
        from src.material.caliber_radius import extract_measured_radii
        from src.material.youngs_modulus import extract_E_and_r_sections

        self.r_measured = extract_measured_radii(self.dfrow)

        (self.E_sections,self.r_sections,self.r_source
         ) = extract_E_and_r_sections(E_dict,self.exp_num,self.r_measured)

    def load_free_shape(self, skeleton_df, recalculated_df):
        """ Attach free (pre-contact) shape geometry to this Plant 
            from extracted skeleton data."""
        from src.geometry.free_shape_calcs import (
            compute_s_ds,
            compute_angle_from_coord,
            compute_curvature_from_angle
            )

        if skeleton_df.empty or recalculated_df.empty:
            print(f"[WARN] no skeleton for plant {self.exp_num}")
            self.free_shape = {
                "x_pix": np.nan,
                "y_pix": np.nan,
                "pix2cm": np.nan,
                "x_cm": np.nan,
                "y_cm": np.nan,
                "s_cm": np.nan,
                "ds_cm": np.nan,
                "curvature_cm": np.nan,
                "angle_rad": np.nan
                }
            return

        # ---- raw skeleton data (stored directly) ----
        x_skel = skeleton_df["x_reoriented(pix)"].values
        y_skel = skeleton_df["y_reoriented(pix)"].values

        # ---- geometry calculations in pixels ----
        s_pix, ds_pix = compute_s_ds(x_skel, y_skel)
        angle = compute_angle_from_coord(x_skel, y_skel)
        curvature_pix = compute_curvature_from_angle(angle, ds_pix)

        pix2cm = float(recalculated_df["pix2cm_new"].iloc[0])

        curvature_cm = curvature_pix / pix2cm
        s_cm = s_pix * pix2cm
        ds_pix_cm = ds_pix * pix2cm

        # lskel_cm = s_cm[-1]
        # yskel_cm = (np.max(y_skel) - np.min(y_skel)) * pix2cm

        # ---- attach derived state ----
        self.free_shape = {
            "x_pix": x_skel,
            "y_pix": y_skel,
            "pix2cm": pix2cm,
            "x_cm": x_skel * pix2cm,
            "y_cm": y_skel * pix2cm,
            "s_cm": s_cm,
            "ds_cm": ds_pix_cm,
            "curvature_cm": curvature_cm,
            "angle_rad": angle
        }
