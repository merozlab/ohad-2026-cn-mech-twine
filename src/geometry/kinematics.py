import numpy as np
from src.io.read_tracking_file import funcget_tracked_data, get_tracked_data
from src.utils.array_tools import adjust_length

def compute_support_length(row, pix2cm_s):
    return (
        row['Dist_straw_from_hinge(pixels)'] * pix2cm_s
        + row['Straw_Length(cm)'])

def compute_support_tracking(ev, view, track_dict):
    """ get track point of support"""
    if view=='side':
        # x,z coordinates of side view tip tracking
        ev.x_track_side0,ev.z_track_side0,time_s = funcget_tracked_data(
            track_dict[(ev.exp_num,ev.event_num,view)],[0,-1],ev.plnt.camera)
        
        # x0,z0 - side view equilibrium coordinates of tracked point
        ev.x0_side,ev.z0_side = ev.x_track_side0[0],ev.z_track_side0[0]

        # transform x,z to coordinates relative to x0,z0
        ev.x_track_side,ev.z_track_side = \
            -np.subtract(ev.x_track_side0,ev.x0_side),\
            np.subtract(ev.z_track_side0,ev.z0_side)
        
        # convert to cm
        ev.x_track_side_cm = np.multiply(ev.x_track_side,ev.plnt.pix2cm_s)
        ev.z_track_side_cm = np.multiply(ev.z_track_side,ev.plnt.pix2cm_s)

        # save within decision timeframe
        ev.x_track_side_dec = ev.x_track_side_cm[ev.frm0_side:ev.frm_dec_side]
        ev.z_track_side_dec = ev.z_track_side_cm[ev.frm0_side:ev.frm_dec_side]

        # get z dist. of tracked spot at equil. to bottom of support + convert to cm
        ev.L_track2suptip = abs(ev.z0_side-ev.plnt.support_base_z_pos_pix)
        ev.L_track2suptip_new = abs(ev.z0_side-ev.plnt.support_base_z_pos_pix_new) # updated track2tip pixels
        ev.L_track2suptip_cm = ev.L_track2suptip*ev.plnt.pix2cm_s
        ev.L_track2suptip_cm_new = ev.L_track2suptip_new*ev.plnt.pix2cm_s # updated track2tip in cm
        ev.h_tip = ev.L_track2suptip_cm
        ev.h_tip_new = ev.L_track2suptip_cm_new # updated 
        ev.L_tracked = ev.plnt.Lsup_cm - ev.h_tip # length of support from hinge to tracked point in cm
        ev.L_tracked_new = ev.plnt.Lsup_cm - ev.h_tip_new # updated length of support from hinge to tracked point in cm

    elif view=='top':
        ev.x_track_top_pix,ev.y_track_top_pix,ev.top_timer = funcget_tracked_data(
            track_dict[(ev.exp_num,ev.event_num,view)],[0,-1],ev.plnt.camera)
        
        # x0,y0 - top view equilibrium coordinates of tracked point
        ev.x0_top,ev.y0_top = ev.x_track_top_pix[0],ev.y_track_top_pix[0]

        # transform x,y to coordinates relative to x0,y0 in top view
        ev.x_track_top,ev.y_track_top = \
            np.subtract(ev.x_track_top_pix,ev.x0_top),np.subtract(ev.y_track_top_pix,ev.y0_top)
        
        # convert to cm
        ev.x_track_top_cm = np.multiply(ev.x_track_top,ev.plnt.pix2cm_t)
        ev.y_track_top_cm = np.multiply(ev.y_track_top,ev.plnt.pix2cm_t)

        # set length of events to dec period only
        ev.dec_x_track_top = ev.x_track_top_cm[ev.frm0_top:ev.frm_dec_top]
        ev.dec_y_track_top = ev.y_track_top_cm[ev.frm0_top:ev.frm_dec_top]
        ev.dec_z_track_side = ev.z_track_side_cm[ev.frm0_side:ev.frm_dec_side]

        # timer for decision period
        ev.timer = np.subtract(ev.top_timer[ev.frm0_top:
                    ev.frm_dec_top],ev.top_timer[ev.frm0_top])
        
        # collect track xyz (in cm) - distance from eq point of tracked point in cm
        ev.xyz = np.zeros((3,len(ev.dec_x_track_top))) # all xyz of sup track
        ev.xyz[0,::] = ev.dec_x_track_top
        ev.xyz[1,::] = ev.dec_y_track_top
        ev.dec_z_track_side,ev.dec_y_track_top = \
            adjust_length(ev.dec_z_track_side,ev.dec_y_track_top,
                choose=ev.dec_y_track_top)
        ev.xyz[2,::] = ev.dec_z_track_side

        # ev.xyz = np.array(ev.xyz_supportrack) # in cm ?
        ev.xyz0 = np.array([[ev.xyz[0][0],ev.xyz[1][0],
                ev.xyz[2][0]]]*len(ev.xyz[0])).T # initial xyz trk point

def compute_contact_tracking(ev, contact_dict):
    """ get track point of contact along support"""
    
    # get contact data from side view
    ev.x_cont,ev.z_cont,ev.contact_timer = funcget_tracked_data(
        contact_dict[(ev.exp_num,ev.event_num)],[0,-1], ev.plnt.camera)
    # transform contact x,z to coordinates relative to x0,z0
    # z_cont should always be larger than z0
    # compare x contact to the initial x position of the tracked contact point - not the support tip
    # take minus the x since the direction is opposite to the top view
    ev.x_cont,ev.z_cont = \
        -np.subtract(ev.x_cont,ev.x_cont[0]),abs(np.subtract(ev.z_cont,ev.z0_side))
    # convert to cm
    ev.x_cont_cm = np.multiply(ev.x_cont,ev.plnt.pix2cm_s)
    ev.z_cont_cm = np.multiply(ev.z_cont,ev.plnt.pix2cm_s)

    # save to dict only coor within decision timeframe
    # ev.xz_contact = np.array([ev.x_cont_cm[ev.frm0_side:ev.frm_dec_side],
    #                 ev.z_cont_cm[ev.frm0_side:ev.frm_dec_side]])
    ev.x_cont_dec = ev.x_cont_cm[ev.frm0_side:ev.frm_dec_side]          
    ev.z_cont_dec = ev.z_cont_cm[ev.frm0_side:ev.frm_dec_side]

    # adjust coordantes lengths
    ev.z_cont_dec,ev.xyz[0] = adjust_length(ev.z_cont_dec,ev.xyz[0],choose=ev.xyz[0])

    # get support vector
    # set equilibrium point as origin (0,0,0), and support hinge as (0,0,L_tracked) in cm
    # ev.hinge = np.array([0,0,ev.L_tracked])
    ev.hinge_xyz = np.array([0,0,ev.L_tracked_new]) # zsup updated, hinge_updt_zsup

    # p*(x,y,z)track is then the parametrized vector describing the support
    # transpose to allow subtraction of hinge from each point, then transpose back
    # ev.dxyz = np.subtract(ev.xyz.T,ev.hinge).T # vector from hinge to track point
    ev.dxyz = np.subtract(ev.xyz.T,ev.hinge_xyz).T # zsup updated,dxyz_updt_zsup

    # ev.dz_cont = np.subtract(ev.z_cont_dec,ev.hinge[2]) # z distance from hinge to contact point
    ev.dz_contact = np.subtract(ev.z_cont_dec,ev.hinge_xyz[2]) # zsup updated, dz_contact_updt_zsup

    # we extract p for the contact point from the relation
    # p = xc/xtr = yc/ytr = zc/ztr. (but we dont know y from this description)
    # ev.px = np.divide(ev.x_cont_dec,ev.dxyz[0][:]) # denominator will be close to zero...
    # ev.px_side = np.divide(ev.x_cont_dec,ev.x_track_side_dec) # using side view coordiantes only

    # ev.pz = np.divide(ev.dz_contact,ev.dxyz[2][:]) # denominator wont be zero, since dz is ~ L_tracked
    ev.pz = np.divide(ev.dz_contact,ev.dxyz[2][:]) # zsup updated,pz_updt_zsup

    # get py from the less volitile p (being pz):
    # ev.yc = np.multiply(ev.pz,ev.dxyz[1][:])
    ev.yc = np.multiply(ev.pz,ev.dxyz[1][:]) # zsup updated,yc_updt_zsup

    # then we get the contact position (x,y,z)c = p*(xtr,ytr,ztr)
    # ev.xyz_contact = np.array(ev.xz_contact[0],ev.yc,ev.xz_contact[1])
    # ev.xyz_contact = np.multiply(ev.pz,ev.dxyz)
    ev.xyz_contact = np.multiply(ev.pz,ev.dxyz) # zsup updated,xyz_contact_updt_zsup

    # so the contact length lc = sqrt(xc^2+yc^2+zc^2)
    # ev.l_c_vec = np.sqrt(np.sum(ev.xyz_contact**2,axis=0))
    ev.l_contact = np.sqrt(np.sum(ev.xyz_contact**2,axis=0)) # zsup updated, l_c_vec_updt_zsup

def compute_pend_angle(ev):
    """ compute angle of support from tracked data"""

    # get r_tr from sum of x and y components squared
    ev.r_tr_xy_raw = np.sqrt(ev.xyz[0]**2 + ev.xyz[1]**2)
    # ev.r_tr_xy = ev.xyz[0]**2 + ev.xyz[1]**2 # check without sqrt

    ev.r_tr_xy = ev.r_tr_xy_raw - ev.r_tr_xy_raw[0] # relative to start point
    # calculate alpha via asin(r_tr/((L-h_tip)))

    # ev.alpha3 = np.arcsin(np.divide(ev.r_tr_xy,ev.L_tracked))
    ev.alpha = np.arcsin(np.divide(ev.r_tr_xy,ev.L_tracked_new)) # zsup updated,alpha3_updt_zsup


def compute_angle_near_contact(ev, track_file):
    """compute support angle near contact using two tracked side-view points."""
    x_track, z_track, _, indx = get_tracked_data(filename=track_file, obj=[0, 1])

    x_track = np.asarray(x_track)
    z_track = np.asarray(z_track)
    indx = np.asarray(indx)

    pre = indx == 0
    post = indx == 1

    x_pre, z_pre = x_track[pre], z_track[pre]
    x_post, z_post = x_track[post], z_track[post]

    # keep equal lengths if needed
    x_pre, x_post = adjust_length(x_pre, x_post, choose=x_pre)
    z_pre, z_post = adjust_length(z_pre, z_post, choose=z_pre)

    angle = np.arctan2(z_post - z_pre, x_post - x_pre)
    return angle