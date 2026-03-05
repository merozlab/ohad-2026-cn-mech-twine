# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:19:53 2026

"""
import math as m
import numpy as np
import math as m
import cv2
import networkx
from scipy.spatial import cKDTree

def nothing(x):
    pass    

def waitandexit():
    k = cv2.waitKey(0) & 0xff # control presentation delay (in ms)
    if k == 27 or k==ord("q"): # Esc key
        print("\nCaught ESC")
        cv2.destroyAllWindows()
    return

def normal_window(win_name,img):
    cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
    cv2.imshow(win_name,img)
    cv2.moveWindow(win_name, 0,0)

def hsv_filter(img):
    '''color filter with track bars, return mask created by choosen hsv'''
    # Create a window
    win_name = 'hsv filter, q to quit'
    normal_window(win_name,img)
    # cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
    # cv2.imshow(win_name,img)
    cv2.moveWindow(win_name, 300,30)  # Move it to (500,30)

    # Create trackbars for color change
    # Hue is from 0-179 for Opencv
    cv2.createTrackbar('HMin', win_name, 0, 179, nothing)
    cv2.createTrackbar('SMin', win_name, 0, 255, nothing)
    cv2.createTrackbar('VMin', win_name, 0, 255, nothing)
    cv2.createTrackbar('HMax', win_name, 0, 179, nothing)
    cv2.createTrackbar('SMax', win_name, 0, 255, nothing)
    cv2.createTrackbar('VMax', win_name, 0, 255, nothing)

    # Set default value for Max HSV trackbars
    cv2.setTrackbarPos('HMax', win_name, 179)
    cv2.setTrackbarPos('SMax', win_name, 255)
    cv2.setTrackbarPos('VMax', win_name, 255)

    # Initialize HSV min/max values
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0
    stay = True
    while stay:
        # Get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin', win_name)
        sMin = cv2.getTrackbarPos('SMin', win_name)
        vMin = cv2.getTrackbarPos('VMin', win_name)
        hMax = cv2.getTrackbarPos('HMax', win_name)
        sMax = cv2.getTrackbarPos('SMax', win_name)
        vMax = cv2.getTrackbarPos('VMax', win_name)

        # Set minimum and maximum HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Convert to HSV format and color threshold
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(img, img, mask=mask)

        # Print if there is a change in HSV value
        if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | 
           (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, \
                  vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display result image
        cv2.imshow(win_name, result)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            stay = False
    cv2.destroyAllWindows()
    # (hMin = 0 , sMin = 0, vMin = 0), (hMax = 179 , sMax = 21, vMax = 255)
    # (hMin = 0 , sMin = 0, vMin = 0), (hMax = 179 , sMax = 23, vMax = 255) 

    return lower,upper

def crop_and_filter(image, roi, lower, upper,low_thresh=120,upper_thresh=255):
    '''crop image to roi and filter by hsv, return white pixels count'''
    # crop
    imCrop = image[int(roi[1]):int(roi[1]+roi[3]), 
                   int(roi[0]):int(roi[0]+roi[2])]
    # convert to hsv
    hsv = cv2.cvtColor(imCrop, cv2.COLOR_BGR2HSV)
    # apply hsv filter
    mask = cv2.inRange(hsv, lower, upper)
    # Keep only relatively bright parts of the masked image
    # Convert cropped image to grayscale
    gray = cv2.cvtColor(imCrop, cv2.COLOR_BGR2GRAY)
    # Apply mask to grayscale image
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    # Threshold to keep only bright pixels (e.g., above 180)
    _, binary = cv2.threshold(masked_gray, low_thresh, upper_thresh,
                               cv2.THRESH_BINARY)
    # count white pixels in binary
    # white_pixels = np.sum(binary == 255)
    return binary

def angle_2lines(pa1,pa2,pb1,pb2):
    '''calculate angle between 2 lines defined by 2 points each'''
    # calculate angle between 2 lines
    ma = (pa2[1]-pa1[1])/(pa2[0]-pa1[0])
    mb = (pb2[1]-pb1[1])/(pb2[0]-pb1[0])
    angle = m.atan(abs((mb-ma)/(1+ma*mb)))
    return angle

def draw_line(event, x, y, flags, param):
    state = param

    if event == cv2.EVENT_LBUTTONDOWN:
        state["drawing"] = True
        state["x0"], state["y0"] = x, y
        state["x1"], state["y1"] = x, y

    elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
        state["x1"], state["y1"] = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        state["drawing"] = False
        state["lines"].append((state["x0"], state["y0"], x, y))

def live_line(filename, window_name, xy, xpic=500, ypix=0):

    if isinstance(filename, str):
        o_img = cv2.imread(filename)
    else:
        o_img = filename.copy()

    window_name += ', d to del line, q to exit'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, xpic, ypix)

    # shared mutable state
    state = {
        "drawing": False,
        "x0": 0, "y0": 0,
        "x1": 0, "y1": 0,
        "lines": xy
    }

    cv2.setMouseCallback(window_name, draw_line, state)
    linewidth = 2
    while True:
        img_display = o_img.copy()

        # draw committed lines
        for line in xy:
            cv2.line(img_display, (line[0], line[1]), (line[2], line[3]), (0,255,0), linewidth)

        # draw live preview line
        if state["drawing"]:
            cv2.line(
                img_display,
                (state["x0"], state["y0"]),
                (state["x1"], state["y1"]),
                (0, 0, 255), linewidth
            )

        cv2.imshow(window_name, img_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d") and xy:
            xy.pop()

    cv2.destroyAllWindows()
    return len(xy)


def distance_2d(p1p2): #p1p2 = [x0,y0,x,y]
    (x1, y1,x2,y2) = p1p2
    return round(np.sqrt((x1-x2)**2 + (y1-y2)**2))

# sum multiple lines in photo to one length
def multiline_dist(file,title='distance measure',all_dist=False):
    '''press q to finish, d to delete last line, returns sum of lines'''
    line = []
    live_line(file,title,line)
    sumlen = 0
    for l in line: # sum all lines drawn
        sumlen += distance_2d(l) # sum over all lines
    if all_dist == False: return sumlen
    else: return [distance_2d(l) for l in line]

def zoom(filename=None,img=None,roi=[], zoom_target='zoom'):
    '''get filename for img to zoom in on, zoom target name optional.
        returns zoomed in image and ROI: (x,y,w,h)'''
    if filename: img = cv2.imread(filename)
    else: 
        if img is None:
            raise ValueError("Either filename or img must be provided.")
        
    if not roi:
        cv2.namedWindow(zoom_target,cv2.WINDOW_NORMAL)
        # cv2.moveWindow(zoom_target, 500, 200)
        roi = cv2.selectROI(zoom_target,img)
        cv2.destroyWindow(zoom_target)
    else: pass
    # ROI: (x,y,w,h)
    cropped_img = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    return cropped_img, roi

def kernel_size2D(ker_size1, ker_size2=None):
    if ker_size2: pass
    else:
        ker_size2 = ker_size1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ker_size1, ker_size2))

def order_skeleton_points(skeleton_coords):
    coords = np.array(skeleton_coords).astype(int)
    tree = cKDTree(coords)
    G = networkx.Graph()

    for i, pt in enumerate(coords):
        dists, idxs = tree.query(pt, k=7, distance_upper_bound=1.5)
        for j in idxs:
            if j != i and j < len(coords):
                G.add_edge(i, j)

    # Find endpoints (nodes with degree 1)
    endpoints = [n for n in G.nodes if G.degree(n) == 1]
    if len(endpoints) < 2:
        raise ValueError("Could not find enough endpoints for a full path.")

    # Compute the longest shortest path between all endpoint pairs
    longest_path = []
    max_len = 0
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            try:
                path = networkx.shortest_path(G, endpoints[i], endpoints[j])
                if len(path) > max_len:
                    max_len = len(path)
                    longest_path = path
            except networkx.NetworkXNoPath:
                continue

    return coords[longest_path]

