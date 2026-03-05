# -*- coding: utf-8 -*-
"""
Created on Tue Jun 5 11:19:53 2025

@author: Amir
"""
#%% imports 
# imports
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


#%% define functions
# define functions
def select_roi(image):
    """
    Allows the user to manually select a region of interest (ROI) in the image using a resizable window.

    Args:
        image: Input image (NumPy array).

    Returns:
        crop_coords: Tuple (x, y, w, h) of the selected ROI.
    """
    window_name = "Select ROI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 800)
    roi = cv2.selectROI(window_name, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    return roi

def is_image_file(file_path):
    """
    Check if a given file is an image based on its extension.
    Args:
        file_path (str): The path to the file to check.
    Returns:
        bool: True if the file is an image, False otherwise.
    """
    image_extensions = {'.jpg','.JPG', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}  # Supported image extensions
    _, ext = os.path.splitext(file_path)
    return ext.lower() in image_extensions

def transparent_overlap_images(image_list, alpha=0.5):
    """
    Overlap multiple images on the same canvas with transparency.

    Args:
        image_list (list of numpy.ndarray): List of images to overlap.
        alpha (float): Transparency factor for blending each image.
        
    Returns:
        numpy.ndarray: The final overlapped image.
    """
    if not image_list:
        raise ValueError("Image list is empty.")
    
    # Use the first image as the base canvas
    canvas = image_list[0].copy()
    
    # Overlay each subsequent image on the canvas
    for i in range(1, len(image_list)):
        overlay_image = cv2.resize(image_list[i], (canvas.shape[1], canvas.shape[0]))
        canvas = blend_images(canvas, overlay_image, alpha)
    
    return canvas

def get_cropped_files(folder_path):
    """
    Get only files with '_crop' in their names from the specified folder.

    Args:
        folder_path: Path to the folder.

    Returns:
        List of cropped file names.
    """
    return [f for f in os.listdir(folder_path) if '_crop' in f and os.path.isfile(os.path.join(folder_path, f))]

def remove_background(image, case="bright", brightness_threshold=20):
    """
    Removes background based on brightness.

    Args:
        image: Input image (NumPy array).
        case: "bright" or "dark" for background type.
        brightness_threshold: Threshold value for brightness.

    Returns:
        Image with background removed.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if case == "bright":
        _, background_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)
    elif case == "dark":
        _, background_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        raise ValueError("Invalid case type. Choose 'bright' or 'dark'.")
    return cv2.bitwise_and(image, image, mask=background_mask)

def apply_hsv_filter(image, lower_bound, upper_bound):
    """
    Apply an HSV filter to an image.

    Args:
        image: Input image (NumPy array).
        lower_bound: Lower HSV bound.
        upper_bound: Upper HSV bound.

    Returns:
        Filtered image.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    return cv2.bitwise_and(image, image, mask=mask)

def blend_images(image1, image2, alpha=0.5):
    """
    Blend two images with transparency.

    Args:
        image1: The first image.
        image2: The second image.
        alpha: The blending factor (0 to 1).

    Returns:
        Blended image.
    """
    return cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)

def hard_overlap_images(image_list):
    """
    Overlap multiple images without transparency by taking the maximum pixel value.

    Args:
        image_list: List of images to overlap.

    Returns:
        Final overlapped image.
    """
    if not image_list:
        raise ValueError("Image list is empty.")

    overlapped_image = image_list[0].copy()
    for img in image_list[1:]:
        resized_img = cv2.resize(img, (overlapped_image.shape[1], overlapped_image.shape[0]))
        overlapped_image = np.maximum(overlapped_image, resized_img)

    return overlapped_image

def filter_thresh(img,hsv_lims=[[0,0,0],[255,255,255]],kernel_size=10,opening=False,show=False,reverse=False):
    ''' get img, filter with given mask, output thresholded img(binary) '''
    threshold = 137
    maxval = 255
    lower,upper = hsv_lims

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    img_hsv = cv2.bitwise_and(img, img, mask=mask)
    imgray = cv2.cvtColor(img_hsv,cv2.COLOR_BGR2GRAY) # convert to grayscale

    # threshold - > closing(regular or reverse)

    # Otsu's thresholding after Gaussian filtering
    gauss_kernel = (41,41)
    blur = cv2.GaussianBlur(imgray,gauss_kernel,0)
    Otsu_ret,Otsu_thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # closing morphology
    a = kernel_size # filter box edge size
    close_kernel = np.ones((int(a),int(a)),np.uint8) # filter shape
    closing = cv2.morphologyEx(Otsu_thresh, cv2.MORPH_CLOSE, close_kernel)
    thresh = closing
    # Global thresholding
    #ret, threshc = cv2.threshold(closing,int(threshold/2),maxval,0) # find threshold of image-> convert to binary

    if reverse:
        close_kernel = np.ones((int(a),int(a)),np.uint8) # filter shape
        closing = cv2.morphologyEx(imgray, cv2.MORPH_CLOSE, close_kernel)
        # ret, threshr = cv2.threshold(closing,threshold,maxval,cv2.THRESH_BINARY_INV) # find inverse threshold of image-> convert to binary
        thresh = closing

    if opening:
        open_kernel = np.ones((int(a),int(a)),np.uint8)
        mask_open=cv2.morphologyEx(thresh,cv2.MORPH_OPEN, open_kernel)
        # ret, thresho = cv2.threshold(mask_open,int(threshold),maxval,0)
        thresh = mask_open

    # present result
    if show:
        pos = [1900,100]
        if opening: title = 'threshold opening'
        elif reverse: title = 'threshold reverse'
        else: title = 'threshold Otsu'
        normal_window(title,thresh)
        cv2.moveWindow(title, pos[0], pos[1])

    return thresh

def hsv_filter(img):
    '''color filter with track bars, return mask created by chosen hsv'''
    # Create a window
    win_name = 'hsv filter, q to quit'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    cv2.moveWindow(win_name, 300, 30)  # Move it to (300, 30)

    # Create trackbars for color change
    # Hue is from 0-179 for OpenCV
    cv2.createTrackbar('HMin', win_name, 0, 179, lambda x: None)
    cv2.createTrackbar('SMin', win_name, 0, 255, lambda x: None)
    cv2.createTrackbar('VMin', win_name, 0, 255, lambda x: None)
    cv2.createTrackbar('HMax', win_name, 0, 179, lambda x: None)
    cv2.createTrackbar('SMax', win_name, 0, 255, lambda x: None)
    cv2.createTrackbar('VMax', win_name, 0, 255, lambda x: None)

    # Set default value for Max HSV trackbars
    cv2.setTrackbarPos('HMax', win_name, 179)
    cv2.setTrackbarPos('SMax', win_name, 255)
    cv2.setTrackbarPos('VMax', win_name, 255)

    # Initialize HSV min/max values
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

        # Display result image
        cv2.imshow(win_name, result)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            stay = False

    cv2.destroyAllWindows()
    return lower, upper

def crop_images(folder_path,crop_path=None,crop_coords=None):
    # Select ROI from the first image in the folder
    if crop_coords is None:
        if is_image_file(os.listdir(folder_path)[0]):
            crop_coords = select_roi(cv2.imread(os.path.join(folder_path, os.listdir(folder_path)[0])))  # Define the region of interest (ROI)
        else:
            crop_coords = select_roi(cv2.imread(os.path.join(folder_path, os.listdir(folder_path)[1]))) 

    # Process all images in the current view folder
    if crop_coords:
        x, y, w, h = crop_coords
        for image_file in os.listdir(folder_path):
            if not is_image_file(image_file):
                continue
            # Read the image
            image = cv2.imread(os.path.join(folder_path, image_file))
            if image is None:
                continue  # Skip if the image can't be loaded
            
            # Crop the image using the ROI
            cropped_image = image[y:y + h, x:x + w]
            
            # Determine crop output folder
            if crop_path is None:
                crop_path = os.path.join(folder_path, "cropped")
            os.makedirs(crop_path, exist_ok=True)

            # Save the cropped image to the selected path
            file_name, file_ext = os.path.splitext(image_file)
            cropped_file_path = os.path.join(crop_path, f"{file_name}_crop{file_ext}")

            # Save the cropped image
            cv2.imwrite(cropped_file_path, cropped_image)

def change_brightness_contrast(image, brightness=0, contrast=0):
    """
    Adjust the brightness and/or contrast of an image.

    Args:
        image: Input image (NumPy array).
        brightness: Brightness adjustment value (-100 to 100).
        contrast: Contrast adjustment value (-100 to 100).
    """
    beta = brightness
    alpha = 1.0 + (contrast / 100.0)
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

def compute_difference(image1, image2, threshold=50):
    """
    Compute the difference between two images and apply a threshold.

    Args:
        image1: First image (NumPy array).
        image2: Second image (NumPy array).
        threshold: Threshold value for emphasizing differences.

    Returns:
        diff_image: Thresholded difference image.
    """
    # Compute absolute difference
    diff = cv2.absdiff(image1, image2)

    # Convert to grayscale
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply threshold
    _, diff_mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)

    # Apply the mask to the difference image
    diff_image = cv2.bitwise_and(diff, diff, mask=diff_mask)

    return diff_image

def process_and_overlay(folder_path, crop_coords=None, diff_threshold=50, brightness_threshold=200):
    """
    Process all images in a folder, compute differences, and overlay changes.

    Args:
        folder_path: Path to the folder containing images.
        crop_coords: Tuple (x, y, w, h) defining the cropping region (optional).
        diff_threshold: Threshold for computing differences.
        brightness_threshold: Threshold for background modification.

    Returns:
        overlay_image: Final overlay image with differences.
    """
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No image files found in the specified folder.")
        return None

    # Read the first image as the baseline
    baseline_path = os.path.join(folder_path, image_files[0])
    baseline_image = cv2.imread(baseline_path)

    if crop_coords:
        x, y, w, h = crop_coords
        baseline_image = baseline_image[y:y+h, x:x+w]

    overlay_image = np.zeros_like(baseline_image)

    for image_file in image_files[1:]:
        image_path = os.path.join(folder_path, image_file)
        current_image = cv2.imread(image_path)

        if crop_coords:
            current_image = current_image[y:y+h, x:x+w]

        # Compute the difference with the baseline
        diff_image = compute_difference(baseline_image, current_image, threshold=diff_threshold)

        # Add the difference to the overlay image
        overlay_image = cv2.add(overlay_image, diff_image)

    # Modify the background using a brightness threshold
    gray_overlay = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2GRAY)
    _, background_mask = cv2.threshold(gray_overlay, brightness_threshold, 255, cv2.THRESH_BINARY_INV)
    overlay_image = cv2.bitwise_and(overlay_image, overlay_image, mask=background_mask)

    return overlay_image

def rotate_image(image, angle):
    """
    Rotate an image by a specified angle.

    Args:
        image: Input image (NumPy array).
        angle: Rotation angle in degrees.

    Returns:
        Rotated image.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, M, (w, h))

    return rotated_image

def rotate_image_square(image, angle, border_value=(0, 0, 0), manual_crop=False,
                        ):
    """
    Pad to square, rotate about center, then crop back to original size.
    """
    h, w = image.shape[:2]
    side = max(h, w)
    pad_y = (side - h)
    pad_x = (side - w)

    padded = cv2.copyMakeBorder(
        image,
        pad_y, side - h - pad_y,
        pad_x, side - w - pad_x,
        borderType=cv2.BORDER_CONSTANT,
        value=border_value
    )


    center = (side / 2, side / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(padded, M, (side, side), 
                             borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)

    if manual_crop:
        print("Manual cropping enabled. Please select the region to crop.")
        crop_coords = select_roi(rotated)
        x, y, w_crop, h_crop = crop_coords
        return rotated[y:y+h_crop, x:x+w_crop]
    
    # crop back to original size
    y0 = pad_y
    x0 = pad_x
    return rotated[y0:y0 + h, x0:x0 + w]

#%% main for overlay
# Define the main folder paths
main_folder = r"C:\Users\Amir\Documents\PHD\Python\GitHub\Amir_Repositories\Repo_article1\Data\image_overlay\overlay2 - 54_2_side"
raw_folder = os.path.join(main_folder, 'raw')
crop_folder = os.path.join(main_folder, 'cropped')
crop_images(raw_folder, crop_folder)


# Step 1: Select HSV filter for green stem
first_image_path = os.path.join(crop_folder, os.listdir(crop_folder)[0])
print(first_image_path)
is_image_file(first_image_path)
#%% steps
first_image = cv2.imread(first_image_path)
print("Select HSV filter for green stem...")
hsv_stem = hsv_filter(first_image)

# Step 2: Process all images in the main folder
image_files = [f for f in os.listdir(crop_folder) if is_image_file(os.path.join(crop_folder, f))]

for image_name in image_files:
    image_path = os.path.join(crop_folder, image_name)
    image = cv2.imread(image_path)
    if image is None:
        continue

    # Apply HSV filter
    filtered_image = apply_hsv_filter(image, hsv_stem[0], hsv_stem[1])

    # Save filtered images
    filtered_path = os.path.join(main_folder, 'filtered')
    os.makedirs(filtered_path, exist_ok=True)
    base_name, file_ext = os.path.splitext(image_name)
    filtered_file_path = os.path.join(filtered_path, f'{base_name}_filtered{file_ext}')
    cv2.imwrite(filtered_file_path, filtered_image)

# Step 4: Overlay filtered images
filtered_images = [
    cv2.imread(os.path.join(filtered_path, f))
    for f in os.listdir(filtered_path)
    if is_image_file(f)
]

# Step 5: Change brightness and contrast of filtered images
adjusted_images = []
for img in filtered_images:
    adjusted_img = change_brightness_contrast(img, brightness=+10, contrast=70)
    adjusted_images.append(adjusted_img)

if adjusted_images:
    overlapped_image = hard_overlap_images(adjusted_images)
    overlay_path = os.path.join(main_folder, "filtered_overlay.png")
    cv2.imwrite(overlay_path, overlapped_image)
    print(f"Overlay image saved at {overlay_path}")

    # Display the result
    plt.imshow(cv2.cvtColor(overlapped_image, cv2.COLOR_BGR2RGB))
    plt.title("Filtered Overlay")
    plt.axis("off")
    plt.show()
else:
    print("No filtered images found for overlay.")

#%% image processing
# cropped_folder = r"C:\Users\Amir Ohad\Documents\GitHub\Repo_article1\20251231_workflow\data\images\shoot_imgs\shoot_img_new\cropped"
cropped_folder = r"C:\Users\Amir\Documents\PHD\Python\GitHub\Amir_Repositories\Repo_article1\20251231_workflow\data\images\shoot_imgs\cropped"
img_path = os.path.join(cropped_folder, "54_3 - DSC_2883_crop.jpg")
img = cv2.imread(img_path)

bright_img = change_brightness_contrast(img, brightness=+50, contrast=-10)
rotated_img = rotate_image_square(bright_img, angle=40,
                                  border_value=(255, 255, 255), manual_crop=True)

plt.imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
cv2.imwrite(os.path.join(cropped_folder, "54_3 - DSC_2883_mod.jpg"), rotated_img)
#%% crop schematic
# sim_schematic_path = r"C:\Users\Amir Ohad\Documents\GitHub\Repo_article1\20251231_workflow\data\images\sim_unlabeled.png"
sim_schematic_path = r"C:\Users\Amir\Documents\PHD\Python\GitHub\Amir_Repositories\Repo_article1\20251231_workflow\data\images\sim\sim_unlabeled.png"

roi = select_roi(cv2.imread(sim_schematic_path))
cropped_sim_schematic = cv2.imread(sim_schematic_path)[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
cv2.imwrite(r"C:\Users\Amir\Documents\PHD\Python\GitHub\Amir_Repositories\Repo_article1\20251231_workflow\data\images\sim\sim_unlabeled_crop.png", cropped_sim_schematic)
#%% crop curvature
main_folder = r"C:\Users\Amir\Documents\PHD\Python\GitHub\Amir_Repositories\Repo_article1\20251231_workflow\data\images"
curvature_path = os.path.join(main_folder, "curvature", "21.JPG")
roi = select_roi(cv2.imread(curvature_path))
cropped_curvature = cv2.imread(curvature_path)[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
cv2.imwrite(os.path.join(main_folder, "curvature", "curvature_crop.png"), cropped_curvature)
#%% extract snapshot from video
# extract snapshot from video
import cv2
# video_path = r"C:\Users\Amir\Documents\PHD\Python\GitHub\Amir_Repositories\Repo_article1\20251231_workflow\figures\SM\sim_vid_E100_loc20_mu0.mp4"
video_path = r"C:\Users\Amir\Documents\PHD\Python\GitHub\Amir_Repositories\Repo_article1\20251231_workflow\data\images\motor\motor_modified_contact_wo_twine.avi"
cap = cv2.VideoCapture(video_path)
frame_number = 20  # Change this to select which frame to capture

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        snapshot_path = r"C:\Users\Amir\Documents\PHD\Python\GitHub\Amir_Repositories\Repo_article1\20251231_workflow\data\images\motor\motor_snapshot.png"
        cv2.imwrite(snapshot_path, frame)
        print(f"Snapshot saved at {snapshot_path}")
    else:
        print("Error: Could not read frame from video.")
cap.release()
#%% crop cantilever schematic
main_folder = r"C:\Users\Amir\Documents\PHD\Python\GitHub\Amir_Repositories\Repo_article1\20251231_workflow\data\images"
cantilever_path = os.path.join(main_folder, "cantilever", "cantilever_schematic v1.png")
roi = select_roi(cv2.imread(cantilever_path))
cropped_cantilever = cv2.imread(cantilever_path)[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
cv2.imwrite(os.path.join(main_folder, "cantilever", "cantilever_schematic_crop.png"), cropped_cantilever)
#%%