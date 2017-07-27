#######################IMPORTS##########################################

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from moviepy.editor import VideoFileClip
import random

####################### CAMERA CALIBRATION ##########################################

# x & y respectively - chess board sizes
nx = 9
ny = 6

# defining object point grid as [ (0, 0, 0), (1, 0, 0), (2, 0, 0) .... (7, 5, 0), (8, 5, 0)]
objp = np.zeros((ny*nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Object and image points
objpoints = []
imgpoints = []

# Read all camera cal images from directory
calibration_images = glob.glob('./camera_cal/*.jpg')

# function to calibrate & return camera matx, dist coefficients
def camera_calibrate(cal_images):

    for image in cal_images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret==True:
            objpoints.append(objp)
            imgpoints.append(corners)

            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

    # calibrate using image and object points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist

####################### UNDISTORTION ################################

# Use the camera matx and distortion coefficients to undistort image
def undist_images(img, mtx, dist):
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)
    return undist_img


# Visualize function
def visualize(img1, img2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    f.subplots_adjust(hspace=.2, wspace=.05)
    ax1.imshow(img1)
    ax1.set_title('Original Image', fontsize=10)
    ax2.imshow(img2)
    ax2.set_title('Transformed Image', fontsize=10)
    plt.show()


# Calibrate - to retrive camera matrix and distortion coefficients
ret, mtx, dist = camera_calibrate(calibration_images)

# plot sample camera images
idx = random.randint(0, len(calibration_images)-1)
sample_cal_img = cv2.imread(calibration_images[idx])

sample_cal_img_undist = undist_images(sample_cal_img, mtx, dist)

visualize(sample_cal_img, sample_cal_img_undist)

####################### PERSPECTIVE TRANSFORM ################################

# Region of interest
src = np.float32([(570, 455), (720, 455), (1100, 700), (200, 700)])

# Define function to perform perspective transformation
def perspective_transform(img):
    # Perspective transformation matx from source and destination points
    H, W = img.shape[:2]
    offset = 200
    dst = np.float32([(offset, 0), (W - offset, 0), (W - offset, H), (offset, H)])

    M = cv2.getPerspectiveTransform(src, dst)
    # Inverse to transform the unwarped image to original
    Minv = cv2.getPerspectiveTransform(dst, src)

    # warp image
    warp = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    return warp, Minv

####################### TEST IMAGE ##########################################

sample_img = cv2.imread('./test_images/test2.jpg')
sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

sample_img_undist = undist_images(sample_img, mtx, dist)
visualize(sample_img, sample_img_undist)

sample_img_persp, sample_Minv = perspective_transform(sample_img_undist)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
f.subplots_adjust(hspace=.2, wspace=.05)
ax1.imshow(sample_img)
ax1.set_title('Original Image', fontsize=10)
x = [src[0][0], src[3][0], src[2][0], src[1][0], src[0][0]]
y = [src[0][1], src[3][1], src[2][1], src[1][1], src[0][1]]
ax1.plot(x, y, color='r', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
ax1.set_ylim([sample_img_undist.shape[0], 0])
ax1.set_xlim([0, sample_img_undist.shape[1]])
ax2.imshow(sample_img_persp)
ax2.set_title('warped Image', fontsize=10)
plt.show()

####################### COLOR CHANNELS ##########################################

# RGB channels
def get_rgb(img):
    rchannel = img[:, :, 0]
    gchannel = img[:, :, 1]
    bchannel = img[:, :, 2]
    return rchannel, gchannel, bchannel

# HLS
def get_hls(img):
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hchannel = img_hls[:, :, 0]
    lchannel = img_hls[:, :, 1]
    schannel = img_hls[:, :, 2]
    return img_hls, hchannel, lchannel, schannel


sample_img_persp_r, sample_img_persp_g, sample_img_persp_b = get_rgb(sample_img_persp)
sample_img_persp_hls, sample_img_persp_h, sample_img_persp_l, sample_img_persp_s = get_hls(sample_img_persp)
# Gray scale
sample_img_persp_gray = cv2.cvtColor(sample_img_persp, cv2.COLOR_RGB2GRAY)


fig, axs = plt.subplots(3,3, figsize=(16, 12))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()
axs[0].imshow(sample_img_persp_r, cmap='gray')
axs[0].set_title('RGB R-channel', fontsize=10)
axs[1].imshow(sample_img_persp_g, cmap='gray')
axs[1].set_title('RGB G-Channel', fontsize=10)
axs[2].imshow(sample_img_persp_b, cmap='gray')
axs[2].set_title('RGB B-channel', fontsize=10)
axs[3].imshow(sample_img_persp_h, cmap='gray')
axs[3].set_title('HLS H-Channel', fontsize=10)
axs[4].imshow(sample_img_persp_l, cmap='gray')
axs[4].set_title('HLS L-channel', fontsize=10)
axs[5].imshow(sample_img_persp_s, cmap='gray')
axs[5].set_title('HLS S-Channel', fontsize=10)
axs[6].imshow(sample_img_persp, cmap='gray')
axs[6].set_title('perspective transformed', fontsize=10)
axs[7].imshow(sample_img_persp_gray, cmap='gray')
axs[7].set_title('Gray', fontsize=10)
axs[8].imshow(sample_img_persp_hls, cmap='gray')
axs[8].set_title('HLS', fontsize=10)
plt.show()


####################### Gradient Thresholding ##########################################

# All the gradient threshold function expect single color channel input img

def sobel_abs_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):

    # choose x or y axis for gradient
    if orient=='x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0,  ksize=sobel_kernel)
    if orient=='y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1,  ksize=sobel_kernel)

    # calculate absolute gradient
    abs_sobel = np.absolute(sobel)

    # scaled sobel gradient
    sobel_scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # Thresholding
    sxbinary = np.zeros_like(sobel_scaled)
    sxbinary[(sobel_scaled>=thresh[0]) & (sobel_scaled <= thresh[1])] = 1

    binary_output = sxbinary

    return binary_output


def sobel_magnitude_thresh(img, sobel_kernel=3, thresh=(0, 255)):

    # absolute x & y gradient
    abs_sobelx = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0,  ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1,  ksize=sobel_kernel))

    # sobel magniture
    sobel_mag = np.sqrt(abs_sobelx**2 + abs_sobely**2)

    sobel_scaled = np.uint8(255 * sobel_mag / np.max(sobel_mag))

    # Thresholding
    sxbinary = np.zeros_like(sobel_scaled)
    sxbinary[(sobel_scaled >= thresh[0]) & (sobel_scaled <= thresh[1])] = 1

    binary_output = sxbinary

    return binary_output


def sobel_dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Absolute direction gradient
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return dir_binary

####################### Combining sobel Thresholds ##########################################

ksize = 7 # Choose a larger odd number to smooth gradient measurements

gradx = sobel_abs_thresh(sample_img_persp_s, orient='x', sobel_kernel=ksize, thresh=(20, 255))
grady = sobel_abs_thresh(sample_img_persp_s, orient='y', sobel_kernel=ksize, thresh=(20, 255))
mag_binary = sobel_magnitude_thresh(sample_img_persp_s, sobel_kernel=ksize, thresh=(20, 255))
dir_binary = sobel_dir_threshold(sample_img_persp_s, sobel_kernel=ksize, thresh=(0.7, 1.1))

combined = np.zeros_like(gradx)
combined[((gradx == 1))] = 1

#######################Visualizing Thresholds##########################################

fig, axs = plt.subplots(2, 3, figsize=(12, 12))
fig.subplots_adjust(hspace = .2, wspace=.1)
axs = axs.ravel()
axs[0].imshow(sample_img_persp_s, cmap='gray')
axs[0].set_title('Perspective Image', fontsize=10)
axs[1].imshow(gradx, cmap='gray')
axs[1].set_title('X absolute thresholding', fontsize=10)
axs[2].imshow(grady, cmap='gray')
axs[2].set_title('Y thresholding', fontsize=10)
axs[3].imshow(mag_binary, cmap='gray')
axs[3].set_title('Magnitude thresholding', fontsize=10)
axs[4].imshow(dir_binary, cmap='gray')
axs[4].set_title('Direction thresholding', fontsize=10)
axs[5].imshow(combined, cmap='gray')
axs[5].set_title('combined', fontsize=10)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

####################### COLOR THRESHOLDING ##########################################

def color_threshold(img, thresh=(0, 255)):

    binary_output = np.zeros_like(img)
    binary_output[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return binary_output


sample_img_persp_color = color_threshold(sample_img_persp_s, thresh=(100, 255))

binary = np.zeros_like(combined)
binary[(sample_img_persp_color > 0) | (combined > 0)] = 1

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 9))
f.tight_layout()
ax1.imshow(sample_img_persp_s, cmap='gray')
ax1.set_title('Original Image', fontsize=5)
ax2.imshow(sample_img_persp_color, cmap='gray')
ax2.set_title('Thresholded S', fontsize=5)
ax3.imshow(binary, cmap='gray')
ax3.set_title('Binary ouput', fontsize=5)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

####################### SLIDING WINDOW SEARCH & CURVATURE ##########################################

def fit_lane_lines(img):

    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)

    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # number of windows
    nwindows = 10
    # Set height of windows
    window_height = np.int(img.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    rectangle_data = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))

        # cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        # cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit, right_fit = (None, None)
    print(left_fit, right_fit)

    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, left_lane_inds, right_lane_inds, rectangle_data, histogram


def plot_slidingwindow(bin_img, left_fit, right_fit, left_lane_inds, right_lane_inds):

    nonzero = bin_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    out_img = np.dstack((bin_img, bin_img, bin_img)) * 255

    ploty = np.linspace(0, bin_img.shape[0] - 1, bin_img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return out_img, ploty, left_fitx, right_fitx


####################### FIT LANES FROM PREVIOUS FRAME SLIDING WINDOW ##########################################

def fit_lane_prevframe(img, left_fit, right_fit):

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    margin = 60
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fitx, right_fitx = (None, None)
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fitx = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fitx = np.polyfit(righty, rightx, 2)


    return left_fitx, right_fitx, left_lane_inds, right_lane_inds


####################### LANE CURVATURE ##########################################

def calc_lanecurvature(img, left_fit, right_fit, left_lane_inds, right_lane_inds):

    # Define y-value where we want radius of curvature
    # maximum y-value, corresponding to the bottom of the image
    h = img.shape[0]
    ploty = np.linspace(0, h - 1, h)
    y_eval = np.max(ploty)

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    ####################### LANE CURVATURE ##########################################

    left_curverad, right_curverad, center_distance = (0, 0, 0)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    if len(leftx) != 0 and len(rightx) != 0:
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

        # Calculate radii of curvature
        left_curveradii = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curveradii = (
                         (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

    if left_fit is not None and right_fit is not None:
        car = img.shape[1] / 2

        left_x_int = left_fit[0] * h ** 2 + left_fit[1] * h + left_fit[2]
        right_x_int = right_fit[0] * h ** 2 + right_fit[1] * h + right_fit[2]
        lane_center = (left_x_int + right_x_int) / 2

        center_distance = (car - lane_center) * xm_per_pix

    return left_curveradii, right_curveradii, center_distance


####################### DRAW LINES BACK ONTO ORIGINAL IMAGE ##########################################

def draw_lane(original_img, binary_img, l_fit, r_fit, Minv, curv_rad, center_dist):

    new_img = np.copy(original_img)

    if l_fit is None or r_fit is None:
        return original_img

    warp = np.zeros_like(binary_img).astype(np.uint8)
    color = np.dstack((warp, warp, warp))

    h, w = binary_img.shape
    ploty = np.linspace(0, h - 1, num=h)  # to cover same y-range as image
    left_fitx = l_fit[0] * ploty ** 2 + l_fit[1] * ploty + l_fit[2]
    right_fitx = r_fit[0] * ploty ** 2 + r_fit[1] * ploty + r_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color, np.int_([pts]), (0, 255, 0))
    cv2.polylines(color, np.int32([pts_left]), isClosed=False, color=(255, 0, 255), thickness=15)
    cv2.polylines(color, np.int32([pts_right]), isClosed=False, color=(0, 255, 255), thickness=15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color, Minv, (w, h))

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Curve radius: ' + '{:04.2f}'.format(curv_rad) + 'm'
    cv2.putText(new_img, text, (40, 100), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

    direction = ''
    if center_dist > 0:
        direction = 'right'
    elif center_dist < 0:
        direction = 'left'
    abs_center_dist = abs(center_dist)
    text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
    cv2.putText(new_img, text, (40, 140), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    return result

####################### IMAGES TEST PIPELIINE ##########################################

def test_pipeline(original_img, mtx, dist):

    #  Kernel size
    ksize = 7
    # undistotion of the test images
    undist = undist_images(original_img, mtx, dist)
    # Warping test images
    warp, Minv = perspective_transform(undist)
    # extract color planes
    warp_hls, warp_h, warp_l, warp_s = get_hls(warp)
    warp_gray = cv2.cvtColor(warp, cv2.COLOR_RGB2GRAY)
    # apply sobel gradient
    gradx = sobel_abs_thresh(warp_s, orient='x', sobel_kernel=ksize, thresh=(30, 150))
    grady = sobel_abs_thresh(warp_s, orient='y', sobel_kernel=ksize, thresh=(30, 150))
    mag_binary = sobel_magnitude_thresh(warp_s, sobel_kernel=ksize, thresh=(30, 150))
    dir_binary = sobel_dir_threshold(warp_s, sobel_kernel=ksize, thresh=(0.5, 1.1))

    # Combine all the thresholding information
    combined = np.zeros_like(gradx)
    combined[((gradx == 1))] = 1
    # apply color thresholding on S
    s_binary = color_threshold(warp_s, thresh=(100, 255))
    # Binary output color and gradient thresholds combined
    binary_output = np.zeros_like(combined)
    binary_output[(s_binary > 0) | (combined > 0)] = 1

    return binary_output, Minv


####################### LINE CLASS ##########################################

# class to receive the characteristics of each line detection

class Line():
    def __init__(self):
        # was the line detected in the last iteration
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = []
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

    def update(self, coeffs):

        if coeffs is not None:
            if self.best_fit is not None:
                self.diffs = abs(coeffs - self.best_fit)

            if (self.diffs[0] > 0.001 or self.diffs[1] > 1.0 or self.diffs[2]>100 and len(self.current_fit)>0):
                self.detected = False
            else:
                self.detected = True
                self.current_fit.append(coeffs)

                if len(self.current_fit) >= 10:
                    self.current_fit = self.current_fit[len(self.current_fit) - 10:]
                self.best_fit = np.average(self.current_fit, axis=0)
        else:
            self.detected = False
            if len(self.current_fit) > 0:
                self.current_fit = self.current_fit[:len(self.current_fit) - 1]
            if len(self.current_fit) > 0:
                self.best_fit = np.average(self.current_fit, axis=0)


###################### TEST IMAGES USING THE PIPELINE ##########################################

test_images = glob.glob('./test_images/*.jpg')

for image in test_images:

    original_im = cv2.imread(image)
    original_im = cv2.cvtColor(original_im, cv2.COLOR_BGR2RGB)

    left_l = Line()
    right_l = Line()

    binary_output, Minv = test_pipeline(original_im, mtx, dist)

    print(left_l.detected, right_l.detected)

    if not left_l.detected or not right_l.detected:
        left_fit, right_fit, left_lane_inds, right_lane_inds, _, _ = fit_lane_lines(binary_output)
    else:
        left_fit, right_fit, left_lane_inds, right_lane_inds = fit_lane_prevframe(binary_output, left_l.best_fit, right_l.best_fit)

    if left_fit is not None and right_fit is not None:
        H = original_im.shape[0]
        left_line_xint = left_fit[0]*H**2 + left_fit[1]*H + left_fit[2]
        right_line_xint = right_fit[0]*H**2 + right_fit[1]*H + right_fit[2]

        lane_off = abs(800 - abs(left_line_xint - right_line_xint))
        if lane_off > 100:
            left_fit = None
            right_fit = None

    left_l.update(left_fit)
    right_l.update(right_fit)

    if left_l.best_fit is not None and right_l.best_fit is not None:
        left_curverad, right_curverad, center_distance = calc_lanecurvature(binary_output, left_l.best_fit, right_l.best_fit, left_lane_inds, right_lane_inds)

    # exit()

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 12))
    f.tight_layout()

    ax1.imshow(original_im)
    ax1.set_title('Original Image', fontsize=10)

    ax2.imshow(binary_output, cmap='gray')
    ax2.set_title('Thresholding Image', fontsize=10)

    lanes, y, left_l, right_l = plot_slidingwindow(binary_output, left_fit, right_fit, left_lane_inds, right_lane_inds)
    ax3.imshow(lanes, cmap='gray')
    ax3.plot(left_l, y, color='yellow')
    ax3.plot(right_l, y, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    ax3.set_title('Lane Finding', fontsize=10)


    result = draw_lane(original_im, binary_output, left_fit, right_fit, Minv, (left_curverad + right_curverad)/2, center_distance)
    ax4.imshow(result, cmap='gray')
    ax3.set_title('Lane Finding', fontsize=10)

    print('done')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


###################### TEST VIDEOS USING THE PIPELINE ##########################################

def process(img):

    original_im = img

    binary_output, Minv = test_pipeline(original_im, mtx, dist)

    print(left_lane.detected, right_lane.detected)
    if not left_lane.detected or not right_lane.detected:
        left_fit, right_fit, left_lane_inds, right_lane_inds, _, _ = fit_lane_lines(binary_output)
        print(left_fit, right_fit)
        print('first video frame..!!')
    else:
        left_fit, right_fit, left_lane_inds, right_lane_inds = fit_lane_prevframe(binary_output, left_lane.best_fit, right_lane.best_fit)
        print(left_fit, right_fit)
        print('from prev')

    if left_fit is not None and right_fit is not None:
        H = original_im.shape[0]
        left_line_xint = left_fit[0]*H**2 + left_fit[1]*H + left_fit[2]
        right_line_xint = right_fit[0]*H**2 + right_fit[1]*H + right_fit[2]
        lane_off = abs(800 - abs(left_line_xint - right_line_xint))
        if lane_off > 50:
            left_fit = None
            right_fit = None

    left_lane.update(left_fit)
    right_lane.update(right_fit)

    print(left_lane.best_fit, right_lane.best_fit)

    if left_lane.best_fit is not None and right_lane.best_fit is not None:
        left_curverad, right_curverad, center_distance = calc_lanecurvature(binary_output, left_lane.best_fit, right_lane.best_fit, left_lane_inds, right_lane_inds)
        image_ouput = draw_lane(original_im, binary_output, left_lane.best_fit, right_lane.best_fit, Minv, (left_curverad + right_curverad) / 2,
                       center_distance)
    else:
        image_ouput = original_im

    return image_ouput


####################### TEST VIDEO USING THE PIPELINE ##########################################

left_lane = Line()
right_lane = Line()

video_output = 'project_video_output_final.mp4'
video_input = VideoFileClip('project_video.mp4')
processed_video = video_input.fl_image(process)
processed_video.write_videofile(video_output, audio=False)

