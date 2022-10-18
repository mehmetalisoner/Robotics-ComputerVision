import numpy as np
import cv2
import glob
import os

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


found = 0   # count the picture read
folder = "test_folder"
size = 0
images = []
for filename in os.listdir(folder):
    path = os.path.join(folder,filename)                    # create path for image
    img = cv2.imread(path)                                  # read image
    
    # Downscale image because photos shot on iPhone are too big
    p = 0.25                                                # select downscaling factor                         
    new_width = int(img.shape[1] * p)                       # calculate new width
    new_height = int(img.shape[0] * p)                      # calculate new height
    img = cv2.resize(img, (new_width, new_height))          # resize
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            # convert to gray image
    
    cv2.imshow('img',gray)
    cv2.waitKey(500)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

        
print("Number of images used for calibration: ", found)