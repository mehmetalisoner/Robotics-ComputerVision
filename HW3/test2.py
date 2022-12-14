import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
a = 7
b = 7
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((a*b,3), np.float32)
objp[:,:2] = np.mgrid[0:a,0:b].T.reshape(-1,2)
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
    
    #cv2.imshow('img',gray)
    #cv2.waitKey(300)
    #cv2.waitKey(3)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (a,b), None)
    #print(ret,corners)

    # If found, add object points, image points (after refining them)
    if ret == True:
        found += 1 
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (a,b), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
print("Number of images used for calibration: ", found)

print(len(tvecs))



fig = plt.figure(figsize=(4,4))

ax = fig.add_subplot(111, projection='3d')


for arrs in tvecs:
    x_coord = arrs[0]
    y_coord = arrs[1]
    z_coord = arrs[2]
    ax.scatter(x_coord,y_coord,z_coord)



ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()