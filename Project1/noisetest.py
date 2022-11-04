import cv2
import numpy as np
import os


image = "balloon1.jpg"

image = os.path.join(os.getcwd(),image)
print(image)

img = cv2.imread(image)
# Generate Gaussian noise
gauss = np.random.normal(0,1,img.size)
gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
# Add the Gaussian noise to the image
img_gauss = cv2.add(img,gauss)
# Display the image
cv2.imshow('a',img_gauss)
cv2.waitKey(0)