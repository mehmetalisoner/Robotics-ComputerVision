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




# Test with images in folder
test_folder = "dataset"
noisy_folder = "noisy_pics"
project_folder = os.getcwd()
test_folder = os.path.join(project_folder,test_folder)
noisy_folder = os.path.join(project_folder,noisy_folder)




for folder in os.listdir(test_folder):
    sub_directory = (os.path.join(test_folder,folder))
    if (os.path.isdir(sub_directory) == False): continue
    print(sub_directory)
    for filename in os.listdir(sub_directory):
        label = folder
        filename_path = os.path.join(sub_directory,filename)
        original_img = cv2.imread(filename_path)
        cv2.imshow('Original', original_img) 
        cv2.waitKey(3000)
        # Generate Gaussian noise
        gauss = np.random.normal(0,1,original_img.size)
        gauss = gauss.reshape(original_img.shape[0],original_img.shape[1],original_img.shape[2]).astype('uint8')
        # Add the Gaussian noise to the image
        img_gauss = cv2.add(original_img,gauss)
        # Display the image
        cv2.imshow('Noisy',img_gauss)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        # os.chdir(noisy_folder)
        # cv2.imwrite("noisy_"+filename,img_gauss)


# # os.chdir(project_folder)
# for image in os.listdir(noisy_folder):
#     image_path = os.path.join(noisy_folder,image)
#     img = cv2.imread(image_path)
#     print(image_path)
#     cv2.imshow('a',img)
#     cv2.waitKey(3)