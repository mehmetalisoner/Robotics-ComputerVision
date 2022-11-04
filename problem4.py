# Example from https://pytorch.org/hub/pytorch_vision_resnet/

# WORKS and PRETTY GOOD, multiple images

import torch
import os
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import urllib
import numpy as np
import cv2
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model = model.to(device)
model.eval()


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Test with images in folder
test_folder = "dataset"
test_folder = os.path.join(os.getcwd(),test_folder)
print(test_folder)
y_pred = []
y_true = []
mean, sigma = 3, 4 # mean and standard deviation


# def add_gaussian_noise(X_imgs):
#     gaussian_noise_imgs = []
#     row, col, _ = X_imgs[0].shape
#     # Gaussian distribution parameters
#     mean = 0
#     var = 0.1
#     sigma = var ** 0.5
    
#     for X_img in X_imgs:
#         gaussian = np.random.random((row, col, 1)).astype(np.float32)
#         gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
#         gaussian_img = cv2.addWeighted(X_img, 0.75, 0.25 * gaussian, 0.25, 0)
#         gaussian_noise_imgs.append(gaussian_img)
#     gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype = np.float32)
#     return gaussian_noise_imgs


for folder in os.listdir(test_folder):
    sub_directory = (os.path.join(test_folder,folder))
    if (os.path.isdir(sub_directory) == False): continue
    print(sub_directory)
    for filename in os.listdir(sub_directory):
        label = folder
        filename_path = os.path.join(sub_directory,filename)
        print(filename_path)
        original_img = cv2.imread(filename_path)
        plt.imshow(original_img)
        plt.show()
        
        # Generate Gaussian noise
        gauss = np.random.normal(0,1,original_img.size)
        gauss = gauss.reshape(original_img.shape[0],original_img.shape[1],original_img.shape[2]).astype('uint8')
        # Add the Gaussian noise to the image
        img_gauss = cv2.add(original_img,gauss)
        # Display the image
        cv2.imshow('a',img_gauss)
        cv2.waitKey(0)

        # Actual training
        
        # input_tensor = preprocess(noisy_image)
        # input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # # move the input and model to GPU for speed if available
        # if torch.cuda.is_available():
        #     input_batch = input_batch.to('cuda')
        #     model.to('cuda')

        # with torch.no_grad():
        #     output = model(input_batch)
        
        # probabilities = torch.nn.functional.softmax(output[0], dim=0)


        # # Read the categories
        # with open("imagenet_classes.txt", "r") as f:
        #     categories = [s.strip() for s in f.readlines()]
        # # Show top categories per image
        # top5_prob, top5_catid = torch.topk(probabilities, 5)
        # for i in range(top5_prob.size(0)):
        #     print(categories[top5_catid[i]], top5_prob[i].item())
        # print("New item:")
        # y_pred.append(categories[top5_catid[0]]) # Save Prediction
        # y_true.append(label)




