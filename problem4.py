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

# Debug variable
DEBUG=0

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

# Folder names
test_folder = "dataset"
noisy_folder = "noisy_pics"
project_folder = os.getcwd()
test_folder = os.path.join(project_folder,test_folder)
noisy_folder = os.path.join(project_folder,noisy_folder)

# Arrays for predictions and actual values
y_pred = []
y_true = []




# Add noise to dataset and perform evaluation in model
for folder in os.listdir(test_folder):
    sub_directory = (os.path.join(test_folder,folder))
    if (os.path.isdir(sub_directory) == False): continue
    print(sub_directory)
    for filename in os.listdir(sub_directory):
        label = folder
        filename_path = os.path.join(sub_directory,filename)
        original_img = cv2.imread(filename_path)
        if(DEBUG): # Show original image if debug
            cv2.imshow('original',original_img)
            cv2.waitKey(2)
        
        # Generate Gaussian noise
        gauss = np.random.normal(0,1,original_img.size)
        gauss = gauss.reshape(original_img.shape[0],original_img.shape[1],original_img.shape[2]).astype('uint8')
        # Add the Gaussian noise to the image
        img_gauss = cv2.add(original_img,gauss)
        if (DEBUG): # Show noisy image if debug
            cv2.imshow('a',img_gauss)
            cv2.waitKey(2)
        noisy_image_name = "noisy_"+ filename
        noisy_image_path = os.path.join(sub_directory,noisy_image_name)
        cv2.imwrite(noisy_image_name,img_gauss)


        # Evaluation
        
        # noisy_image = Image.open(noisy_image_path)
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


cv2.destroyAllWindows()


# # Build confusion matrix
# classes = ('cats','dogs')
# cf_matrix = confusion_matrix(y_true, y_pred)
# df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *2, index = [i for i in classes],
#                      columns = [i for i in classes])
# plt.figure(figsize = (12,7))
# sn.heatmap(df_cm, annot=True)
# plt.show()

# # Print recall, f-score, precision
# print(classification_report(y_true, y_pred))



