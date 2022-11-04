# Example from https://pytorch.org/hub/pytorch_vision_resnet/

# WORKS and PRETTY GOOD, multiple images

import torch
import os
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import seaborn as sn
import pandas as pd
import urllib
import cv2
import matplotlib.pyplot as plt
import numpy as np



# Init device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model = model.to(device)
model.eval()




# Transform input images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Folder names
test_folder = "dataset"
project_folder = os.getcwd()
test_folder = os.path.join(project_folder,test_folder)

# Arrays for predictions and actual values
y_pred = []
y_true = []


for folder in os.listdir(test_folder):
    sub_directory = (os.path.join(test_folder,folder))
    if (os.path.isdir(sub_directory) == False): continue
    # print(sub_directory)
    for filename in os.listdir(sub_directory):
        print(filename)
        label = folder
        filename_path = os.path.join(sub_directory,filename)

        # Evaluation
        original_img = Image.open(filename_path)
        input_tensor = preprocess(original_img)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        input_batch = input_batch.to(device)    # move batch to device

        with torch.no_grad():
            output = model(input_batch)
        
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Read the categories
        with open("imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
        # Show top categories per image
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        for i in range(top5_prob.size(0)):
            cv2.imshow('Original image', cv2.imread(filename_path))
            cv2.waitKey(500)
            print(categories[top5_catid[i]], top5_prob[i].item())
        print("\n")
        y_pred.append(categories[top5_catid[0]]) # Save Prediction
        y_true.append(label)


print(y_pred)
print(y_true)

# constant for classes
classes = ('balloon', 'Labrador retriever', 'barbell','ski', 'velvet')

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *5, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.show()
print(classification_report(y_true, y_pred))
