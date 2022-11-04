# Packages
import os
import time
import copy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision.utils import make_grid


from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torch import optim

import matplotlib.pyplot as plt
from PIL import Image

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sn
from torchvision.datasets import ImageFolder


DEBUG=0

# Process images
preprocess_alex = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

preprocess_res= transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Init device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

# Init models
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features,2)
model = model.to(device)

model2=models.alexnet(pretrained=True)
model2.classifier[4] = nn.Linear(4096,1024)
model2.classifier[6] = nn.Linear(1024,2)

# Define loss, optimizer, num_epochs
loss_func = nn.CrossEntropyLoss()   
optimizer = optim.Adam(model.parameters(), lr = 0.001)   
num_epochs = 3

# Prepare data
train_data = ImageFolder(root='dog_cat/training_set',transform=preprocess_res)
test_data = ImageFolder(root='dog_cat/test_set',transform=preprocess_res)

# DataLoaders
train_loader = DataLoader(train_data,batch_size=32,shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

# Get example batches
examples = enumerate(test_loader)
batch_idx, (example_data, example_labels) = next(examples)

print(train_data.classes)



# Helper functions

# Plot example data
def plot_example(data,labels):
# Plot the example data set
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        img = data[i][0]
        # img.thumbnail((256,256), Image.ANTIALIAS)
        plt.imshow(img, cmap='gray',interpolation='none')
        plt.title("Ground Truth: {}".format(labels[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()  

# Train model
def train_model(model,num_epochs,loader,loss_func,optimizer):
    for epoch in range(num_epochs):
        losses = []

        for batch_idx, (data, labels) in enumerate(loader):
            # Get data to cuda if possible
            data = data.to(device=device)
            labels = labels.to(device=device)

            # gradient descent or adam step
            optimizer.step()

            # forward
            scores = model(data)
            loss = loss_func(scores, labels)

            losses.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()  

        print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")
    print("Finished training")

# Calculate accuracy of model
def check_accuracy(loader, model):
    # Arrays for confusion matrix
    predicted = []
    actual = []
    
    # Keep track of correct output
    num_correct = 0
    num_samples = 0
    
    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            # Load data to cuda if possible
            images = images.to(device=device)
            labels = labels.to(device=device)

            scores = model(images)
            _, predictions = scores.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)
            temp = (torch.max(torch.exp(scores), 1)[1]).data.cpu().numpy() # for confusion matrix
            predicted.extend(temp) # for confusion matrix
            actual.extend(labels.data.cpu().numpy()) # for confusion matrix
        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()
    return predicted, actual

# Evaluate on one example from testset
def eval_example(exp_data):
    # Get classification for example data
    with torch.no_grad():
        exp_data_cuda = exp_data.to(device=device)  
        output = model(exp_data_cuda)

    # Plot output for example data
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

# Build confusion matrix
def conf_matrix(true,prediction):
    # Build confusion matrix
    classes = ('cats','dogs')
    cf_matrix = confusion_matrix(true, prediction)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *2, index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()

    # Print recall, f-score, precision
    print(classification_report(y_true, y_pred))

# Execution of helper functions
plot_example(example_data,example_labels)
train_model(model=model,num_epochs=num_epochs,loader=train_loader,loss_func=loss_func,optimizer=optimizer)

# Arrays for output of check_accuracy
y_pred = []
y_true = []


# Check accuracy of train and test data
print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
y_pred, y_true = check_accuracy(test_loader, model)

# Evaluate example
eval_example(example_data)
conf_matrix(y_true,y_pred)