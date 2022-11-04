# Packages
import os
import time
import copy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torch import optim

import matplotlib.pyplot as plt
from PIL import Image

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sn

DEBUG=0


# Init device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

# Init ResNet and related
model = models.resnet18(pretrained=True)
# Freeze layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze and modify last layer
for param in model.fc.parameters():
    param.requires_grad = True


# modify convolution layer, input size is (7,7) instead of (28,28)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Modify number of classes/features, we only got 10 (0,1,2,...9)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features,10)
model = model.to(device)



# Initialize data
train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = transforms.ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = transforms.ToTensor()
)


# Get train data size
print(train_data)
print(train_data.data.size())

# Plot one data object from MNIST
plt.imshow(train_data.data[0], cmap='gray')
plt.title('%i' % train_data.targets[0])
plt.show()


# Plot multiple data objects from MNIST
if (DEBUG == 1):
    figure = plt.figure(figsize=(10, 8))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, label = train_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


# prepare loaders for training and testing
train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

# Get example batches
examples = enumerate(test_loader)
batch_idx, (example_data, example_labels) = next(examples)


# Plot the example data set
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_labels[i]))
  plt.xticks([])
  plt.yticks([])
plt.show()  


# Define loss, optimizer, num_epochs
loss_func = nn.CrossEntropyLoss()   
optimizer = optim.Adam(model.parameters(), lr = 0.001)   
num_epochs = 3


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


# Call functions and perform
train_model(model=model,num_epochs=num_epochs,loader=train_loader,loss_func=loss_func,optimizer=optimizer)


# Arrays for output of check_accuracy
y_pred = []
y_true = []


# Check accuracy of train and test data
print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
y_pred, y_true = check_accuracy(test_loader, model)



# Get classification for example data
with torch.no_grad():
  example_data_cuda = example_data.to(device=device)  
  output = model(example_data_cuda)

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
classes = ('0','1','2','3','4','5','6','7','8','9')
cf_matrix = confusion_matrix(y_true, y_pred)
cm = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize = (12,7))
sn.heatmap(cm, annot=True)
plt.show()

# Print recall, f-score, precision
print(classification_report(y_true, y_pred))