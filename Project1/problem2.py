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


# Init device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

# Init ResNet and related
model = models.resnet18(pretrained=False)

# modify convolution layer, input size is (7,7) instead of (28,28)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Modify number of classes/features, we only got 10 (0,1,2,...9)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features,10)
model = model.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)



# Init data and data_loaders
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



print(train_data)
print(train_data.data.size())

# Plot one data object from MNIST
plt.imshow(train_data.data[0], cmap='gray')
plt.title('%i' % train_data.targets[0])
plt.show()


# Plot multiple data objects from MNIST
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


train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
    
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)


loss_func = nn.CrossEntropyLoss()   
optimizer = optim.Adam(model.parameters(), lr = 0.01)   
num_epochs = 2

for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = loss_func(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")

# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)












# EXTRAS



# def compute_accuracy(model, data_loader, device):
#     correct_pred, num_examples = 0, 0
#     for i, (features, targets) in enumerate(data_loader):
            
#         features = features.to(device)
#         targets = targets.to(device)

#         logits, probas = model(features)
#         _, predicted_labels = torch.max(probas, 1)
#         num_examples += targets.size(0)
#         correct_pred += (predicted_labels == targets).sum()
#     return correct_pred.float()/num_examples * 100
    

# start_time = time.time()
# for epoch in range(NUM_EPOCHS):
#   model.train()
#   for batch_idx, (features, targets) in enumerate(train_loader):    
#     features = features.to(DEVICE)
#     targets = targets.to(DEVICE)
        
#     ### FORWARD AND BACK PROP
#     logits, probas = model(features)
#     cost = F.cross_entropy(logits, targets)
#     optimizer.zero_grad()
    
#     cost.backward()
    
#     ### UPDATE MODEL PARAMETERS
#     optimizer.step()
    
#     ### LOGGING
#     if not batch_idx % 50:
#       print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
#                 %(epoch+1, NUM_EPOCHS, batch_idx, 
#                 len(train_loader), cost))

#     model.eval()
#     with torch.set_grad_enabled(False): # save memory during inference
#       print('Epoch: %03d/%03d | Train: %.3f%%' % (
#               epoch+1, NUM_EPOCHS, 
#               compute_accuracy(model, train_loader, device=DEVICE)))
        
#     print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
# print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))




# predicted=[]
# with torch.no_grad():
#     n_correct=0
#     n_samples=0
#     for images,labels in test_dataloader:
#         images=images.reshape(-1,784)
#         output=Mnist_model(images) #applying the model we have built
#         labels=labels
#         _,prediction=torch.max(output,1)
#         predicted.append(prediction)
# print(prediction)