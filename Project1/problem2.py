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
import pandas as pd


# Init device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

# Init ResNet and related
model = models.resnet18(pretrained=True)

# modify convolution layer, input size is (7,7) instead of (28,28)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Modify number of classes/features, we only got 10 (0,1,2,...9)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features,10)
model = model.to(device)
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


# # Plot multiple data objects from MNIST
# figure = plt.figure(figsize=(10, 8))
# cols, rows = 5, 5
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(train_data), size=(1,)).item()
#     img, label = train_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(label)
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()


train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

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

for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, labels) in enumerate(train_loader):
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

# Check accuracy on training to see how good our model is, works
def check_accuracy(loader, model):
    predicted = []
    actual = []
    
    num_correct = 0
    num_samples = 0
    
    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device=device)
            labels = labels.to(device=device)

            scores = model(images)
            _, predictions = scores.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)
            temp = (torch.max(torch.exp(scores), 1)[1]).data.cpu().numpy()
            predicted.extend(temp)
            actual.extend(labels.data.cpu().numpy())
        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()
    return predicted, actual


y_pred = []
y_true = []


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
y_pred, y_true = check_accuracy(test_loader, model)




with torch.no_grad():
  example_data_cuda = example_data.to(device=device)  
  output = model(example_data_cuda)

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




# print(y_pred)
# print(y_true)

classes = ('0','1','2','3','4','5','6','7','8','9')

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.show()



print(classification_report(y_true, y_pred))