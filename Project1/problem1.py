# Packages
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
from torchvision import transforms
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform = transforms.Compose([            
#     transforms.Resize(256),                    
#     transforms.CenterCrop(224),                
#     transforms.ToTensor(),                     
#     transforms.Normalize(                      
#     mean=[0.485, 0.456, 0.406],                
#     std=[0.229, 0.224, 0.225]                  
# )])




# learning_rate = 1e-3
# num_epochs = 10
# batch_size = 2
# num_classes = 5



# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label




# train_data = CustomImageDataset(csv_file = 'a.csv', root_dir = 'train1',transform=transform)
# test_data = CustomImageDataset(csv_file = 'a.csv', root_dir = 'test1',transform=transform)
# train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
# test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True)

# model = torchvision.models.resnet50(preTrained=True)
# model.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# for epoch in range(num_epochs):
#     losses = []

#     for batch_idx, (data, targets) in enumerate(train_loader):
#         # Get data to cuda if possible
#         data = data.to(device=device)
#         targets = targets.to(device=device)

#         # forward
#         scores = model(data)
#         loss = criterion(scores, targets)

#         losses.append(loss.item())

#         # backward
#         optimizer.zero_grad()
#         loss.backward()

#         # gradient descent or adam step
#         optimizer.step()

#     print(f"Cost at epoch {epoch} is {sum(losses) / len(losses)}")


# # Check accuracy on training to see how good our model is
# def check_accuracy(loader, model):
#     num_correct = 0
#     num_samples = 0
#     model.eval()

#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device=device)
#             y = y.to(device=device)

#             scores = model(x)
#             _, predictions = scores.max(1)
#             num_correct += (predictions == y).sum()
#             num_samples += predictions.size(0)

#         print(
#             f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
#         )

#     model.train()


# print("Checking accuracy on Training Set")
# check_accuracy(train_loader, model)

# print("Checking accuracy on Test Set")
# check_accuracy(test_loader, model)