# Example from https://pytorch.org/hub/pytorch_vision_resnet/

# Create own dataset

import torch
import os
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import urllib
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.



model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model.eval()


# labels_map = {
#     0: "balloon",
#     1: "labrador retriever",
#     2: "sports car",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }


labels_map = {
    0: "balloon",
    1: "labrador retriever",
    2: "sports car"
}

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

batch_size=2
train_folder = "test1"
data_set = CustomImageDataset('train.csv', train_folder,transform=preprocess)
train_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)




# Training network

# test_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)
# num_epochs=2
# learning_rate = 1e-3
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# with torch.no_grad():
#     output = torch.nn.functional.softmax(model(train_loader), dim=1)
    
# results = utils.pick_n_best(predictions=output, n=5)
# print(results
# )





# Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # Train Network
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

#     print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")

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
#             f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
#         )

#     model.train()


# print("Checking accuracy on Training Set")
# check_accuracy(train_loader, model)

# print("Checking accuracy on Test Set")
# check_accuracy(test_loader, model)


